# in this one I'm doing logit per logit rewards
from utils.tools import scores_to_proba_dists, wt_cumsum, pad_merge
from utils.game import init_game
from accelerate import Accelerator
from torch.distributions import Categorical

import hydra
import torch
import numpy as np
import time
from tqdm import tqdm

# Accelerate
accelerator = Accelerator()


@hydra.main(
    config_path="/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning",
    config_name="local_gpu_config",
    version_base="1.1",
)
def main(config_args):
    froz_model, wmodel, env, optimizer, lr_scheduler, buf, pad_token = init_game(
        config_args, accelerator
    )
    # instead of a critic REINFORCE can use as a baseline the average of previous returns
    if config_args.rl_script_args.value_loss_coef == 0:
        av_val = 0
        nvals = 1
    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    infos = {"turn": 0}

    # Main loop: collect experience in env and update/log each epoch
    accelerator.print(f"Done. Training...")
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        batch_actions = None
        for t in range(config_args.rl_script_args.steps_per_epoch):
            with torch.no_grad():
                a = wmodel.generate(
                    o,
                    max_new_tokens=config_args.rl_script_args.max_new_tokens,
                    pad_token_id=pad_token,
                )
                # Here introduce a function that looks for the stop word and replaces all lolgits after it with the pad token
            # logits should also be masked, as should values (or any other output)
            if infos["turn"] == 0:
                for i, _a in enumerate(a["text"]):
                    if " Answer:" in _a:
                        _cut_a = _a.split(" Answer:")[0]
                        if _cut_a != _a:
                            a["text"][i] = _cut_a + " Answer:"
                # get the gradients
                out = wmodel(a["text"], return_sequences=True)
                logits = out["logits"][:, :-1, :]
                seq = out["sequences"][:, :-1]
                # introducing kl_divergence to original model (& NOT to last iteration -- that way we do not stray too far from natural language)
                with torch.no_grad():
                    klogits = froz_model(a["text"])["logits"][:, :-1, :]
            accelerator.print(
                f"Generated turn {infos['turn']}, max_new_tokens: {config_args.rl_script_args.max_new_tokens}, a: {a['text']} expected: {env.batch[2]}"
            )

            # send query to environment and get response
            new_o, r, d, infos = env.step(a["text"])
            ep_ret += sum(r)
            ep_len += 1

            if new_o[0] is not None:
                # Store msg1 before rerunning the model to get msg2
                old_seq = seq
                old_logits = logits
                old_klogits = klogits
                if config_args.rl_script_args.value_loss_coef != 0:
                    old_val = a["value"][:, : old_logits.shape[1]].squeeze()
                # no intermediate reward
                # second part of the conversation has access to entire history
                o = [o[i] + a["text"][i] + new_o[i] + "You:" for i in range(len(o))]
            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            # epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal:
                # save and log
                with open(config_args.rl_script_args.log_file, "a") as f:
                    f.write(
                        f"Episode {n_episodes} - GPU {accelerator.device} - Acc {ep_ret}/{len(o)}\n"
                    )
                    for k, v in env.render().items():
                        f.write(f"{k} : {v}\n")
                n_episodes += 1
                accelerator.print(
                    f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)}"  # ans_score: {ans_score * config_args.rl_script_args.score_coef}"
                )

                # prepare data for loss
                # prepare logits
                # TODO: all pad tokens should be ignored and therefore replaced by ones in the logits?
                logits = pad_merge(old_logits, logits)
                # prepare klogits
                klogits = pad_merge(old_klogits, klogits)
                # prepare sequences
                seq = pad_merge(old_seq, seq, pad_value=pad_token)
                # prepare values (either critic or baseline = average return)
                if config_args.rl_script_args.value_loss_coef != 0:
                    val = pad_merge(old_val, a["value"][:, : logits.shape[1]].squeeze())
                else:
                    # make tensor with average reward so for each element in batch
                    val = torch.ones(len(r)) * av_val
                    # keep a running average of the average reward
                    av_val = sum(r) / len(r) * 1 / nvals + av_val * (1 - 1 / nvals)
                    nvals += 1
                    # discount as would be done for real rewards
                    val = torch.functional.F.pad(
                        val.unsqueeze(1),
                        (0, logits.shape[1] - 1, logits.shape[0] - len(r), 0),
                        "constant",
                        0,
                    )
                    val = (
                        wt_cumsum(
                            val.flatten(),
                            config_args.rl_script_args.gamma,
                        )
                        .reshape(-1, val.shape[1])
                        .to(accelerator.device)
                    )
                # prepare rewards
                # no old rewards, we repeat turn2 rewards at turn1, fill additional space with 0s
                r = torch.tensor(r).repeat(2).unsqueeze(1)
                r = torch.functional.F.pad(
                    r,
                    (0, logits.shape[1] - 1, 0, 0),
                    "constant",
                    0,
                )
                ret = (
                    wt_cumsum(
                        r.flatten(),
                        config_args.rl_script_args.gamma,
                    )
                    .reshape(-1, r.shape[1])
                    .to(accelerator.device)
                )

                adv = ret - val

                # Perform REINFORCE update! as in https://github.com/robertodessi/EGG/blob/rll/egg/zoo/emergent_captioner/finetuning/game.py
                dist = Categorical(logits=logits)
                entropy = dist.entropy()
                weighted_entropy = (
                    entropy * config_args.rl_script_args.entropy_loss_coef
                )
                # lets calculate the KL divergence between the original (froz_model) and new policies
                kl_div = torch.nn.functional.kl_div(
                    logits.log_softmax(-1),
                    klogits.log_softmax(-1),
                    log_target=True,
                    reduction="none",
                )
                kl_div = kl_div.sum(dim=-1)

                # compute normalized log_probs of generated captions
                log_probs = torch.gather(
                    logits, dim=2, index=seq.unsqueeze(2)
                ).squeeze()
                # log_probs *= mask
                # log_probs = log_probs.sum(1) / msg_lengths

                weighted_kl_div = kl_div * config_args.rl_script_args.kl_loss_coeff
                weighted_value_loss = (
                    (ret.detach() - val) ** 2
                ) * config_args.rl_script_args.value_loss_coef
                # batch normalisation of advantage trick
                adv = (adv - adv.mean()) / (adv.std() + 1e-12)
                policy_loss = adv * log_probs
                optimized_loss = (
                    policy_loss
                    - weighted_entropy
                    + weighted_value_loss
                    + weighted_kl_div
                ).mean()

                accelerator.backward(optimized_loss)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                # log
                accelerator.print(
                    f"return: {ret.mean().item()} - val: {val.mean().item()} - advantage: {adv.mean()} - policy_loss: {policy_loss.mean().item()} - entropy: {entropy.mean().item()} - weighted_entropy: {weighted_entropy.mean().item()} - weighted_value_loss: {weighted_value_loss.mean().item()} - weighted_kl_div: {weighted_kl_div.mean().item()} - kl_div: {kl_div.mean().item()} - lr: {optimizer.param_groups[0]['lr']}"
                )
                # reset
                o, ep_ret, ep_len = env.reset(), 0, 0
                infos = {"turn": 0}


if __name__ == "__main__":
    main()
