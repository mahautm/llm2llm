# in this one I'm doing logit per logit rewards
from utils.tools import scores_to_proba_dists, pad_merge, discount_cumsum
from utils.game import init_game, evaluate
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
    froz_model, wmodel, env, venv, optimizer, lr_scheduler, _, pad_token = init_game(
        config_args, accelerator
    )
    space_token = 1437 # hard coded attempt at making the space token less important - kicks in at opt-350m-fromin

    # instead of a critic REINFORCE can use as a baseline the average of previous returns
    if config_args.rl_script_args.value_loss_coef == 0:
        av_val = 0
        nvals = 1
    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    infos = {"turn": 0}

    # Main loop: collect experience in env and update/log each epoch
    accelerator.print(f"Launching training...")
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        if (
            config_args.rl_script_args.affix_num_steps is not None
            and config_args.rl_script_args.affix_num_steps < epoch
        ):
            config_args.rl_script_args.affixes = None
            env.affix = None

        for t in range(config_args.rl_script_args.steps_per_epoch):
            with torch.no_grad():
                a = wmodel.generate(
                    o,
                    max_new_tokens=config_args.rl_script_args.max_new_tokens,
                    pad_token_id=pad_token,
                    do_sample=True,
                    top_k=config_args.rl_script_args.top_k,
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
            _max_size = min(
                config_args.rl_script_args.max_new_tokens, out["logits"].shape[1]
            )
            logits = out["logits"][
                :, :_max_size, :
            ]  # check why -1 instead of max_new_tokens is not working
            seq = out["sequences"][:, :_max_size]
            # introducing kl_divergence to original model (& NOT to last iteration -- that way we do not stray too far from natural language)
            with torch.no_grad():
                klogits = froz_model(a["text"])["logits"][:, :_max_size, :]
            accelerator.print(
                f"Generated turn {infos['turn']}, max_new_tokens: {_max_size}, a: {a['text']} expected: {env.batch[2]}"
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
                if config_args.rl_script_args.cur_coef != 0:
                    old_cur = a["curiosity"]
                if config_args.rl_script_args.value_loss_coef != 0:
                    # TODO: check size
                    old_val = a["value"][:, : old_logits.shape[1]].squeeze()
                # curiosity isn't initially computed. This is uncortunately due to sentence based curiosity and should be completely reframed
                # no intermediate reward
                # second part of the conversation has access to entire history
                if config_args.rl_script_args.affixes is not None:
                    # TODO: make as a parameter and not a hardcoded value
                    o = [o[i] + a["text"][i] + new_o[i] + "You:" for i in range(len(o))]
                else:
                    o = [o[i] + a["text"][i] + new_o[i] for i in range(len(o))]
            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            # epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal:
                # save and log
                with open(config_args.rl_script_args.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"Episode {n_episodes} - GPU {accelerator.device} - Acc {ep_ret}/{len(o)}\n"
                    )
                    for k, v in env.render().items():
                        f.write(f"{k} : {v}\n")
                n_episodes += 1
                accelerator.print(
                    # f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)}"  # ans_score: {ans_score * config_args.rl_script_args.score_coef}"
                    f"Episode {n_episodes} --> Acc: {sum(r)/len(r)} batch_size: {len(r)}"
                )

                # prepare data for loss
                # prepare logits
                # TODO: all pad tokens should be ignored and therefore replaced by ones in the logits?

                # in some (mostly nonsense) cases, the first sequence will stop being the shortest
                if len(old_logits[0]) >= len(logits[0]):
                    print("swapping")
                    logits, old_logits = old_logits, logits
                    klogits, old_klogits = old_klogits, klogits
                    seq, old_seq = old_seq, seq
                    if config_args.rl_script_args.value_loss_coef != 0:
                        val = a["value"][:, : logits.shape[1]].squeeze()
                        val, old_val = old_val, val

                logits = pad_merge(old_logits, logits)
                # prepare klogits
                klogits = pad_merge(old_klogits, klogits)
                # prepare sequences
                seq = pad_merge(old_seq, seq, pad_value=pad_token)
                # prepare values (either critic or baseline = average return)
                if config_args.rl_script_args.value_loss_coef != 0:
                    val = pad_merge(old_val, val)
                else:
                    # make tensor with average reward so for each element in batch
                    val = torch.ones(2 * len(r), 1, device=accelerator.device) * av_val
                    # keep a running average of the average reward
                    av_val = sum(r) / len(r) * 1 / nvals + av_val * (1 - 1 / nvals)
                    nvals += 1
                    # discount as would be done for real rewards
                    val = torch.functional.F.pad(
                        val,
                        (logits.shape[1] - val.shape[-1], 0, 0, 0),
                        "constant",
                        0,
                    )
                    val = discount_cumsum(
                        val,
                        config_args.rl_script_args.gamma,
                    )
                # prepare rewards
                # no old rewards, we repeat turn2 rewards at turn1, fill additional space with 0s
                r = torch.tensor(r, device=accelerator.device).unsqueeze(1)
                r = torch.functional.F.pad(
                    r,
                    (logits.shape[1] - r.shape[-1], 0, 0, 0),
                    "constant",
                    0,
                ).repeat(2, 1)
                ret = discount_cumsum(
                    r,
                    config_args.rl_script_args.gamma,
                )
                if config_args.rl_script_args.cur_coef:
                    cur = a["curiosity"]
                    cur = pad_merge(old_cur, cur)
                    cur = torch.functional.F.pad(
                        cur,
                        (logits.shape[1] - cur.shape[1], 0, 0, 0),
                        "constant",
                        0,
                    )
                    ret = ret + cur * config_args.rl_script_args.cur_coef
                # compute advantage
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
                # log_probs = torch.gather(
                #     logits.log_softmax(-1), dim=2, index=seq.unsqueeze(2)
                # ).squeeze()
                log_probs = dist.log_prob(seq)
                # add normalisation
                mask = (seq != pad_token).float()
                space_mask = (seq != space_token).float()
                log_probs *= mask * space_mask
                # log_probs = log_probs.sum(1) / msg_lengths

                weighted_kl_div = kl_div * config_args.rl_script_args.kl_loss_coeff
                if config_args.rl_script_args.value_loss_coef != 0:
                    weighted_value_loss = (
                        (ret.detach() - val) ** 2
                    ) * config_args.rl_script_args.value_loss_coef
                else:
                    weighted_value_loss = torch.zeros_like(val)
                # batch normalisation of advantage trick
                # adv = (adv - adv.mean()) / (adv.std() + 1e-12)
                policy_loss = -adv * log_probs
                optimized_loss = (
                    policy_loss
                    - weighted_entropy
                    + weighted_value_loss
                    + weighted_kl_div
                ).mean()
                if config_args.rl_script_args.cur_coef:
                    curiosity_loss = wmodel.curiosity_module.get_loss()
                    optimized_loss += curiosity_loss.mean()

                accelerator.backward(optimized_loss)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                # log
                log = f"return: {ret.mean().item()} - val: {val.mean().item()} - advantage: {adv.mean()} - policy_loss: {policy_loss.mean().item()} - entropy: {entropy.mean().item()} - weighted_entropy: {weighted_entropy.mean().item()} - weighted_value_loss: {weighted_value_loss.mean().item()} - weighted_kl_div: {weighted_kl_div.mean().item()} - kl_div: {kl_div.mean().item()} - lr: {optimizer.param_groups[0]['lr']} - optimized_loss: {optimized_loss.mean().item()}"
                if config_args.rl_script_args.cur_coef != 0:
                    log += f" - curiosity: {curiosity_loss.mean().item()}"
                accelerator.print(log)
                if epoch % config_args.rl_script_args.validation_interval == 0:
                    evaluate(
                        wmodel,
                        venv,
                        pad_token,
                        accelerator,
                        config_args.rl_script_args.steps_per_epoch,
                        config_args.rl_script_args.max_new_tokens,
                        config_args.rl_script_args.log_file,
                        n_episodes,
                    )
                # reset
                o, ep_ret, ep_len = env.reset(), 0, 0
                infos = {"turn": 0}


if __name__ == "__main__":
    main()
