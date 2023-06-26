# Calls PPO to train a language model to earn rewards from a given environment
# Update from previous versions:
#   Provides per token rewards and loss
#   Each step is ppo-constrained to never be too far from the previous one
#   Each token is also constrained to be close to the original language model (kl divergence)
# Requires
#   game.py, which loads models and environment (game may use deepspeed, which is why we launch with accelerate)
#   utils.tools.py for some operations (discounted cumulative reward)
#   config file in yaml format to set all parameters (see local_gpu_config.yaml for an example)
#
# example call: 
# NCCL_P2P_DISABLE='1' NCCL_IB_DISABLE='1' accelerate launch --config_file=/shared_dir/exps/boring_auxiliary3/4/accelerate_config.yaml PPO_finetuning/logit_ppo.py --config-path=/shared_dir/exps/boring_auxiliary3/4 --config-name='config'
# Another way to call is to use sweeper.py to go through a grid of hyperparameters

from utils.tools import scores_to_proba_dists, pad_merge, discount_cumsum
from utils.game import init_game, evaluate
from accelerate import Accelerator
from torch.distributions import Categorical
import hydra
import torch
import numpy as np
import time
from tqdm import tqdm
import copy

# Accelerate
accelerator = Accelerator()

def generate_samples(env, o, froz_model, wmodel, config_args, accelerator, pad_token, epoch, av_val, nvals, verbose=False):
    # Prepare for interaction with environment
    ep_ret, ep_len, infos = 0, 0, {"turn": 0}
    # Main loop: collect experience in env and update/log each epoch
    if (
        config_args.rl_script_args.affix_num_steps is not None
        and config_args.rl_script_args.affix_num_steps < epoch
    ):
        config_args.rl_script_args.affixes = None
        env.affix = None
    d = False
    while not d:
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
        logits = out["logits"][:, :_max_size, :]
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
            else:
                old_val=None
            # curiosity isn't initially computed. This is uncortunately due to sentence based curiosity and should be completely reframed
            # no intermediate reward
            # second part of the conversation has access to entire history
            if config_args.rl_script_args.affixes is not None:
                # TODO: make as a parameter and not a hardcoded value
                o = [o[i] + a["text"][i] + new_o[i] + "You:" for i in range(len(o))]
            else:
                o = [o[i] + a["text"][i] + new_o[i] for i in range(len(o))]
    if config_args.rl_script_args.value_loss_coef != 0:
        val =  a["value"][:, : old_logits.shape[1]].squeeze()
    else:
        val = None
    ans_tokens = wmodel.tokenizer(env.batch[2], return_tensors="pt", padding=True)["input_ids"].squeeze().to(accelerator.device)
    logits, klogits, seq, val, ret, av_val, nvals= format_data(config_args, accelerator, pad_token, old_logits, logits, old_klogits, klogits, old_seq, seq, old_val, val, r, ans_tokens, av_val, nvals)
    
    if verbose:
        # save and log
        with open(config_args.rl_script_args.log_file, "a", encoding="utf-8") as f:
            f.write(
                f"Episode {epoch} - GPU {accelerator.device} - Acc {ep_ret}/{len(o)}\n"
            )
            for k, v in env.render().items():
                f.write(f"{k} : {v}\n")
        accelerator.print(
            # f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)}"  # ans_score: {ans_score * config_args.rl_script_args.score_coef}"
            f"Episode {epoch} --> Acc: {sum(r)/len(r)} batch_size: {len(r)}"
        )
    return logits, klogits, seq, val, ret, av_val, nvals

def format_data(config_args, accelerator, pad_token, old_logits, logits, old_klogits, klogits, old_seq, seq, old_val, val, r, ans_tokens, av_val, nvals):
    # prepare data for loss
    # prepare logits
    # TODO: all pad tokens should be ignored and therefore replaced by ones in the logits?

    # now we compute the auxiliary loss, that assigns a reward equal to the probability of the correct answer being the first token
    # TODO: this score function should go in the model (Str2Str in game.py)
    r = torch.tensor(r, device=accelerator.device).float()
    if config_args.rl_script_args.score_coef != 0:
        mask = torch.zeros_like(logits)
        # detect pad tokens
        mask[logits.argmax(2) != 0] = 1
        print("mask", mask.sum(), pad_token)
        # keep only as many non-pad tokenas as there are in the answer
        mask = mask.cumsum(1)
        mask[mask > ans_tokens.shape[0]] = 0
        print(ans_tokens.shape[1], "mask", mask.sum())
        # keep only non 0 values from masked logits
        mask = mask * logits.softmax(2)
        # reduce dimension to non-pad tokens
        mask = mask.sum(1)
        print("mask", mask.sum())
        # gather the probability of the correct answer
        auxiliary = mask.gather(-1, ans_tokens)
        print("auxiliary", auxiliary.sum())
        auxiliary = auxiliary.sum(1) * config_args.rl_script_args.score_coef
        r += auxiliary

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
    r = torch.functional.F.pad(
        r.unsqueeze(1),
        (logits.shape[1] - 1, 0, 0, 0),
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
    return logits, klogits, seq, val, ret, av_val, nvals

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
    for epoch in range(config_args.rl_script_args.epochs):
        # reset
        o = env.reset()
        with torch.no_grad():
            logits, klogits, seq, val, ret, av_val, nvals = generate_samples(env, copy.deepcopy(o), froz_model, wmodel, config_args, accelerator, pad_token, epoch, av_val, nvals, verbose=True)

        # Perform PPO update!
        for ppo_update in range(config_args.rl_script_args.ppo_updates):
            env.soft_reset()
            # compute new log_probs
            ppo_logits, ppo_klogits, ppo_seq, ppo_val, ppo_ret, _, _ = generate_samples(env, copy.deepcopy(o), froz_model, wmodel, config_args, accelerator, pad_token, epoch, av_val, nvals)
            # compute advantage
            ppo_adv = ppo_ret - ppo_val
            # pad ppo_logits to match logits
            ppo_logits = torch.functional.F.pad(
                ppo_logits,
                (0, logits.shape[1] - ppo_logits.shape[1], 0, 0),
                "constant",
                0,
            )
            # compute ratio
            ratio = torch.exp(
                ppo_logits.log_softmax(-1).gather(
                    dim=2, index=seq.unsqueeze(2)
                ).squeeze()
                - logits.log_softmax(-1).gather(
                    dim=2, index=seq.unsqueeze(2)
                ).squeeze()
            )
            # compute surrogate loss
            surr1 = ratio * ppo_adv
            surr2 = (
                torch.clamp(ratio, 1 - config_args.rl_script_args.clip_eps, 1 + config_args.rl_script_args.clip_eps)
                * ppo_adv
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            dist = Categorical(logits=ppo_logits)
            entropy = dist.entropy()
            weighted_entropy = (
                entropy * config_args.rl_script_args.entropy_loss_coef
            )
            # lets calculate the KL divergence between the original (froz_model) and new policies
            kl_div = torch.nn.functional.kl_div(
                ppo_logits.log_softmax(-1),
                ppo_klogits.log_softmax(-1),
                log_target=True,
                reduction="none",
            )
            kl_div = kl_div.sum(dim=-1)

            # compute normalized log_probs of generated captions
            # log_probs = torch.gather(
            #     logits.log_softmax(-1), dim=2, index=seq.unsqueeze(2)
            # ).squeeze()
            # log_probs = dist.log_prob(seq)
            # add normalisation
            # mask = (seq != pad_token).float()
            # space_mask = (seq != space_token).float()
            # log_probs *= mask * space_mask
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
            log = f"return: {ret.mean().item()} - val: {val.mean().item()} - advantage: {ppo_adv.mean()} - policy_loss: {policy_loss.mean().item()} - entropy: {entropy.mean().item()} - weighted_entropy: {weighted_entropy.mean().item()} - weighted_value_loss: {weighted_value_loss.mean().item()} - weighted_kl_div: {weighted_kl_div.mean().item()} - kl_div: {kl_div.mean().item()} - lr: {optimizer.param_groups[0]['lr']} - optimized_loss: {optimized_loss.mean().item()} - ratio: {ratio.mean().item()} - surr1: {surr1.mean().item()} - surr2: {surr2.mean().item()} - ppo_adv: {ppo_adv.mean().item()}"
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
                    epoch,
                )


if __name__ == "__main__":
    main()
