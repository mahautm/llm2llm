# in this one I'm doing logit per logit rewards
from utils.tools import scores_to_proba_dists, pad_merge, discount_cumsum
from utils.game import init_game, evaluate
from accelerate import Accelerator
from torch.distributions import Categorical
import hydra
import torch
import numpy as np
import time
import re
from tqdm import tqdm

# Accelerate
accelerator = Accelerator()


@hydra.main(
    config_path="/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning",
    config_name="local_gpu_config",
    version_base="1.1",
)
def main(config_args):
    froz_model, wmodel, _, _, optimizer, lr_scheduler, _, pad_token = init_game(
        config_args, accelerator
    )
    space_token = 1437 # hard coded attempt at making the space token less important - kicks in at opt-350m-min9
    # instead of a critic REINFORCE can use as a baseline the average of previous returns
    if config_args.rl_script_args.value_loss_coef == 0:
        av_val = 0
        nvals = 1

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    o = ["prompt: "] * config_args.rl_script_args.batch_size
    for epoch in range(config_args.rl_script_args.epochs):
        with torch.no_grad():
            a = wmodel.generate(
                o,
                max_new_tokens=config_args.rl_script_args.max_new_tokens,
                pad_token_id=pad_token,
                do_sample=True,
                top_k=50,
            )
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
        accelerator.print(f"max_new_tokens: {_max_size}, a: {a['text']}")
        r = [
            sum(1 for _ in re.finditer(r"\b%s\b" % re.escape("the"), text.lower()))
            for text in a["text"]
        ]
        # save and log
        n_episodes += 1
        accelerator.print(
            # f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)}"  # ans_score: {ans_score * config_args.rl_script_args.score_coef}"
            f"Episode {n_episodes} --> Acc: {sum(r)/len(r)} batch_size: {len(r)}"
        )

        # prepare data for loss
        # prepare logits
        # TODO: all pad tokens should be ignored and therefore replaced by ones in the logits?

        # make tensor with average reward so for each element in batch
        val = torch.ones(len(r), 1, device=accelerator.device) * av_val
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
        )
        ret = discount_cumsum(
            r,
            config_args.rl_script_args.gamma,
        )
        # compute advantage
        adv = ret - val

        # Perform REINFORCE update! as in https://github.com/robertodessi/EGG/blob/rll/egg/zoo/emergent_captioner/finetuning/game.py
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        weighted_entropy = entropy * config_args.rl_script_args.entropy_loss_coef
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
        # ).squeeze(2)
        log_probs = dist.log_prob(seq)
        # add normalisation
        mask = (seq != pad_token).float()
        space_mask = (seq != space_token).float()

        print(sum(mask))
        log_probs *= mask * space_mask
        print(log_probs)
        # log_probs = log_probs.sum(1) / msg_lengths

        weighted_kl_div = kl_div * config_args.rl_script_args.kl_loss_coeff
        # batch normalisation of advantage trick
        # adv = (adv - adv.mean()) / (adv.std() + 1e-12)
        policy_loss = adv * log_probs
        optimized_loss = (policy_loss - weighted_entropy + weighted_kl_div).mean()

        accelerator.backward(optimized_loss)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # log
        log = f"return: {ret.mean().item()} - log_probs: {log_probs.mean().item()} - val: {val.mean().item()} - advantage: {adv.mean()} - policy_loss: {policy_loss.mean().item()} - entropy: {entropy.mean().item()} - weighted_entropy: {weighted_entropy.mean().item()} - weighted_kl_div: {weighted_kl_div.mean().item()} - kl_div: {kl_div.mean().item()} - lr: {optimizer.param_groups[0]['lr']} - optimized_loss: {optimized_loss.mean().item()}"
        accelerator.print(log)


if __name__ == "__main__":
    main()
