# in this one I'm doing logit per logit rewards
from utils.tools import scores_to_proba_dists
from utils.game import init_game
from accelerate import Accelerator

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
    wmodel, env, optimizer, lr_scheduler, buf, pad_token = init_game(
        config_args, accelerator
    )
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
            for i, _a in enumerate(a["text"]):
                if " Answer:" in _a:
                    _cut_a = _a.split(" Answer:")[0]
                    if _cut_a != _a:
                        a["text"][i] = _cut_a + " Answer:"
            # get the gradients
            output = wmodel(a["text"])
            log_probs = output["logits"][:, :-1, :]
            accelerator.print(
                f"Generated turn {infos['turn']}, max_new_tokens: {config_args.rl_script_args.max_new_tokens}, a: {a['text']} expected: {env.batch[2]}"
            )

            # score the right answer, will be used as part of the reward
            if infos["turn"] == 1:
                with torch.no_grad():
                    # to deal with tiny values we use logsoftmax instead of softmax, this is more of a penalty to minimize than a reward to maximize
                    ans_score = wmodel.score(
                        context=o, out_logs=log_probs, expected=env.batch[2]
                    )
            else:
                # no reward before episode ends
                ans_score = [[0] * len(a["out_sequence"][0])] * len(a["out_sequence"])

            new_o, r, d, infos = env.step(a["text"])
            ep_ret += sum(r)
            ep_len += 1

            # Store experience to replay buffer
            # each added token is considered a step
            for i, obs in enumerate(o):
                # all sequence sizes should match though
                # there is probably a way to batch this
                assert len(a["out_sequence"][i]) == len(log_probs[i])
                for j in range(len(a["out_sequence"][i])):
                    reward = ans_score[i][j] * config_args.rl_script_args.score_coef
                    reward += r[i] if j == len(a["out_sequence"][i]) - 1 else 0
                    # TODO: add curiosity reward
                    buf.store(
                        None,
                        reward,
                        a["value"][i][j],
                        log_probs[i][j],
                    )

            if new_o[0] is not None:
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

                buf.finish_logit_path(len(o), len(a["out_sequence"][0]))
                if terminal:
                    n_episodes += 1
                    accelerator.print(
                        f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)} ans_score: {ans_score * config_args.rl_script_args.score_coef}"
                    )
                o, ep_ret, ep_len = env.reset(), 0, 0
                infos = {"turn": 0}
                traj = buf.get()

                # Perform REINFORCE update! as in https://github.com/robertodessi/EGG/blob/rll/egg/zoo/emergent_captioner/finetuning/game.py
                dist = scores_to_proba_dists(traj["logp"])
                entropy = dist.entropy().mean()
                weighted_entropy = (
                    entropy * config_args.rl_script_args.entropy_loss_coef
                )
                # weighted_kl_div = kl_div * config_args.rl_script_args.kl_div_coeff
                weighted_value_loss = (
                    (traj["ret"].detach() - traj["adv"]) ** 2
                ).mean() * config_args.rl_script_args.value_loss_coef
                policy_loss = (
                    (traj["ret"].detach() - traj["adv"]) * traj["logp"]
                ).mean()

                optimized_loss = policy_loss - weighted_entropy + weighted_value_loss
                accelerator.backward(optimized_loss)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()


if __name__ == "__main__":
    main()
