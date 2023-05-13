# in this one I'm doing logit per logit rewards


from utils.tools import scores_to_proba_dists
from utils.game import init_game
from accelerate import init_empty_weights, Accelerator

import hydra
import torch
import numpy as np
import time
from tqdm import tqdm

# Accelerate
accelerator = Accelerator()


def perform_update(model, optimizer, lr_scheduler, contexts, **kwargs):
    current_process_buffer = {}
    for k in ["advantages", "returns", "logprobs"]:
        current_process_buffer[k] = kwargs[k][:]

    # use torch pad, add corresponding attention mask
    # contexts = torch.nn.utils.rnn.pad_sequence(
    #     contexts,
    #     batch_first=True,
    #     padding_value=model.tokenizer.pad_token_id,
    # )

    # left padding
    contexts = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(_c.flip(dims=[0])) for _c in contexts], batch_first=True
    ).flip(dims=[1])
    attention_mask = (contexts != model.tokenizer.pad_token_id).to(torch.float32)

    for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
        # Use LLM to compute again action probabilities and value
        for c_idx in range(0, len(contexts), kwargs["batch_size"]):
            with torch.no_grad():
                actions = model.generate(
                    {
                        "input_ids": contexts[c_idx : c_idx + kwargs["batch_size"]].to(
                            accelerator.device
                        ),
                        "attention_mask": attention_mask[
                            c_idx : c_idx + kwargs["batch_size"]
                        ].to(accelerator.device),
                    },
                    do_sample=True,
                    max_new_tokens=1,  # max_new_tokens,
                    prepare=False,
                )
            output = model(actions["text"])
            # :/ redundant with the generate function
            log_probs = output["logits"][:, :-1, :]
            entropy = scores_to_proba_dists(log_probs).entropy().mean()
            ratio = torch.exp(
                log_probs
                - torch.stack(
                    current_process_buffer["logprobs"][
                        c_idx : c_idx + kwargs["batch_size"]
                    ]
                )
            ).cpu()
            clip_adv = (
                torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]).T
                * current_process_buffer["advantages"][
                    c_idx : c_idx + kwargs["batch_size"]
                ]
            )
            value_loss = (
                (
                    actions["value"].cpu()
                    - current_process_buffer["returns"][
                        c_idx : c_idx + kwargs["batch_size"]
                    ]
                )
                ** 2
            ).mean()

            policy_loss = -(
                torch.min(
                    ratio.T
                    * (
                        current_process_buffer["advantages"][
                            c_idx : c_idx + kwargs["batch_size"]
                        ]
                    ),
                    clip_adv,
                )
            ).mean()

            # Compute final loss
            loss = (
                policy_loss
                - kwargs["entropy_loss_coef"] * entropy
                + kwargs["value_loss_coef"] * value_loss
            )

            # Optimize
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        accelerator.print(
            "policy_loss = ",
            policy_loss.item(),
            "entropy = ",
            entropy.item() * kwargs["entropy_loss_coef"],
            "value_loss = ",
            value_loss.item() * kwargs["value_loss_coef"],
            "advantages = ",
            current_process_buffer["advantages"].mean().item(),
            "returns = ",
            current_process_buffer["returns"].mean().item(),
            "ratio = ",
            ratio.mean().item(),
            "clip_adv = ",
            clip_adv.mean().item(),
        )
    if kwargs["save_after_update"]:
        accelerator.print("Saving model...")
        # TODO with accelerator (deal with at the wrapper level)
        model.save_pretrained(kwargs["log_dir"] + "/model")
        accelerator.print(f"Model saved in {kwargs['log_dir']}/model")

    return {"loss": loss}


@hydra.main(
    config_path="/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning",
    config_name="local_gpu_config",
    version_base="1.1",
)
def main(config_args):
    wmodel, env, optimizer, lr_scheduler, buf, pad_token = init_game(
        config_args, accelerator
    )
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
                # Her introduce a function that looks for the stop word and replaces all lolgits after it with the pad token
            # logits should also be masked, as should values (or any other output)
            for i, _a in enumerate(a["text"]):
                if " Answer:" in _a:
                    _cut_a = _a.split(" Answer:")[0]
                    if _cut_a != _a:
                        a["text"][i] = _cut_a + " Answer:"
                        _len = len(wmodel.tokenizer(_cut_a)["input_ids"])
                        _pad_log = torch.zeros(
                            config_args.rl_script_args.max_new_tokens - _len,
                            a["logits"][i].shape[-1],
                        ).to(accelerator.device)
                        _pad_log[:, pad_token] = 1
                        a["logits"][i] = torch.concat(
                            [
                                a["logits"][i][:_len],
                                _pad_log,
                            ]
                        )
            accelerator.print(
                f"Generated turn {infos['turn']}, max_new_tokens: {config_args.rl_script_args.max_new_tokens}, a: {a['text']} expected: {env.batch[2]}"
            )

            # score the right answer, will be used as part of the reward
            if infos["turn"] == 1:
                with torch.no_grad():
                    # to deal with tiny values we use logsoftmax instead of softmax, this is more of a penalty to minimize than a reward to maximize
                    ans_score = wmodel.score(
                        context=o, out_logs=a["logits"], expected=env.batch[2]
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
                for j in range(len(a["out_sequence"][i])):
                    reward = ans_score[i][j] * config_args.rl_script_args.score_coef
                    reward += r[i] if j == len(a["out_sequence"][i]) - 1 else 0
                    # TODO: add curiosity reward
                    buf.store(
                        torch.concat(
                            [a["in_sequence"][i], a["out_sequence"][i][: j + 1]], dim=0
                        ),
                        reward,
                        a["value"][i][j],
                        a["logits"][i][j],
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
                # base_prompt = o[0]

        # Perform PPO update!
        save_model = (
            epoch % config_args.rl_script_args.save_freq == 0
            or epoch == config_args.rl_script_args.epochs - 1
        ) and epoch != 0
        # buf.to_txt(config_args.rl_script_args.log_dir + "/buffer.txt")
        collected_trajectories = buf.get()
        # print(batch_actions)
        update_results = perform_update(
            wmodel,
            optimizer,
            lr_scheduler,
            contexts=collected_trajectories["obs"],
            returns=collected_trajectories["ret"],
            advantages=collected_trajectories["adv"],
            logprobs=collected_trajectories["logp"],
            answers=env.batch[2],
            clip_eps=config_args.rl_script_args.clip_eps,
            entropy_loss_coef=config_args.rl_script_args.entropy_loss_coef,
            ce_coef=config_args.rl_script_args.ce_coef,
            value_loss_coef=config_args.rl_script_args.value_loss_coef,
            cur_coef=config_args.rl_script_args.cur_coef,
            ppo_epochs=config_args.rl_script_args.ppo_epochs,
            save_after_update=save_model,
            log_dir=config_args.rl_script_args.log_dir,
            batch_size=config_args.rl_script_args.batch_size,
        )
        # print(f"Update results: {update_results}")
    # lm_server.close()


if __name__ == "__main__":
    main()
