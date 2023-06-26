from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_com_env_text import LLMComEnvText
from utils.tools import scores_to_proba_dists
from utils.game import init_game
from accelerate import Accelerator
import hydra

import torch
import numpy as np
import time
from tqdm import tqdm


accelerator = Accelerator()


def perform_update(model, optimizer, lr_scheduler, contexts, **kwargs):
    # def perform_update(model, optimizer, contexts, **kwargs):
    # TODO : get pad token
    # pad_token = 50256
    max_new_tokens = 30
    current_process_buffer = {}
    for k in ["advantages", "returns", "logprobs", "answers"]:
        current_process_buffer[k] = kwargs[k][:]
    # extension of advantages and returns with discounted reward to match the length of the logprobs

    for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
        # _ = model(["Initialising model"])

        # Use LLM to compute again action probabilities and value
        with torch.no_grad():
            actions = model.generate(
                contexts,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
        output = model(actions["text"])
        log_probs = output["logits"][:, :-1, :]
        entropy = scores_to_proba_dists(log_probs).entropy().mean()
        ratio = torch.exp(
            log_probs - torch.stack(current_process_buffer["logprobs"])
        ).cpu()
        clip_adv = (
            torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]).T
            * current_process_buffer["advantages"]
        )
        value_loss = (
            # TODO : uniformize the way I bring things to cpu. Why is this the only one that needs it?
            (actions["value"].cpu() - current_process_buffer["returns"])
            ** 2
        ).mean()

        policy_loss = -(
            torch.min(
                ratio.T * (current_process_buffer["advantages"]),
                clip_adv,
            )
        ).mean()

        # Add cross entropy loss (only for final turns)
        # for now forces the answer to be at the very end, which is catastrophic
        # _ce_answers = model.tokenizer(
        #     current_process_buffer["answers"],
        #     padding="max_length",
        #     max_length=log_probs.shape[1],
        #     return_tensors="pt",
        # )["input_ids"]
        ce_loss = 0
        # for _i in range(log_probs.shape[1]):
        #     # cross entropy loss
        #     ce_loss += torch.nn.functional.cross_entropy(
        #         log_probs[len(log_probs) // 2 :, _i, :],
        #         torch.tensor(_ce_answers)[:, _i].to(accelerator.device),
        #     )
        # #
        # ce_loss /= log_probs.shape[1]

        # Compute final loss
        loss = (
            policy_loss
            - kwargs["entropy_loss_coef"] * entropy
            + kwargs["value_loss_coef"] * value_loss
        )
        if model.curiosity_module is not None:
            loss += model.curiosity_module.loss.mean()
            accelerator.print("curiosity loss = ", model.curiosity_module.loss)
        # + kwargs["ce_coef"] * ce_loss
        # wait for all processes to finish computing the loss
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
        # "ce_loss = ",
        # ce_loss.item(),
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
    # init
    _, wmodel, env, optimizer, lr_scheduler, buf, pad_token = init_game(
        config_args, accelerator
    )
    # slight object modification as evil spaggheti code
    buf.val_buf = torch.zeros(
        [len(buf.val_buf), config_args.rl_script_args.max_new_tokens]
    )
    accelerator.print(f"Done. Training...")
    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    infos = {"turn": 0}
    # Main training loop: collect experience in env and update/log each epoch
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        batch_actions = None
        for t in range(config_args.rl_script_args.steps_per_epoch):
            # call a forward before any generate, sets fsdp properly
            # _ = wmodel(["Initialising model"])
            max_new_tokens = (
                config_args.rl_script_args.max_new_tokens  # if infos["turn"] == 0 else 5
            )
            with torch.no_grad():
                a = wmodel.generate(
                    o,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token,
                )
            # looks for the stop word and replaces all lolgits after it with the pad token
            # logits should also be masked, as should values (or any other output)
            for i, _a in enumerate(a["text"]):
                if " Answer:" in _a:
                    _cut_a = _a.split(" Answer:")[0]
                    if _cut_a != _a:
                        a["text"][i] = _cut_a + " Answer:"
                        _len = len(wmodel.tokenizer(_cut_a)["input_ids"])
                        _pad_log = torch.zeros(
                            max_new_tokens - _len, a["logits"][i].shape[-1]
                        ).to(accelerator.device)
                        _pad_log[:, pad_token] = 1
                        a["logits"][i] = torch.concat(
                            [
                                a["logits"][i][:_len],
                                _pad_log,
                            ]
                        )
            accelerator.print(
                f"Generated turn {infos['turn']}, max_new_tokens: {max_new_tokens}, a: {a['text']} expected: {env.batch[2]}"
            )
            # transform curiosity reward
            if hasattr(a, "curiosity"):
                cur_r = (1 - config_args.rl_script_args.cur2_coef) * cur_r[
                    0
                ] + config_args.rl_script_args.cur2_coef * cur_r[1]
                accelerator.print(
                    f"cur_r: {cur_r.mean() * config_args.rl_script_args.cur_coef}"
                )
            else:
                cur_r = np.array([0] * len(a["text"]))

            # score the right answer, will be used as part of the reward
            # if infos["turn"] == 1:
            #     with torch.no_grad():
            #         ans_score = wmodel.score(
            #             context=o, out_logs=a["logits"], expected=env.batch[2]
            #         )
            # else:
            #     # no reward before episode ends
            #     ans_score = [0] * len(a["text"])

            new_o, r, d, infos = env.step(a["text"])
            ep_ret += sum(r)
            ep_len += 1

            # Store experience to replay buffer
            # TODO: just once per batch, no need to loop
            for i, obs in enumerate(o):
                buf.store(
                    obs,
                    r[i]
                    # + config_args.rl_script_args.score_coef * ans_score[i]
                    + config_args.rl_script_args.cur_coef * cur_r[i],
                    a["value"][i].squeeze(),
                    a["logits"][i],
                )

            if new_o[0] is not None:
                # second part of the conversation has access to entire history
                o = [o[i] + a["text"][i] + new_o[i] + "You:" for i in range(len(o))]
            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal or epoch_ended:
                # save and log
                with open(config_args.rl_script_args.log_file, "a") as f:
                    f.write(
                        f"Episode {n_episodes} - GPU {accelerator.device} - Acc {ep_ret}/{len(o)}\n"
                    )
                    for k, v in env.render().items():
                        f.write(f"{k} : {v}\n")

                buf.finish_path(len(o), expand=a["logits"][i].shape[-1])
                if terminal:
                    n_episodes += 1
                    accelerator.print(
                        f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)} ",
                        f"ans_score: {ans_score.mean() * config_args.rl_script_args.score_coef}",
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
        )
        # print(f"Update results: {update_results}")
    # lm_server.close()


if __name__ == "__main__":
    main()
