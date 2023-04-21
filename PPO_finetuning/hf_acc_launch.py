from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_com_env_text import LLMComEnvText
from utils.ppo_buffer import PPOBuffer
from utils import scores_to_proba_dists
from accelerate import init_empty_weights
import hydra
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import deepspeed

# from utils.generate_prompt import generate_prompt
import torch
import numpy as np
import time
from tqdm import tqdm

from utils.curiosity_modules import CuriosityModule

# Accelerate
from accelerate import Accelerator

accelerator = Accelerator()


class CastOutputToFloat(torch.nn.Sequential):
    # for stability
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class StrToStrWrapper(torch.nn.Module):
    # create an str to str model wrapper
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def prepare_inputs(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(accelerator.device)
        return inputs

    def forward(self, inputs, **kwargs):
        inputs = self.prepare_inputs(inputs)
        output = self.model(**inputs, **kwargs)
        return output

    def save_pretrained(self, save_directory):
        model = accelerator.unwrap_model(self.model)
        model.save_pretrained(
            save_directory,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

    def score(self, context, candidates, **kwargs):
        assert len(context) == len(
            candidates
        ), "context and candidate must have the same length"
        all_inputs = self.prepare_inputs(
            [candidates[i] + context[i] for i in range(len(context))]
        )
        # TODO: check that running prepare thrice is really the fastest way to do this
        context_inputs = self.prepare_inputs(context)
        candidate_inputs = self.prepare_inputs(candidates)

        output = self.model(**all_inputs, **kwargs)
        # accelerator.print(output.logits.shape, inputs["input_ids"].shape)
        logits = output.logits[:, len(context_inputs) - 1 :, :].softmax(-1)
        # gather the logits of the input tokens
        _token_logs = logits.gather(
            2, candidate_inputs["input_ids"][:, :, None]
        ).squeeze(-1)
        # ignore padding
        # accelerator.print(_token_logs)
        scores = _token_logs.masked_fill(
            candidate_inputs["input_ids"] == self.tokenizer.pad_token_id, 1.0
        ).prod(-1)
        return scores

    def fast_generate(self, inputs, pad_token=50256, **kwargs):
        # legacy
        # fast greedy generation?
        # using past_key values for now blocks everything because of an attention mask issue
        batch = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        )
        query_ids = batch["input_ids"]
        past_key_values = None
        in_hs = None

        # generated = []
        score = 0
        out_hs = []
        logits = []

        for _i in range(kwargs.pop("max_new_tokens", 30)):
            # accelerator.print(_i, batch, kwargs)
            output = self.model(
                **batch,
                past_key_values=past_key_values,
                output_hidden_states=True,
                **kwargs,
            )
            past_key_values = output.past_key_values
            new_id = output.logits[:, -1, :].argmax(-1).unsqueeze(-1)
            batch["input_ids"] = torch.cat(
                (batch["input_ids"], new_id),
                dim=1,
            )
            # batch.pop("attention_mask", None)
            batch["attention_mask"] = torch.cat(
                (batch["attention_mask"], torch.ones_like(new_id)),
                dim=1,
            )
            logits.append(
                output.logits[:, -1, :].softmax(-1)
            )  # TODO: maybe remove softmax (see 9 point video)
            out_hs.append(output.hidden_states[-1][:, -1, :])
            if in_hs is None:
                in_hs = output.hidden_states[-1][:-1]
            # if batch["input_ids"].item() == self.tokenizer.eos_token_id:
            #     break
        logits = torch.cat(logits, dim=1)
        out_hs = torch.cat(out_hs, dim=1)  # TODO: check if this is required
        in_hs = torch.cat(in_hs, dim=1)

        result = {
            "text": self.tokenizer.batch_decode(
                batch["input_ids"][len(in_hs) :], skip_special_tokens=True
            ),
            "logits": logits,
            "out_hs": out_hs,
            "in_hs": in_hs,
        }

        if kwargs.get("get_score", True):
            _token_logs = logits.gather(2, query_ids).squeeze(-1)
            # ignore padding
            scores = _token_logs.masked_fill(
                query_ids == self.tokenizer.pad_token_id, 1.0
            ).prod(-1)
            result["scores"] = self.get_score(logits, query_ids)

        return results

    def generate(self, inputs, **kwargs):
        inputs = self.prepare_inputs(inputs)
        output = self.model.generate(
            **inputs,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            synced_gpus=True,
            **kwargs,
        )

        _out_logs = torch.stack(output["scores"])
        _out_hs = output["hidden_states"][-1][-1][:, -1, :].squeeze()
        _in_hs = output["hidden_states"][0][-1][:, -1, :].squeeze()
        result = {
            "text": self.tokenizer.batch_decode(
                output.sequences[
                    :, inputs["input_ids"].shape[1] :
                ],  # check if a -1 is required
                skip_special_tokens=True,
            ),
            "logits": _out_logs.swapaxes(0, 1),
            "out_hs": _out_hs.to(torch.float32),
            "in_hs": _in_hs.to(torch.float32),
        }
        return result


def perform_update(model, optimizer, contexts, **kwargs):
    # TODO : get pad token
    pad_token = 50256
    max_new_tokens = 30
    current_process_buffer = {}
    for k in ["advantages", "returns", "logprobs", "answers"]:
        current_process_buffer[k] = kwargs[k][:]

    for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
        # _ = model(["Initialising model"])

        # Use LLM to compute again action probabilities and value
        with torch.no_grad():
            actions = model.generate(
                contexts,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token,
            )
            # actions = [[_a["text"]] for _a in actions]
        output = model(actions["text"])
        log_probs = output["logits"].log_softmax(-1)

        entropy = scores_to_proba_dists(log_probs).entropy()
        ratio = torch.exp(
            log_probs.mean(-2) - torch.stack(current_process_buffer["logprobs"])
        ).cpu()

        clip_adv = (
            torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]).T
            * current_process_buffer["advantages"]
        )
        # accelerator.print(
        #     "ratio",
        #     ratio.T.mean(),
        #     "adv",
        #     current_process_buffer["advantages"].mean(),
        #     "clip_adv",
        #     clip_adv.mean(),
        #     "returns",
        #     current_process_buffer["returns"].mean(),
        # )
        policy_loss = -(
            torch.min(
                ratio.T
                * (
                    current_process_buffer["advantages"]
                    + kwargs["cur_coef"] * current_process_buffer["returns"]
                ),
                clip_adv,
            )
        ).mean()

        # Add cross entropy loss (only for final turns)
        _ce_answers = model.tokenizer(current_process_buffer["answers"], padding=True)[
            "input_ids"
        ]
        # print(_ce_answers)
        ce_loss = 0
        # accelerator.print("log_probs", log_probs.shape)
        # accelerator.print("answers", torch.tensor(_ce_answers).shape)
        # for _i in range(len(_ce_answers[0])):
        #     # cross entropy loss
        #     ce_loss += torch.nn.functional.cross_entropy(
        #         log_probs[len(log_probs) // 2 :, _i, :],
        #         torch.tensor(_ce_answers)[:, _i].to(accelerator.device),
        #     )
        # #
        # ce_loss /= len(_ce_answers[0])

        # Compute final loss
        loss = (policy_loss - kwargs["entropy_coef"] * entropy).mean() + kwargs[
            "ce_coef"
        ] * ce_loss
        # wait for all processes to finish computing the loss
        # Optimize
        optimizer.zero_grad()
        accelerator.backward(loss)
        # loss.mean().backward()
        optimizer.step()

    accelerator.print(
        "policy_loss = ",
        policy_loss.mean().item(),
        "entropy = ",
        entropy.mean().item(),
        # "ce_loss = ",
        # ce_loss.item(),
    )
    if kwargs["save_after_update"]:
        accelerator.print("Saving model...")
        # TODO with accelerator (deal with at the wrapper level)
        model.save_pretrained(kwargs["log_dir"] + "/model")
        accelerator.print(f"Model saved in {kwargs['log_dir']}/model")

    return {"loss": loss}


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config_args):
    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # pad token to default to if not found
    pad_token = 50256

    # Create LLM agent
    model_path = config_args.lamorel_args.llm_args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", use_fast=False
    )
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id else pad_token
    accelerator.print(f"Creating {model_path} agent...")
    # with init_empty_weights:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    # model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    accelerator.print(f"Done. Creating curiosity module...")
    # Setting up curiosity module
    c_module = CuriosityModule(llm_hidden_size=model.config.hidden_size)
    c_module.optimizer = torch.optim.Adam(
        c_module.parameters(), lr=config_args.rl_script_args.lr
    )
    # Instantiate environment with non LoRa, untrained LLM
    env = LLMComEnvText(
        StrToStrWrapper(model, tokenizer),
        config_args.rl_script_args.dataset_path,
        max_length=config_args.rl_script_args.max_new_tokens,
        batch_size=config_args.rl_script_args.batch_size,
        affix=config_args.rl_script_args.affixes,
    )

    accelerator.print(f"Done. Adding LoRa to {model_path} agent...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        model.parameters(), lr=config_args.rl_script_args.lr
    )
    # torch.optim.Adam(model.parameters(), lr=config_args.rl_script_args.lr)
    accelerator.print(f"Done. Preparing...")

    (model, env.dataloader, optimizer) = accelerator.prepare(
        model, env.dataloader, optimizer
    )
    accelerator.print(f"Done. Training...")
    # wrap the LoRa model
    wmodel = StrToStrWrapper(model, tokenizer)
    # TODO: include curiosity in accelerator?
    # TODO: freeze all non-LoRa parameters?
    # Set up experience buffer
    buf = PPOBuffer(
        config_args.rl_script_args.steps_per_epoch
        * config_args.rl_script_args.batch_size,
        config_args.rl_script_args.gamma,
        config_args.rl_script_args.lam,
    )

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    infos = {"turn": 0}

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        batch_actions = None
        for t in range(config_args.rl_script_args.steps_per_epoch):
            # call a forward before any generate, sets fsdp properly
            # _ = wmodel(["Initialising model"])
            max_new_tokens = (
                config_args.rl_script_args.max_new_tokens if infos["turn"] == 0 else 5
            )
            with torch.no_grad():
                a = wmodel.generate(
                    o,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token,
                )
            accelerator.print(
                f"Generated turn {infos['turn']}, max_new_tokens: {max_new_tokens}, a: {a['text']} expected: {env.batch[2]}"
            )
            # compute curiosity reward
            # for k, v in a.items():
            #     if isinstance(v, torch.Tensor):
            #         accelerator.print(f"{k}: {v.shape}")

            cur_r = c_module.auto_run(
                a["in_hs"].cpu(),
                a["out_hs"].cpu(),
            )
            if cur_r is None:
                cur_r = np.array([0] * len(a["text"]))
            else:
                cur_r = (1 - config_args.rl_script_args.cur2_coef) * cur_r[
                    0
                ] + config_args.rl_script_args.cur2_coef * cur_r[1]
                c_module.update()
                # c_module.optimizer.zero_grad()
                # accelerator.backward(c_module.loss.mean())
                # c_module.optimizer.step()

            accelerator.print(f"cur_r: {cur_r}")

            # if infos["turn"] == 1:
            #     # Get probability of correct answer as reward on episode end
            #     with torch.no_grad():
            #         # to deal with tiny values we use logsoftmax instead of softmax, this is more of a penalty to minimize than a reward to maximize
            #         ans_score = wmodel.score(context=o, candidates=env.batch[2])
            #     accelerator.print(f"ans_score: {ans_score}")

            # else:
            #     # no reward before episode ends
            #     ans_score = [0] * len(a["text"])

            new_o, r, d, infos = env.step(a["text"])
            ep_ret += sum(r)
            ep_len += 1

            # save and log
            with open(config_args.rl_script_args.log_file, "a") as f:
                for k, v in env.render().items():
                    f.write(f"{k} : {v}\n")

            # Store experience to replay buffer
            for i, obs in enumerate(o):
                buf.store(
                    # obs, ans_score[i], cur_r[i], a["logits"][i].log_softmax(-1).mean(-2)
                    obs,
                    r[i],
                    cur_r[i],
                    a["logits"][i].log_softmax(-1).mean(-2),
                )
            # accelerator.print(f"a: {a['logits'].shape}")
            if new_o[0] is not None:
                # second part of the conversation has access to entire history
                o = [o[i] + a["text"][i] + new_o[i] for i in range(len(o))]
            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal or epoch_ended:
                buf.finish_path(len(o))
                if terminal:
                    n_episodes += 1
                    accelerator.print(
                        f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)} Reward: {torch.Tensor(r).mean()}"
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
            contexts=collected_trajectories["obs"],
            returns=collected_trajectories["ret"],
            advantages=collected_trajectories["adv"],
            logprobs=collected_trajectories["logp"],
            answers=env.batch[2],
            lr=config_args.rl_script_args.lr,
            clip_eps=config_args.rl_script_args.clip_eps,
            entropy_coef=config_args.rl_script_args.entropy_coef,
            ce_coef=config_args.rl_script_args.ce_coef,
            cur_coef=config_args.rl_script_args.cur_coef,
            ppo_epochs=config_args.rl_script_args.ppo_epochs,
            save_after_update=save_model,
            log_dir=config_args.rl_script_args.log_dir,
        )
        # print(f"Update results: {update_results}")
    # lm_server.close()


if __name__ == "__main__":
    main()
