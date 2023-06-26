from llm_com_env_text import LLMComEnvText
import hydra
from utils.ppo_buffer import PPOBuffer
from utils import scores_to_proba_dists

# from utils.generate_prompt import generate_prompt
import torch
import numpy as np

from tqdm import tqdm

import gym
import random
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction
from utils.curiosity_modules import CuriosityModule

lamorel_init()
# Accelerate
from accelerate import Accelerator

accelerator = Accelerator()


class LogitsFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            # log softmax
            # logits = self.log_soft(forward_outputs['logits'][:, len(tokenized_context["input_ids"]) - 1:-1, :])
            # print("logits = ", logits.shape)
            logits = forward_outputs["logits"][
                :, len(tokenized_context["input_ids"]) - 1 : -1, :
            ]
        else:
            logits = forward_outputs["logits"][
                :, :-1, :
            ]  # skip </s> token appended by tokenizer

        return logits.softmax(-1).cpu()


class HiddenStatesFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs["hidden_states"][-1][
                :, len(tokenized_context["input_ids"]) - 1, :
            ]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        return model_head.cpu()


class PPOUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        # TODO : get pad token
        pad_token = 50256
        max_new_tokens = 30
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(
                self._llm_module.parameters(), kwargs["lr"]
            )

        current_process_buffer = {}
        for k in ["advantages", "returns", "logprobs", "answers"]:
            current_process_buffer[k] = kwargs[k][:]

        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            # Use LLM to compute again action probabilities and value
            with torch.no_grad():
                candidates = self._llm_module.module.generate(
                    tuple(contexts),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token,
                )
                candidates = [[_c[0]["text"]] for _c in candidates]
            output = self._llm_module(
                ["word_logits"],
                contexts=contexts,
                candidates=candidates,
                require_grad=True,
            )
            log_probs = torch.stack([_o["word_logits"][0].mean(-2) for _o in output])
            # prevent small values to become nan
            # log_probs = torch.clamp(log_probs, -1e-10, 1000)

            entropy = scores_to_proba_dists(log_probs).entropy()
            ratio = torch.exp(
                log_probs - torch.stack(current_process_buffer["logprobs"])
            )
            clip_adv = (
                torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]).T
                * current_process_buffer["advantages"]  # <-- am I rewarding properly?
            )

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
            _ce_answers = []
            for _t in current_process_buffer["answers"]:
                _ce_answers.append(
                    self._llm_module.module._LLM_tokenizer(_t)["input_ids"][0]
                )
            # pad with eos if needed (this too feels suboptimal)
            # max_len = max([len(_ce) for _ce in _ce_answers])
            # for _ce in _ce_answers:
            #     _ce += [pad_token] * (max_len - len(_ce))

            # _ce_tokens = torch.tensor([_ce["input_ids"] for _ce in _ce_answers])
            ce_loss = torch.nn.functional.cross_entropy(
                log_probs[len(log_probs) // 2 :], torch.tensor(_ce_answers)
            )

            # Compute final loss
            loss = (policy_loss - kwargs["entropy_coef"] * entropy).mean() + kwargs[
                "ce_coef"
            ] * ce_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        print(
            "policy_loss = ",
            policy_loss.mean().item(),
            "entropy = ",
            entropy.mean().item(),
            "ce_loss = ",
            ce_loss.item(),
        )
        if kwargs["save_after_update"]:
            print("Saving model...")
            torch.save(
                self._llm_module.state_dict(), kwargs["log_dir"] + "/model.checkpoint"
            )
            print("Model saved")

        return {"loss": loss}


@hydra.main(config_path="config", config_name="config")
def main(config_args):
    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pad_token = 50256  # TODO : get pad token
    hidden_size = 5120

    # Create LLM agent
    print("Creating LLM agent...")
    lm_server = Caller(
        config_args.lamorel_args,
        custom_updater_class=PPOUpdater,
        custom_module_functions={
            # 'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type),
            "word_logits": LogitsFn(config_args.lamorel_args.llm_args.model_type),
            "hidden_states": HiddenStatesFn(
                config_args.lamorel_args.llm_args.model_type
            ),
        },
    )

    # lm_server2 = Caller(config_args.lamorel_args2)

    # Instantiate environment
    env = LLMComEnvText(
        lm_server,
        config_args.rl_script_args.dataset_path,
        max_length=config_args.rl_script_args.max_new_tokens,
        batch_size=config_args.rl_script_args.batch_size,
        affix=config_args.rl_script_args.affixes,
    )

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

    # Setting up curiosity module
    c_module = CuriosityModule(llm_hidden_size=hidden_size, device="cpu")

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        batch_actions = None
        for t in range(config_args.rl_script_args.steps_per_epoch):
            # Get actions
            max_new_tokens = (
                config_args.rl_script_args.max_new_tokens if infos["turn"] == 0 else 1
            )
            print("Generating actions...")
            _gen = lm_server.generate(
                tuple(o),
                max_new_tokens=max_new_tokens if infos["turn"] == 0 else 1,
                pad_token_id=pad_token,
            )

            actions = [[a[0]["text"]] for a in _gen]
            # scores = [a[0]["score"] for a in _gen]
            if batch_actions is None:
                batch_actions = actions
            else:
                batch_actions.extend(actions)

            if infos["turn"] == 1:
                # Get logits for answer
                ans = lm_server.score(
                    contexts=o, candidates=np.expand_dims(env.batch[2], axis=1)
                )
            else:
                ans = [0] * len(actions)
            # print(o)
            print("Scoring actions hs...")
            hs_o = lm_server.score(
                contexts=o,
                candidates=tuple([" "] * len(o)),
                additional_module_function_keys=["hidden_states"],
            )
            # here maybe generate removes the need for context
            hs_o = torch.stack([h["hidden_states"] for h in hs_o])
            print("Getting hidden states...")
            output = lm_server.score(
                contexts=o,
                candidates=actions,
                additional_module_function_keys=["word_logits", "hidden_states"],
            )
            hs_a = torch.stack([h["hidden_states"] for h in output])

            # get max length of output['word_logits']
            # max_len = max([_o['word_logits'].shape[1] for _o in output]) > max_len
            # pad with one_hot of pad_token and get mask
            # one_hot_eos = torch.nn.functional.one_hot(torch.tensor(pad_token), num_classes=output[0]['word_logits'].shape[-1])
            # print(output[0]['word_logits'].shape)
            # log_probs = torch.stack([torch.nn.functional.pad(_o['word_logits'], (0, 0, 0, max_len - _o['word_logits'].shape[1]), value=0) for _o in output])

            new_o, r, d, infos = env.step(actions)
            ep_ret += r.sum()
            ep_len += 1
            if new_o[0] is not None:
                print("Scoring new hidden states...")
                hs_new_o = lm_server.score(
                    contexts=new_o,
                    candidates=tuple([" "] * len(new_o)),
                    additional_module_function_keys=["hidden_states"],
                )
                hs_new_o = torch.stack([h["hidden_states"] for h in hs_new_o])
                # calculate curiosity reward
                c_rew = c_module(hs_o, hs_new_o, hs_a).detach().numpy()
                # update curiosity
                c_module.update()
            else:
                c_rew = np.array([0] * len(actions))

            # save and log
            with open(config_args.rl_script_args.log_file, "a") as f:
                for k, v in env.render().items():
                    f.write(f"{k} : {v}\n")

            # Store experience to replay buffer
            for i, obs in enumerate(o):
                buf.store(
                    obs, ans[i], c_rew[i], output[i]["word_logits"][0].mean(dim=-2)
                )

            o = new_o
            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal or epoch_ended:
                buf.finish_path(len(o))
                if terminal:
                    n_episodes += 1
                    print(
                        f"Episode {n_episodes} --> Acc: {ep_ret}/{len(o)} Reward: {torch.stack(ans).mean()}"
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
        print(batch_actions)
        update_results = lm_server.update(
            collected_trajectories["obs"],
            # [_a for _a_list in batch_actions for _a in _a_list],
            batch_actions,
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
    lm_server.close()


if __name__ == "__main__":
    # test_lamorel_llmEnv()
    main()
