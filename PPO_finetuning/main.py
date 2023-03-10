from llm_com_env_text import LLMComEnvText
import hydra
from utils.ppo_buffer import PPOBuffer
from utils import scores_to_proba_dists
# from utils.generate_prompt import generate_prompt
import torch
import numpy as np

from tqdm import tqdm

import gym

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction

lamorel_init()

class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        llm_hidden_size = 768 #self.llm_config.to_dict()[self.llm_config.attribute_map['hidden_size']]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][-1][:, len(tokenized_context["input_ids"]) - 1, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(self.device))
        return value.cpu()

class PPOUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self._llm_module.parameters(), kwargs["lr"])

        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs', 'answers']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            # Use LLM to compute again action probabilities and value

            output = self._llm_module(['__score', 'value'], contexts=contexts, candidates=candidates, require_grad=True)
            scores = torch.stack([_o['__score'] for _o in output]).squeeze()
            # print("scores = ", scores)
            probas = scores_to_proba_dists(scores)
            values = torch.stack([_o["value"][0] for _o in output])

            # Compute policy loss
            entropy = probas.entropy().mean()
            log_prob = probas.log_prob(current_process_buffer['actions'])
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=-1)
            ratio = torch.exp(log_prob - current_process_buffer['logprobs'])
            clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages']
            policy_loss = -(torch.min(ratio * current_process_buffer['advantages'], clip_adv)).mean()

            # Compute value loss
            value_loss = ((values - current_process_buffer['returns']) ** 2).mean()

            # Add cross entropy loss (only for final turns)
            _ce_mask = (current_process_buffer['answers'] != -1)
            _ce_scores = scores[_ce_mask]
            _ce_answers = current_process_buffer['answers'][_ce_mask].squeeze() 
            ce_loss = torch.nn.functional.cross_entropy(_ce_scores, _ce_answers.type(torch.LongTensor))

            # Compute final loss
            # print("policy_loss = ", policy_loss, "entropy = ", entropy, "value_loss = ", value_loss, "ce_loss = ", ce_loss)
            loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss + kwargs["ce_coef"] * ce_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if kwargs["save_after_update"]:
            print("Saving model...")
            torch.save(self._llm_module.state_dict(), kwargs["log_dir"] + "/model.checkpoint")
            print("Model saved")

        return {'loss': loss}

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create LLM agent
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater_class=PPOUpdater,
                       custom_module_functions={
                            'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
                        }
                )
    
    # lm_server2 = Caller(config_args.lamorel_args2)

    # Instantiate environment
    env = LLMComEnvText(lm_server,
        config_args.rl_script_args.dataset_path, 
        max_length=config_args.rl_script_args.max_new_tokens, 
        batch_size=config_args.rl_script_args.batch_size,
        affix=config_args.rl_script_args.affixes)

    # Set up experience buffer
    buf = PPOBuffer(config_args.rl_script_args.steps_per_epoch*config_args.rl_script_args.batch_size,
        config_args.rl_script_args.gamma,
        config_args.rl_script_args.lam)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    infos = {"turn": 0}

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        batch_actions = []
        for t in range(config_args.rl_script_args.steps_per_epoch):
            # prompt = generate_prompt(o, infos)
                # Get actions
            _gen = lm_server.generate(o,
                        num_return_sequences=config_args.rl_script_args.num_candidates,
                        max_new_tokens=config_args.rl_script_args.max_new_tokens,
                        do_sample=True,
                        top_k=config_args.rl_script_args.top_k,
                        pad_token_id=50256,
                    )
                    
            actions = [[a["text"] for a in aset] for aset in _gen]
            batch_actions.append(actions)
            ans_idxs = [-1] * len(actions)
            if infos["turn"] == 1:    
                for i, _a in enumerate(env.batch[2]):
                    if _a not in actions[i]:
                        actions[i][-1] = _a
                    ans_idxs[i] = actions[i].index(_a)
            
            output = lm_server.score(contexts=o, candidates=actions,
                                     additional_module_function_keys=['value'])
            values = []
            a_idxs = []
            log_probs = []
            a = []
            for i, _out in enumerate(output):
                proba_dist = scores_to_proba_dists(torch.reshape(_out['__score'], (1, len(actions[i]))))
                values.append(_out["value"][0]) # is the first value representative of the whole action space ?
                a_idx = proba_dist.sample()
                a_idxs.append(a_idx)
                log_probs.append(proba_dist.log_prob(a_idx))
                a.append(actions[i][a_idx.cpu().item()])

            new_o, r[i], d, infos = env.step(a)
            ep_ret += sum(r)
            ep_len += 1

            # TODO : makefile if not exists
            with open(config_args.rl_script_args.log_file, "a") as f:
                for k,v in env.render().items():
                    f.write(f"{k} : {v}\n")

            # save and log
            for i, a_idx in enumerate(a_idxs):
                buf.store(o[i], a_idx, r, values[i], ans_idxs[i], log_probs[i])
            o = new_o
            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal or epoch_ended:  
                buf.finish_path(values)              
                if terminal:
                    n_episodes += 1
                    print(f"Episode {n_episodes}:")
                    print(f"Ret: {ep_ret}")
                    print(f"Len: {ep_len}")
                o, ep_ret, ep_len = env.reset(), 0, 0
                base_prompt = o[0]

        # Perform PPO update!
        save_model = (epoch % config_args.rl_script_args.save_freq == 0 or
                      epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        # buf.to_txt(config_args.rl_script_args.log_dir + "/buffer.txt")
        collected_trajectories = buf.get()
        update_results = lm_server.update(collected_trajectories['obs'],
                                            # [_a for _a_list in batch_actions for _a in _a_list],
                                            np.array(batch_actions).reshape(-1, config_args.rl_script_args.num_candidates),
                                            actions=collected_trajectories['act'],
                                            returns=collected_trajectories['ret'],
                                            advantages=collected_trajectories['adv'],
                                            logprobs=collected_trajectories['logp'],
                                            answers=collected_trajectories['ans'],
                                            lr=config_args.rl_script_args.lr,
                                            clip_eps=config_args.rl_script_args.clip_eps,
                                            entropy_coef=config_args.rl_script_args.entropy_coef,
                                            value_loss_coef=config_args.rl_script_args.value_loss_coef,
                                            ce_coef=config_args.rl_script_args.ce_coef,
                                            ppo_epochs=config_args.rl_script_args.ppo_epochs,
                                            save_after_update=save_model,
                                            output_dir=config_args.rl_script_args.log_dir)
        print(f"Update results: {update_results}")
    lm_server.close()

@hydra.main(config_path='/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning', config_name='local_gpu_config')
def test_lamorel_llmEnv(config_args):
    class Config:
        model_name = "gpt2"
        dataset_path = "/homedtcl/mmahaut/projects/llm2llm/data/mindless_dataset_randomized_train.txt"
    config = Config()
    
    # lm_server = Caller(config_args.lamorel_args,
    #                    custom_updater_class=PPOUpdater,
    #                    custom_module_functions={
    #                         'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
    #                     }
    #             )
    
    lm_server = Caller(config_args.lamorel_args)
    prompt = [
        ["Q: ","Q: "],
        ["Context: ","Q: ","A: "],
    ]
    env = LLMComEnvText(lm_server, config.dataset_path, max_length=100, batch_size=10)
    obs = env.reset()
    print(obs)
    obs = env.step(["hello0", "hello1", "hello2", "hello3", "hello4", "hello5", "hello6", "hello7", "hello8", "hello9"])
    print(obs)

    lm_server.close()

if __name__ == '__main__':
    # test_lamorel_llmEnv()
    main()