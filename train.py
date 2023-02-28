import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from torch.utils.data import DataLoader
from llm_com_env import LLMComEnv
from data import FruitDataset
from loss import PPOLoss
from utils import StopTokenLogitsProcessor, combine_inputs
from stable_baselines3.common.vec_env import SubprocVecEnv

def prepare_env(config):
    # Load the model TODO: use lamorel
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    # get the tokenizer, setup padding
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
    device = "cpu" if config.force_cpu else "cuda"
    env = LLMComEnv(model, tokenizer, config.dataset_path, device, max_length=20)
    return env, model

def base_ppo_train(env, model, device="cuda"):
    # params
    # TODO: use the config
    num_epochs = 1000
    do_sample = True
    max_len = 20
    beam_size = 1
    num_return_sequences = 1
    top_k = 50 # roberto used len(tokenizer), to be tested
    logits_processor = StopTokenLogitsProcessor(env.tokenizer, do_sample)

    model = model.to(device)
    loss_function = PPOLoss(baseline="MeanBaseline", device="cpu")
    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Train the model
    for episode in range(num_epochs):
        # zero the gradients
        episode_logits = []
        optimizer.zero_grad()
        done = False
        obs = env.reset()
        prompt_len = obs["input_ids"].shape[1]
        logits = None
        episode_logits = []
        latest_saved_turn = 0
        len_context = None
        while not done:
            # get answer to prompt in a single llm pass
            generated = model.generate(
                **obs,
                do_sample=do_sample,
                max_length=prompt_len + max_len,
                num_beams=beam_size,
                num_return_sequences=num_return_sequences,
                logits_processor=LogitsProcessorList([logits_processor]),
                top_k=top_k,
            )
            # get logits corresponding to the answer
            logits = model(**combine_inputs(obs, generated))
            logits = logits[0][:, prompt_len - 1 : , :]
            logits = logits.log_softmax(-1)
            # step the environment
            obs, reward, done, info = env.step(generated[:, prompt_len - 1 :])
            if info["turn"] > latest_saved_turn:
                # deal with end of episode before max_token length
                logits = logits[:,len_context:,:]
                episode_logits.append(logits)
                latest_saved_turn = info["turn"]
                len_context = None
        # prepare logits
        logits = torch.cat(episode_logits, dim=1)
        # no cross entropy loss, just policy loss
        policy_loss = loss_function(float(reward), logits.cpu(), entropy_coeff=0.0)
        # backward pass
        policy_loss.backward()
        # update the weights
        optimizer.step()
        # print the loss
        print(
            "policy loss: ",
            policy_loss.item(),
            "episode:\n",
            env.render("json"),
            "reward: ",
            reward,
        )

    # save the model
    model.save("./fine_tune", "fine_tune")
    print("DONE TRAINING : saved model")

def test_play(env, model, device="cuda"):
    # todo deal with device (through lamorel)
    env = SubprocVecEnv([lambda: env]*2)
    obs = env.reset()
    # print(obs)
    print("SHAPE: ",obs["input_ids"].shape)
    

    for i in range(100):
        # print(obs)
        logits = model.forward(**obs).logits
        action = torch.argmax(logits[:,-1:,:], dim=-1)
        obs, reward, done, info = env.step(action)
        if done:
            print(repr(env.render()))
            break

def train(env):
    # use rl4lm nlpo to train
    pass

if __name__ == "__main__":
    # TODO: make yaml config
    class Config:
        model_name = "gpt2"
        dataset_path = "./data/mindless_dataset_randomized_train.txt"
        force_cpu = False
    config = Config()
    env, model = prepare_env(config)
    test_play(env, model)
    # model = AutoModelForCausalLM.from_pretrained(config.model_name)
    # base_ppo_train(env, model)


