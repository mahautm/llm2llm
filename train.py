import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from llm_com_env import LLMComEnv
from data import FruitDataset

def prepare_env(config):
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    # get the tokenizer, setup padding
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
    # Load the dataset
    dataset = FruitDataset(config.dataset_path)
    # Load the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = "cpu" if config.force_cpu else "cuda"
    env = LLMComEnv(model, tokenizer, dataloader, device, max_length=20)
    return env

def test_play(env):
    obs = env.reset()

    for i in range(30):
        logits = model.forward(**obs).logits
        action = torch.argmax(logits[:,-1:,:], dim=-1)
        obs, reward, done, info = env.step(action)

    env.render() 

if __name__ == "__main__":
    # TODO: make yaml config
    class Config:
        model_name = "gpt2"
        dataset_path = "./data/mindless_dataset_randomized_train.txt"
        force_cpu = False
    config = Config()
    env = prepare_env(config)
    test_play(env)

