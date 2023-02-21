import gymnasium as gym
from gymnasium import spaces
from utils import combine_inputs
import torch

class LLMComEnv(gym.Env):
    """Environment where the agent communicates with a frozen language model."""
    def __init__(self, model, tokenizer, dataloader, device, max_length=10, n_turns=2):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataloader = iter(dataloader)
        self.device = device
        self.max_length = max_length
        self.action_space = spaces.Discrete(len(tokenizer))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(max_length, len(tokenizer)), dtype=float
        )
        self.n_turns = n_turns
        # self.reset()

    def reset(self):
        self.batch = self.dataloader.next()
        self.prompt = self.tokenizer(list(self.batch[0]), return_tensors="pt", padding=True) # question
        self.response = {"input_ids": torch.zeros((1,1), dtype=torch.long), "attention_mask": torch.zeros((1,1), dtype=torch.long)}
        self.logs = {"prompt" : self.batch[0], "context" : self.batch[1], "answer" : self.batch[2], "llm1_response": [], "llm2_response": []}
        self.turn = 0
        self.done = False
        return self.prompt

    def step(self, action):
        # a step is a single token. 
        # max_step steps make a turn
        # in the first turn it answers the prompt
        # in the second turn it answers the llm's response
        if self.done:
            raise RuntimeError("Episode is done, call reset() to start a new episode.")
        else:
            reward = 0
            # add the token to the input ids of state and auto complete the attention mask
            self.response = combine_inputs(self.response, action)
            if len(self.response["input_ids"].squeeze()) == self.max_length or self.tokenizer.eos_token_id in action:
                self.turn += 1
                self.logs["llm1_response"].append(self.tokenizer.decode(self.response["input_ids"].squeeze()))
                if self.turn == self.n_turns:
                    reward = self.batch[2][0] in self.tokenizer.decode(self.response["input_ids"].squeeze()) # reward is 1 if the answer is correct
                    self.done = True
                else:
                    # send the response to the llm and get a new prompt
                    model_input = combine_inputs(self.tokenizer(list(self.batch[1]), return_tensors="pt", padding=True), self.response)
                    len_input = model_input["input_ids"].shape[1]
                    model_output = self.model.generate(**model_input, max_length=len_input + self.max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7, num_return_sequences=1)
                    # remove the prompt from the model output
                    model_output = model_output[:,model_input["input_ids"].shape[1]:]
                    self.logs["llm2_response"].append(self.tokenizer.decode(model_output.squeeze()))
                    # make prompt for next turn
                    self.prompt = combine_inputs(self.tokenizer(list(self.batch[0]), return_tensors="pt", padding=True), model_output)
                    self.response = {"input_ids": torch.zeros((1,1), dtype=torch.long), "attention_mask": torch.zeros((1,1), dtype=torch.long)}
            return combine_inputs(self.prompt,self.response), reward, self.done, {}

    def render(self, mode="human"):
        """
        Renders the environment. Calling once per epoch gives best display
        """
        if mode == "human" and self.turn == self.n_turns:
            print("Prompt: " + self.logs["prompt"][0])
            print("LLM1 Response: " + self.logs["llm1_response"][0])
            print("Context: " + self.logs["context"][0])
            print("LLM2 Response: " + self.logs["llm2_response"][-1])
            print("LLM1 Response: " + self.logs["llm1_response"][1])
            print("Expected Answer: " + self.logs["answer"][0])
        return self.logs

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __str__(self):
        return "LLMComEnv"

    def __repr__(self):
        return "LLMComEnv"
