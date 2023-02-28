import gym
import torch
from torch.utils.data import DataLoader
from data import FruitDataset
from env_utils import combine_inputs

# pad for llm

from torch.nn.utils.rnn import pad_sequence
class LLMComEnv(gym.Env):
    """Environment where the agent communicates with a frozen language model."""
    def __init__(self, model, tokenizer, dataset_path, device, max_length=10, max_prompt_length=50, n_turns=2):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        # Load the dataset
        dataset = FruitDataset(dataset_path)
        # Load the dataloader
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.data = None
        self.device = device
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        # self.action_space = gym.spaces.box.Box(
        #     low=0, high=1, shape=(max_length, len(tokenizer)), dtype=float
        # ) # used to accelerate by giving many generated tokens at once
        self.action_space = gym.spaces.Discrete(len(tokenizer))
        # self.observation_space = gym.spaces.box.Box(
        #     low=0, high=1, shape=(max_length + max_prompt_length, len(tokenizer)), dtype=float
        # ) my simplified version
        self.observation_space = gym.spaces.dict.Dict(
            {
                # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
                # while creating rollout buffers, observations are concatenated for each key
                "input_ids": gym.spaces.Box(
                    low=0, high=len(tokenizer), shape=(self.max_length + max_prompt_length,)
                ),
                "attention_mask": gym.spaces.Box(
                    low=0, high=1, shape=(self.max_length + max_prompt_length,)
                )
            }
        )
        # self.reward_function = None # required by the RL4LM library, but not used here
        self.n_turns = n_turns
        # self.max_steps = max_length * n_turns # required by the RL4LM library, but not used here
        # self.reset()

    def reset(self, allow_rewind=True):
        if self.data is None:
            self.data = iter(self.dataloader)
        try:
            self.batch = next(self.data)
        except StopIteration:
            if allow_rewind:
                # rewind dataloader
                self.data = iter(self.dataloader)
                self.batch = next(self.data)
            else:
                raise RuntimeError("Dataloader has been exhausted, call rewind_dataloader() or set allow_rewind to True to start a new episode from the same data.")
        
        self.prompt = self.tokenizer(list(self.batch[0]), return_tensors="pt", padding=True) # question
        # set device
        for key in self.prompt:
            self.prompt[key] = self.prompt[key].to(self.device)
        self.response = {"input_ids": torch.zeros((1,1), dtype=torch.long).to(self.device), "attention_mask": torch.zeros((1,1), dtype=torch.long).to(self.device)}
        self.logs = {"prompt" : self.batch[0], "context" : self.batch[1], "answer" : self.batch[2], "llm1_response": [], "llm2_response": []}
        self.turn = 0
        self.done = False
        # return {
        #         # pad for sb3
        #         # while creating rollout buffers, observations are concatenated for each key
        #         "prompt_or_input_encoded_pt": self.prompt["input_ids"].cpu(),
        #         "prompt_or_input_attention_mask_pt": self.prompt["attention_mask"].cpu(),
        #         "context_encoded_pt":[[]],
        #         "context_attention_mask_pt": [[]],
        #         "input_encoded_pt": self.response["input_ids"].cpu(),
        #         "input_attention_mask_pt": self.response["attention_mask"].cpu(),
        #     }   
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
            if len(self.response["input_ids"].squeeze()) >= self.max_length or self.tokenizer.eos_token_id in action:
                self.turn += 1
                self.logs["llm1_response"].append(self.tokenizer.decode(self.response["input_ids"].squeeze()))
                if self.turn == self.n_turns:
                    reward = self.batch[2][0] in self.tokenizer.decode(self.response["input_ids"].squeeze()) # reward is 1 if the answer is correct
                    self.done = True
                else:
                    # send the response to the llm and get a new prompt
                    model_input = combine_inputs(self.tokenizer(list(self.batch[1]), return_tensors="pt", padding=True).to(self.device), self.response)
                    len_input = model_input["input_ids"].shape[1]
                    model_output = self.model.generate(**model_input, max_length=len_input + self.max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7, num_return_sequences=1).to(self.device)
                    # remove the prompt from the model output
                    model_output = model_output[:,model_input["input_ids"].shape[1]:]
                    self.logs["llm2_response"].append(self.tokenizer.decode(model_output.squeeze()))
                    # make prompt for next turn
                    # extract context and push to device
                    context = self.tokenizer(list(self.batch[0]), return_tensors="pt", padding=True)
                    for key in context:
                        context[key] = context[key].to(self.device)
                    self.prompt = combine_inputs(context, model_output)
                    self.response = {"input_ids": torch.zeros((1,1), dtype=torch.long).to(self.device), "attention_mask": torch.zeros((1,1), dtype=torch.long).to(self.device)}
            _obs = combine_inputs(self.prompt,self.response)
            # return combine_inputs(self.prompt,self.response), reward, self.done, {"turn":self.turn}
            # return {
            #     # pad for sb3
            #     # while creating rollout buffers, observations are concatenated for each key
            #     "prompt_or_input_encoded_pt": self.prompt["input_ids"].unsqueeze(),
            #     "prompt_or_input_attention_mask_pt": self.prompt["attention_mask"].unsqueeze(),
            #     "context_encoded_pt":[[]],
            #     "context_attention_mask_pt": [[]],
            #     "input_encoded_pt": self.response["input_ids"].unsqueeze(),
            #     "input_attention_mask_pt": self.response["attention_mask"].unsqueeze(),
            # }
            return _obs, reward, self.done, {"turn":self.turn}

    def render(self, mode="human"):
        """
        Renders the environment. Calling once per epoch gives best display
        """
        if mode == "human" and self.turn == self.n_turns:
            return "Prompt: " + self.logs["prompt"][0] \
                + "LLM1 Response: " + self.logs["llm1_response"][0] \
                + "Context: " + self.logs["context"][0] \
                + "LLM2 Response: " + self.logs["llm2_response"][-1] \
                + "LLM1 Response: " + self.logs["llm1_response"][1] \
                + "Expected Answer: " + self.logs["answer"][0]
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
