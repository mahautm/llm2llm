import gym
import torch
from torch.utils.data import DataLoader
from data import FruitDataset
from env_utils import combine_inputs
import numpy as np

# pad for llm

from torch.nn.utils.rnn import pad_sequence
class LLMComEnvText(gym.Env):
    """Environment where the agent communicates with a frozen language model. Input and output are text to accomodate lamorel"""
    def __init__(self, model, dataset_path, max_length=10, batch_size=1, n_turns=2, affix=None):
        super().__init__()
        # non-trained model
        self.model = model
        # Load the dataset
        dataset = FruitDataset(dataset_path)
        self.affix = affix
        # Load the dataloader
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.data = None
        # additional params
        self.max_length = max_length # max length of the response
        self.n_turns = n_turns

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
        
        self.logs = {"question" : self.batch[0], "context" : self.batch[1], "answer" : self.batch[2], "llm1_response": [], "llm2_response": []}
        self.turn = 0
        self.done = False  

        if self.affix is None:
           return self.batch[0]  # question
        else:
            return [self.affix[0][0] + b + self.affix[0][1] for b in self.batch[0]]

    def step(self, action):
        # actions are  a list of strings
        # the llm responds with a list of strings in turn one
        # in turn two the action is evaluated and reward is given
        if self.done:
            raise RuntimeError("Episode is done, call reset() to start a new episode.")
        else:
            assert len(action) == len(self.batch[0]), "Action must be a list of strings of length equal to the batch size."
            self.turn += 1
            self.logs["llm1_response"].append(action)
            if self.turn == self.n_turns:
                # reward could be adapted outside of env to match logits
                # reward is 1 per correct answer
                reward = (np.array(self.batch[2]) == np.array(action)).astype(int)
                return [None]*len(action), reward, True, {"turn":self.turn}

            # send the response to the llm and get a new question
            # combine the context and the action strings for each input, add suffix and prefix if needed
            if self.affix is None:
                _input = [self.batch[1][i] + action[i][0] for i in range(len(action))]
            else:
                _input = [self.affix[1][0] + self.batch[1][i] + self.affix[1][1] + action[i][0] + self.affix[1][2] for i in range(len(action))]

            model_output = self.model.generate(_input, max_new_tokens=self.max_length, pad_token_id=50256)
            obs = [_mo[0]["text"] for _mo in model_output]
            self.logs["llm2_response"].append(obs)
            return obs, 0, self.done, {"turn":self.turn}

    def render(self, mode="human"):
        """
        Renders the environment. Calling once per epoch gives best display
        """
        # if mode == "human" and self.turn == self.n_turns:
        #     return "Question: " + self.logs["prompt"][0] \
        #         + "LLM1 Response: " + self.logs["llm1_response"][0] \
        #         + "Context: " + self.logs["context"][0] \
        #         + "LLM2 Response: " + self.logs["llm2_response"][-1] \
        #         + "LLM1 Response: " + self.logs["llm1_response"][1] \
        #         + "Expected Answer: " + self.logs["answer"][0]
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
