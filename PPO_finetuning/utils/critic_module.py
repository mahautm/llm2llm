from torch import nn
import torch


class Critic(nn.Module):
    def __init__(self, llm_hidden_size):
        super().__init__()
        self.val_estimator = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        )

    def forward(self, hidden_states):
        v = self.val_estimator(hidden_states)
        return v
