from torch import nn
import torch
from lamorel import BaseModuleFunction


class CuriosityModule(nn.Module):  # (BaseModuleFunction):
    def __init__(self, **kwargs):
        # inspired from https://arxiv.org/pdf/1705.05363.pdf
        # main difference is everything first goes through llm
        super().__init__()
        self.initialize(**kwargs)  # when using as basemodule deactivate this line

    def initialize(self, **kwargs):
        """
        Initialize module operations
        """
        # self.device = kwargs["device"] if "device" in kwargs else "cpu"
        self.lr = kwargs["lr"] if "lr" in kwargs else 1e-4
        self.optimizer = kwargs["optimizer"] if "optimizer" in kwargs else None

        self.encoder = ICMEncodeModuleFn(kwargs["llm_hidden_size"])
        self.predictor = ICMPredictorModuleFn(kwargs["llm_hidden_size"])
        self.invertor = ICMInverseModuleFn(kwargs["llm_hidden_size"])
        self.loss = None

    def auto_run(self, hs_new_o, hs_new_a):
        # now with double curiosity, using the same module!
        # first run as curiosity on o + a --> o' then on a + o' --> a'
        if not hasattr(self, "hs_o"):
            self.hs_o, self.hs_new_o, self.hs_a, self.hs_new_a = None, None, None, None

        self.hs_new_o = hs_new_o
        self.hs_new_a = hs_new_a

        if (
            self.hs_o is not None
            and self.hs_new_o is not None
            and self.hs_a is not None
        ):
            rew = self.forward(self.hs_o, self.hs_a, self.hs_new_o)
            rew2 = self.forward(self.hs_a, self.hs_new_o, self.hs_new_a)
            self.hs_o, self.hs_new_o, self.hs_a, self.hs_new_a = None, None, None, None
            return rew, rew2
        else:
            self.hs_o = hs_new_o
            self.hs_a = hs_new_a

    def forward(
        self, hs_o, hs_a, hs_new_o
    ):  # (self, forward_outputs, minibatch, tokenized_context,  **kwargs):
        """
        hs_o: hidden states of the previous observation
        hs_new_o: hidden states of the new observation
        hs_a: hidden states of the action
        """
        phi_st0 = self.encoder(hs_o)
        phi_st1 = self.encoder(hs_new_o)
        pred_phi_st1 = self.predictor(hs_a, phi_st0)
        pred_act = self.invertor(phi_st0, phi_st1)
        # calculate curiosity reward
        c_rew = (
            torch.nn.functional.mse_loss(phi_st1, pred_phi_st1, reduction="none")
            .mean(-1)
            .squeeze()
        )
        inverse_loss = torch.nn.functional.mse_loss(pred_act, hs_a)
        print(f"curiosity reward: {c_rew.mean()}, inverse loss: {inverse_loss}")
        self.loss = c_rew + inverse_loss
        return c_rew

    def get_loss(self):
        return self.loss

    def update(self):
        # legacy
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        # can use mean as batch size is constant
        self.loss.mean().backward()
        self.optimizer.step()


class ICMEncodeModuleFn(nn.Module):
    def __init__(self, llm_hidden_size):
        super().__init__()
        self.encode_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
        )

    def forward(self, hidden_states):
        phi = self.encode_head_op(hidden_states)
        return phi


class ICMPredictorModuleFn(nn.Module):
    def __init__(self, llm_hidden_size):
        super().__init__()
        self.step_ahead_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size + 1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
        )

    def forward(self, action, phi_st0):
        # print(action.shape, phi_st0.shape)
        pred_phi_st1 = self.step_ahead_op(torch.cat([action, phi_st0], dim=-1))
        return pred_phi_st1


class ICMInverseModuleFn(nn.Module):
    def __init__(self, llm_hidden_size):
        super().__init__()
        self.inverse_op = torch.nn.Sequential(
            torch.nn.Linear(1024 * 2, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, llm_hidden_size),
        )

    def forward(self, st_0, st_1):
        act_pred = self.inverse_op(torch.cat([st_0, st_1], dim=-1))
        return act_pred
