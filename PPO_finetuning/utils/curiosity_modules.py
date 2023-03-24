from torch import nn
import torch

class CuriosityModule(nn.Module):
    def __init__(self, llm_hidden_size, device="cpu"):
        # inspired from https://arxiv.org/pdf/1705.05363.pdf
        # main difference is everything first goes through llm
        super().__init__()

        self.device = device
        self.encoder = ICMEncodeModuleFn(llm_hidden_size, device)
        self.predictor = ICMPredictorModuleFn(llm_hidden_size, device)
        self.invertor = ICMInverseModuleFn(llm_hidden_size, device)
        self.loss = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, hs_o, hs_new_o, hs_a):
            phi_st0 = self.encoder(hs_o)
            phi_st1 = self.encoder(hs_new_o)
            pred_phi_st1 = self.predictor(hs_a,phi_st0)
            pred_act = self.invertor(phi_st0,phi_st1)
            # calculate curiosity reward
            c_rew = torch.nn.functional.mse_loss(phi_st1, pred_phi_st1, reduction='none').mean(2).squeeze()
            inverse_loss = torch.nn.functional.mse_loss(pred_act, hs_a)
            print(f"curiosity reward: {c_rew.mean()}, inverse loss: {inverse_loss}")
            self.loss = c_rew + inverse_loss
            return c_rew
    
    def update(self):
        self.optimizer.zero_grad()
        # can use mean as batch size is constant
        self.loss.mean().backward()
        self.optimizer.step()

class ICMEncodeModuleFn(nn.Module):
    def __init__(self, llm_hidden_size, device="cpu"):
        super().__init__()
        self.device = device
        self.encode_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
        ).to(self.device)

    def forward(self, hidden_states):
        phi = self.encode_head_op(hidden_states.to(self.device))
        return phi.cpu()

class ICMPredictorModuleFn(nn.Module):
    def __init__(self, llm_hidden_size, device="cpu"):
        self.device = device
        super().__init__()
        self.step_ahead_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size  + 1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
        ).to(self.device)

    def forward(self, action, phi_st0):
        phi_st1 = self.step_ahead_op(torch.cat([action, phi_st0], dim=-1))
        return phi_st1.cpu()

class ICMInverseModuleFn(nn.Module):
    def __init__(self, llm_hidden_size, device="cpu"):
        super().__init__()
        self.device = device
        self.inverse_op = torch.nn.Sequential(
            torch.nn.Linear(1024*2, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, llm_hidden_size),
        ).to(self.device)

    def forward(self, st_0, st_1):
        act_pred = self.inverse_op(torch.cat([st_0, st_1], dim=-1))
        return act_pred.cpu()


