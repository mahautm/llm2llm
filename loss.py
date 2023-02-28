import torch
from baselines import MeanBaseline, NoBaseline

class PPOLoss:
    # inspired from Roberto's image captioner implementation (EGG --> rll)
    def __init__(self, baseline= "MeanBaseline", device="cuda"):
        self.baseline = MeanBaseline() if baseline == "MeanBaseline" else NoBaseline()
        self.device = device
        self.old_logits = None

    def __call__(self, reward, logits, kl_coeff=0.1, entropy_coeff=0.01):
        # TODO : get the mask
        # compute normalized log_probs of generated captions
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        # substract kl divergence between old and new policy
        if False: #self.old_logits is not None: TODO:kl_div
            kl_div = torch.nn.functional.kl_div(
                logits, self.old_logits, reduction="batchmean", log_target=True
            )
        else:
            kl_div = torch.tensor([0], device=self.device)
        # save the logits for the next iteration
        self.old_logits = logits.detach()
        # compute avg per timestep entropy of captioner distribution
        entropies = []
        for timestep in range(logits.shape[1]):
            dist = torch.distributions.Categorical(probs=logits[:, timestep].detach())
            ent = dist.entropy().to(self.device) # * mask[:, timestep]
            entropies.append(ent)
        entropy = torch.stack(entropies, dim=1)
        entropy = entropy.mean(1) # maybe if I batch it I'll need to .sum(-1) / logits.shape[1]

        # update the baseline
        baseline = self.baseline.predict(torch.tensor(reward))
        weighted_kl_div = kl_div * kl_coeff
        weighted_entropy = entropy * entropy_coeff
        policy_loss = ((reward - baseline)* logits).mean()

        optimized_loss = policy_loss - weighted_entropy + weighted_kl_div
        return optimized_loss


def legacy_ppo_loss(logits, old_logits, expected_tokens, mask, beta=0.1):
    # format the logits
    logits = torch.cat(logits)
    softmax = torch.nn.functional.softmax(logits)
    logsoftmax = torch.nn.functional.log_softmax(logits)
    # select the logits of the correct answer
    expected_logits = torch.gather(
        softmax, 1, expected_tokens.flatten().unsqueeze(1)
    )
    # compute according to policy choices for first message
    policy_loss = ((logsoftmax * expected_logits).sum(1) * mask.flatten()).mean()
    # substract kl divergence between old and new policy
    if old_logits is not None:
        kl_regularisation = torch.nn.functional.kl_div(
            logsoftmax, old_logits, reduction="batchmean", log_target=True
        )
    else:
        warnings.warn("No old logits provided, kl regularisation is 0")
        kl_regularisation = torch.tensor([0], device=device)
    return policy_loss - beta * kl_regularisation

def clippo_loss(logits):
    # clipped ppo loss
    return 

class curiosity_module(torch.nn.Module):
    """
    Curiosity module for the agent. The module is trained to predict the next state given the current state and action.
    the loss is the mean squared error between the predicted next state and the actual next state.
    reward = -loss
    reference: https://arxiv.org/pdf/1705.05363.pdf
    """
    def __init__(self, feature_length):
        super(curiosity_module, self).__init__()
        # define the models (might want to look at those architectures again)
        # input sequence and output sequence are the same size so they can use the same feature extractor
        # check that sequence order is really taken into account
        self.features = torch.nn.Sequential(
            torch.nn.TransformerEncoderLayer(feature_length, nhead=1, dim_feedforward=768),
            # nn.TransformerEncoderLayer(768, nhead=1, dim_feedforward=768),
            # nn.TransformerEncoderLayer(768, nhead=1, dim_feedforward=768),
            # nn.Linear(768, 768),
        )
        self.a_predictor = torch.nn.Sequential(
            torch.nn.Linear(feature_length + feature_length, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768),
            # nn.ReLU(),
            # nn.Linear(768, feature_length),
        )
        self.s_predictor = torch.nn.Sequential(
            torch.nn.TransformerEncoderLayer(feature_length + feature_length, nhead=1, dim_feedforward=768),
            # nn.TransformerEncoderLayer(768, nhead=1, dim_feedforward=768),
            # nn.TransformerEncoderLayer(768, nhead=1, dim_feedforward=768),
            # nn.Linear(768, 768),
        )
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # define the loss function
        self.mse_loss = torch.nn.MSELoss()
        self.crossent_loss = torch.nn.CrossEntropyLoss()

    def predict_train(self, s, a, s_next):
        f_s = self.features(s)
        f_sn = self.features(s_next)

        a_pred = self.a_predictor(torch.cat([f_s, f_sn], dim=1))
        fs_pred = self.s_predictor(torch.cat([f_s, a], dim=1))

        a_loss = self.crossent_loss(a_pred, a)
        s_loss = self.mse_loss(fs_pred, f_sn)

        optimizer.zero_grad()
        (a_loss + s_loss).backward()
        optimizer.step()

        return s_loss

def social_loss():
    # make the agent try to influence through its actions those of other agents
    pass