from torch import nn


class AdapterModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.adapter = nn.Linear(
            768, 768
        )  # I could make a more complex adapter layer... three layers with non linearities ? check litterature
        self.last_module = model.transformer.ln_f

    def forward(self, *args, **kwargs):
        # model.transformer.h[last index]
        outputs = self.model(*args, **kwargs)
        logits = self.adapter(outputs.hidden_states[-1])
        logits = self.last_module(logits)
        outputs.logits = logits
        return outputs
