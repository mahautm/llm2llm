import torch
from transformers import LogitsProcessor

class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, do_sample):
        self.eos_token_id = tokenizer.eos_token_id

        self.stop_word_ids = set(
            [
                idx
                for idx in range(len(tokenizer))
                if "." in tokenizer.convert_ids_to_tokens(idx)
            ]
        )
        self.vocab_size = len(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for i, input_id in enumerate(input_ids):
            if input_id[-1].item() in self.stop_word_ids:
                scores[i, : self.vocab_size] = torch.finfo().min
                scores[i, self.vocab_size :] = float("-inf")
                scores[i, self.eos_token_id] = 0.0
        return scores

def combine_inputs(input_1, input_2):
    """takes two tokenized inputs and combines them into one tokenized input"""
    # format the additional input
    if isinstance(input_2, torch.Tensor):
        input_2 = format_as_input(input_2)
    # combine the inputs
    output = {}
    for key in input_1:
        assert key in input_2, "keys from input_1 must also be in input_2"
        output[key] = torch.cat([input_1[key], input_2[key]], dim=1)
    return output


def format_as_input(sequence):
    """takes a sequence of tokens and formats it as an input for the model"""
    return {
        "input_ids": sequence,
        "attention_mask": torch.ones_like(sequence),
    }
