import torch

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
