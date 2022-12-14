from data import FruitDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def combine_inputs(input_1, input_2):
    """takes two tokenized inputs and combines them into one tokenized input"""
    output = {}
    for key in input_1:
        assert key in input_2, "The two inputs should have the same keys"
        output[key] = torch.cat([input_1[key], input_2[key]], dim=1)
    return output


def format_as_input(sequence):
    """takes a sequence of tokens and formats it as an input for the model"""
    return {
        "input_ids": sequence,
        "attention_mask": torch.ones_like(sequence),
    }


def generate(model, input, max_length):
    """generates logits by selecting the highest probability token at each step using model.forward()"""
    if isinstance(input, torch.Tensor):
        input = format_as_input(input)
    all_logits = []
    generated_tokens = []
    logits = model.forward(**input).logits
    for _ in range(max_length):
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        input = combine_inputs(input, format_as_input(next_token))
        logits = model.forward(**input).logits
        all_logits.append(next_token_logits)
        generated_tokens.append(next_token)
        # check if the model has generated an end token
        if next_token.all() == model.config.eos_token_id:
            break
    return all_logits, torch.cat(generated_tokens, dim=1)


def train(model, dataloader, optimizer, tokenizer, device):
    old_logits = None
    # Train the model
    for epoch in range(10):
        for batch in dataloader:
            # get the data, tokenize it, and put it on the GPU
            questions = tokenizer.batch_encode_plus(
                list(batch[0]), return_tensors="pt", padding=True
            ).to(device)
            context = tokenizer.batch_encode_plus(
                list(batch[1]), return_tensors="pt", padding=True
            ).to(device)
            answers = tokenizer.batch_encode_plus(
                list(batch[2]), return_tensors="pt", padding="max_length", max_length=3
            ).to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass question -> lm1 (+ context) -> lm2 -> lm1 -> answer
            # at every step we remove the input from the output
            lm1_logits, lm1_tokens = generate(model, questions, 10)
            lm2_input = combine_inputs(
                context,
                format_as_input(lm1_tokens),
            )
            lm2_outputs = model.generate(
                **lm2_input,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=20
            )
            print(
                tokenizer.batch_decode(
                    lm2_outputs.sequences[:, len(lm2_input["input_ids"]) :]
                )
            )
            lm1b_logits, lm1b_tokens = generate(
                model, lm2_outputs.sequences[:, len(lm2_input["input_ids"]) :], 3
            )
            lm1b_logits = torch.cat(lm1b_logits)
            lm1b_softmax = torch.nn.functional.softmax(lm1b_logits)
            lm1b_logsoftmax = torch.nn.functional.log_softmax(lm1b_logits)
            # compute the loss
            # policy loss
            # select the logits of the correct answer
            expected_logits = torch.gather(
                lm1b_softmax, 1, answers.input_ids.flatten().unsqueeze(1)
            )
            # multiply for token from the same sequence, mean across batch
            R = (
                expected_logits.reshape(-1, len(answers.input_ids[0]))
                .prod(dim=1)
                .mean()
            )
            # compute according to policy choices for first message
            policy_loss = (
                (lm1b_logsoftmax * R).mean(dim=1) * answers.attention_mask.flatten()
            ).mean()
            # ppo : substract kl divergence between old and new policy
            if old_logits is not None:
                kl_regularisation = torch.nn.functional.kl_div(
                    lm1b_logsoftmax, old_logits, reduction="batchmean", log_target=True
                )
            else:
                kl_regularisation = torch.Tensor([0])
            old_logits = lm1b_logsoftmax.detach()
            # total loss = crossent loss + policy loss
            crossent_loss = (
                torch.nn.functional.cross_entropy(
                    lm1b_logits, answers.input_ids.flatten(), reduction="none"
                )
                * answers.attention_mask.flatten()
            ).mean()
            loss = crossent_loss + policy_loss - kl_regularisation
            print(tokenizer.batch_decode(lm1b_tokens))
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()
            # print the loss
            accuracy = (lm1b_tokens == answers.input_ids).float().mean()
            print(
                "cross ent loss: ",
                crossent_loss.item(),
                "policy loss: ",
                policy_loss.item(),
                "reward: ",
                R.item(),
                "kl ",
                kl_regularisation.item(),
                "acc ",
                accuracy.item(),
            )
    # save the model
    model.save_adapter("fine_tune", "./fine_tune")
    print("DONE TRAINING : saved adapter")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Using device", device)
    # Load the model
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    # Add a new adapter
    model.add_adapter("fine_tune")
    # Activate the adapter for training
    model.train_adapter("fine_tune")
    # get the tokenizer, setup padding
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
    # Load the dataset
    dataset = FruitDataset(".\data\mindless_dataset_randomized_train.txt")
    # Load the dataloader
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True)
    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, dataloader, optimizer, tokenizer, device)
