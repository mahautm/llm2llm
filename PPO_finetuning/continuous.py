import deepspeed
import torch
import numpy as np

# from utils.curiosity_modules import CuriosityModule
# from utils.critic_module import Critic
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_com_env_text import LLMComEnvText
from llm2llm.PPO_finetuning.utils.ppo_buffer import PPOBuffer
from accelerate.utils import DummyOptim, DummyScheduler
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
import hydra
from accelerate import Accelerator
from copy import deepcopy

# accelerator
accelerator = Accelerator()


class OPTLearnedPositionalLinear(torch.nn.Linear):
    """
    This module copies what is done for OPT learned positional embeddings
    but addapts it to the continuous case by working with weighted probabilities
    """

    def __init__(self, input_dim: int, output_dim: int):
        # ignored comment from original version:
        # (OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack)
        self.offset = 2
        super().__init__(input_dim, output_dim)

    def forward(
        self, attention_mask: torch.LongTensor, past_key_values_length: int = 0
    ):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (
            torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
        ).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]
        # TODO Matéo --> harcoded cast to bf16 should be revisited to work from the accelerator alone
        oh_positions = torch.nn.functional.one_hot(
            positions + self.offset, num_classes=self.weight.shape[1]
        ).to(torch.bfloat16)

        return super().forward(oh_positions.to(accelerator.device))


class ContinuousModel(torch.nn.Module):
    def __init__(self, model, tokenizer, accelerator):
        super().__init__()
        self.model = model
        self.accelerator = accelerator
        self.tokenizer = tokenizer

        # when using continuous word representations as input for the next turn, we need to project the logits to the embedding space
        # works with OPT family
        # three times model because unwrapping lora, and transformers
        self.log_to_emb = torch.nn.Linear(
            model.model.model.decoder.embed_tokens.weight.shape[0],
            model.model.model.decoder.embed_tokens.weight.shape[1],
        )
        self.log_to_emb.weight.data = (
            model.model.model.decoder.embed_tokens.weight.T.contiguous()
        )
        # isn't trainable
        self.log_to_emb.weight.requires_grad = False
        self.log_to_emb.bias = None

        self.att_to_pos = OPTLearnedPositionalLinear(
            model.model.model.decoder.embed_positions.weight.shape[0],
            model.model.model.decoder.embed_positions.weight.shape[1],
        )
        self.att_to_pos.weight.data = (
            model.model.model.decoder.embed_positions.weight.T.contiguous()
        )
        self.att_to_pos.weight.requires_grad = False
        self.att_to_pos.bias = None

    # le = torch.nn.Sequential(
    #     model.model.decoder.embed_tokens,
    #     model.model.decoder.embed_positions,
    #     model.model.decoder.final_layer_norm,
    # )

    def prepare_inputs(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.accelerator.device)
        return inputs

    def forward(self, inputs=None, return_sequences=False, **kwargs):
        # if self.disable_adapters:
        #     self.model.disable_adapter_layers()
        output = self.model(inputs, **kwargs)
        # if self.disable_adapters:
        #     self.model.enable_adapter_layers()
        if return_sequences:
            output["sequences"] = inputs["input_ids"]
        soft_logits = torch.nn.functional.log_softmax(output["logits"], dim=-1)
        eb = self.log_to_emb(soft_logits)
        if "attention_mask" in kwargs:
            pos = self.att_to_pos(kwargs["attention_mask"])
        else:
            pos = self.att_to_pos(torch.ones(kwargs["inputs_embeds"].shape[:2]))
        output["embeddings"] = self.model.model.model.decoder.final_layer_norm(eb + pos)
        return output


def init_game(config_args, accelerator):
    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # pad token to default to if not found
    pad_token = 50256

    # Create LLM agent
    model_path = config_args.lamorel_args.llm_args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", use_fast=False
    )
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id else pad_token
    accelerator.print(f"Creating {model_path} agent...")
    # with init_empty_weights:
    model = AutoModelForCausalLM.from_pretrained(model_path)

    accelerator.print(f"Done. Adding LoRa to {model_path} agent...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # config_args.rl_script_args.lora_r,
        lora_alpha=32,
        lora_dropout=0,
    )
    model = prepare_model_for_kbit_training(model)
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()
    lora_model.gradient_checkpointing_enable()  # reduce number of stored activations

    lora_model = ContinuousModel(lora_model, tokenizer, accelerator)

    # ensure minimal memory usage

    optimizer = DummyOptim(lora_model.parameters(), lr=config_args.rl_script_args.lr)

    env = LLMComEnvText(
        None,  # not used, not needed, we basically only use the env as a dataloader
        config_args.rl_script_args.dataset_path,
        max_length=config_args.rl_script_args.max_new_tokens,
        batch_size=config_args.rl_script_args.batch_size,
        affix=config_args.rl_script_args.affixes,
    )
    if config_args.rl_script_args.valid_dataset_path:
        venv = LLMComEnvText(
            None,
            config_args.rl_script_args.valid_dataset_path,
            max_length=config_args.rl_script_args.max_new_tokens,
            batch_size=config_args.rl_script_args.batch_size,
            affix=config_args.rl_script_args.affixes,
        )
    else:
        venv = None

    lr_scheduler = (
        DummyScheduler(
            optimizer,
            warmup_num_steps=config_args.rl_script_args.lr_warmup_steps,
            warmup_min_lr=config_args.rl_script_args.lr_warmup_min,
            warmup_max_lr=config_args.rl_script_args.lr_warmup_max,
            total_num_steps=config_args.rl_script_args.epochs
            * config_args.rl_script_args.ppo_updates,
        )
        if config_args.rl_script_args.lr_warmup_steps != 0
        else None
    )

    accelerator.print(f"Done. Preparing...")

    lora_model, env.dataloader, optimizer, lr_scheduler = accelerator.prepare(
        lora_model, env.dataloader, optimizer, lr_scheduler
    )
    accelerator.print(f"Done. Game init successful.")

    return lora_model, env, venv, optimizer, lr_scheduler, pad_token


def gen_with_grad(model, eb, max_new_tokens, **kwargs):
    # TODO Matéo --> could work with the "previous key" part for optimization?
    for i in range(max_new_tokens):
        o = model(inputs_embeds=eb, **kwargs)
        # add last new token to the input
        eb = torch.cat([eb, o.embeddings[:, -1:, :]], dim=1)
        # TODO Matéo --> check for eos?

    return eb[:, -max_new_tokens:, :], o.logits[:, -max_new_tokens:, :]


def game_forward(wmodel, env, loss, max_new_tokens, pad_token, **kwargs):
    # creating a wrapper and then unwrapping is counterproductive
    o = env.reset()
    input1 = wmodel.prepare_inputs(o)
    input2 = wmodel.prepare_inputs(env.batch[1])

    inputs_embeds = wmodel.model.model.model.decoder.final_layer_norm(
        wmodel.log_to_emb(
            torch.nn.functional.one_hot(
                input1.input_ids,
                num_classes=wmodel.model.model.model.decoder.embed_tokens.weight.shape[
                    0
                ],
            ).to(torch.bfloat16)
        )
        + wmodel.att_to_pos(
            input1.attention_mask,
        )
    )

    inputs_embeds2 = wmodel.model.model.model.decoder.final_layer_norm(
        wmodel.log_to_emb(
            torch.nn.functional.one_hot(
                input2.input_ids,
                num_classes=wmodel.model.model.model.decoder.embed_tokens.weight.shape[
                    0
                ],
            ).to(torch.bfloat16)
        )
        + wmodel.att_to_pos(input2.attention_mask)
    )
    # go throuth the model once
    print("input: ", o[0])
    print("input2: ", env.batch[1][0])
    print("answer: ", env.batch[2][0])
    eb1, logits1 = gen_with_grad(wmodel, inputs_embeds, max_new_tokens, **kwargs)
    print("llm1.1: ", wmodel.tokenizer.decode(logits1.argmax(dim=2)[0]))
    with torch.no_grad():
        # deactivate lora
        wmodel.model.disable_adapter_layers()
        # get kl loss
        # _, ologits1 = gen_with_grad(wmodel, inputs_embeds, max_new_tokens, **kwargs)
        # kl1 = torch.nn.functional.kl_div(ologits1, logits1, reduction="batchmean")
        # go through the model again adding previous continuous output and inputs2
        eb2, logits2 = gen_with_grad(
            wmodel,
            torch.cat([inputs_embeds2, eb1], dim=1),
            max_new_tokens,
            **kwargs,
        )
        print("llm2: ", wmodel.tokenizer.decode(logits2.argmax(dim=2)[0]))
        # get kl loss for the next step before reactivating lora
        # _, ologits3 = gen_with_grad(
        #     wmodel, torch.cat([inputs_embeds, eb2], dim=1), max_new_tokens, **kwargs
        # )
        # activate lora
        wmodel.model.enable_adapter_layers()
    # go through the model again adding previous continuous output and inputs1
    eb3, logits3 = gen_with_grad(
        wmodel, torch.cat([inputs_embeds, eb2], dim=1), max_new_tokens, **kwargs
    )
    print("llm1.2", wmodel.tokenizer.decode(logits3.argmax(dim=2)[0]))
    # kl3 = torch.nn.functional.kl_div(ologits3, logits3, reduction="batchmean")
    l = loss(
        logits3, wmodel.prepare_inputs(env.batch[2]).input_ids, pad_token=pad_token
    )
    # entropy
    dist1 = torch.distributions.Categorical(logits=logits1)
    dist2 = torch.distributions.Categorical(logits=logits2)
    dist3 = torch.distributions.Categorical(logits=logits3)
    entropy1 = dist1.entropy().mean()
    entropy2 = dist2.entropy().mean()
    entropy3 = dist3.entropy().mean()
    # print(kl1, kl3, entropy1, entropy2, entropy3)
    return (
        l,
        # kl1 + kl3 if kl1 + kl3 > 0 else torch.tensor(1000),
        0,
        (entropy1 + entropy2 + entropy3) / 3,
    )


# def multi_token_cross_entropy_loss(out_tok, target_tok):
#     loss = 0
#     for el in range(target_tok.shape[1]):
#         loss += torch.nn.functional.cross_entropy(out_tok[:, el, :], target_tok[:, el])
#     return loss
def multi_token_cross_entropy_loss_v2(out_tok, target_tok, pad_token):
    loss = 0
    # print(out_tok.shape, target_tok.shape)
    out_tok = torch.nn.functional.log_softmax(out_tok, dim=-1)
    loss += torch.nn.functional.cross_entropy(
        out_tok[:, : target_tok.shape[1], :].swapaxes(1, 2),
        target_tok,
    )
    # push everything beyond to the pad token
    out_pad = out_tok[:, target_tok.shape[1] :, :].swapaxes(1, 2)
    loss += torch.nn.functional.cross_entropy(
        out_pad,
        torch.ones_like(out_pad[:, 0, :]).long().to(loss.device) * pad_token,
    )
    return loss


def multi_token_cross_entropy_loss(out_tok, target_tok, pad_token):
    # loss = 0
    # for el in range(target_tok.shape[1]):
    #     loss.append(
    #         torch.nn.functional.cross_entropy(
    #             out_tok[:, el, :],
    #             target_tok[:, el],
    #             ignore_index=pad_token,
    #         )
    #     )

    # do not apply cross entropy to the pad token
    loss = torch.nn.functional.cross_entropy(
        out_tok[:, : target_tok.shape[1], :].swapaxes(1, 2),
        target_tok,
        ignore_index=pad_token,
    )

    return loss


@hydra.main(
    config_path="/homedtcl/mmahaut/projects/llm2llm/PPO_finetuning",
    config_name="local_gpu_config",
    version_base="1.1",
)
def main(config_args):
    # init_game
    wmodel, env, venv, optimizer, lr_scheduler, pad_token = init_game(
        config_args, accelerator
    )

    def accuracy(out_tok, target, pad_token=None):
        """
        loss function for validation
        which simply returns a count of the number of tokens that are the same
        """

        acc = sum(
            [
                wmodel.tokenizer.decode(out_tok[i].argmax(dim=1))
                == wmodel.tokenizer.decode(target[i])
                for i in range(len(target))
            ]
        )
        # print(wmodel.tokenizer.decode(out_tok[0].argmax(dim=1)))
        return acc / len(target)

    for epoch in range(config_args.rl_script_args.epochs):
        loss, kl, entropy = game_forward(
            wmodel,
            env,
            multi_token_cross_entropy_loss,
            config_args.rl_script_args.max_new_tokens,
            pad_token=pad_token,
        )
        weighted_kl_div = kl * config_args.rl_script_args.kl_loss_coeff
        weighted_entropy = entropy * config_args.rl_script_args.entropy_loss_coef
        optimized_loss = loss + weighted_kl_div - weighted_entropy
        print(
            "loss", loss, "kl", kl, "entropy", entropy, "weighted loss", optimized_loss
        )
        accelerator.backward(optimized_loss)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # validation
        if venv is not None:
            # use validation loss to count the number of tokens that are the same
            acc, _, _ = game_forward(
                wmodel,
                venv,
                accuracy,
                config_args.rl_script_args.max_new_tokens,
                pad_token=pad_token,
            )
            print("validation acc", acc)


if __name__ == "__main__":
    main()
