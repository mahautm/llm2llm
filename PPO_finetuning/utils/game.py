import cohere
import deepspeed
import torch
import numpy as np

from utils.curiosity_modules import CuriosityModule
from utils.critic_module import Critic
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_com_env_text import LLMComEnvText
from utils.ppo_buffer import PPOBuffer
from accelerate.utils import DummyOptim, DummyScheduler
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class CohereWrapper:
    def __init__(self, client):
        self.client = client

    def generate(self, input_text, max_new_tokens=10):
        _generated = self.client.batch_generate(input_text, max_tokens=max_new_tokens)
        _output = []
        for _g in _generated:
            _output.append(_g[0].text)
        return {"text": _output}


class CastOutputToFloat(torch.nn.Sequential):
    # for stability
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class StrToStrWrapper(torch.nn.Module):
    # create an str to str model wrapper
    def __init__(
        self, model, tokenizer, accelerator, critic=None, curiosity_module=None
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.critic = critic
        self.curiosity_module = curiosity_module
        self.accelerator = accelerator

    def prepare_inputs(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.accelerator.device)
        return inputs

    def forward(self, inputs, **kwargs):
        inputs = self.prepare_inputs(inputs)
        output = self.model(**inputs, **kwargs)
        return output

    def save_pretrained(self, save_directory):
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(
            save_directory,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(model),
        )

    def score(self, out_logs, expected, **kwargs):
        """only takes the first token from the answer.
        might become relevant to take the rest into account
        """
        expected = self.prepare_inputs(expected)
        _tokens = expected["input_ids"][:, -1:].expand(
            out_logs.shape[0], out_logs.shape[1]
        )
        # gather the logits of the input tokens
        _token_logs = out_logs.softmax(-1).gather(2, _tokens[:, :, None]).squeeze(-1)
        # ignore padding
        scores = (
            _token_logs.masked_fill(_tokens == self.tokenizer.pad_token_id, 1.0)
            .max(-1)
            .values
        )
        return scores

    def get_out_hs(self, hidden_states):
        _out_hs = []
        for _hs in hidden_states:
            _out_hs.append(_hs[-1][:, -1, :].squeeze())
        return torch.stack(_out_hs).swapaxes(0, 1)

    def generate(self, inputs, **kwargs):
        inputs = self.prepare_inputs(inputs)
        output = self.model.generate(
            **inputs,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            synced_gpus=True,
            **kwargs,
        )
        _out_logs = torch.stack(output["scores"])
        # _out_hs = output["hidden_states"][-1][-1][:, -1, :].squeeze()
        # _in_hs = output["hidden_states"][0][-1][:, -1, :].squeeze()
        result = {
            "text": self.tokenizer.batch_decode(
                output.sequences[
                    :, inputs["input_ids"].shape[1] :
                ],  # check if a -1 is required
                skip_special_tokens=True,
            ),
            "logits": _out_logs.swapaxes(0, 1),
            "out_sequence": output.sequences[:, inputs["input_ids"].shape[1] :],
            # "out_hs": _out_hs.to(torch.float32),
            # "in_hs": _in_hs.to(torch.float32),
        }
        if self.critic is not None:
            # do it for every word?
            if not hasattr(self, "out_hs"):
                self.out_hs = self.get_out_hs(output["hidden_states"])
            result["value"] = self.critic(self.out_hs)
        if self.curiosity_module is not None:
            if not hasattr(self, "out_hs"):
                self.out_hs = self.get_out_hs(output["hidden_states"])
            _in_hs = output["hidden_states"][0][-1][:, -1, :].squeeze()
            result["curiosity"] = self.curiosity_module.auto_run(_in_hs, self.out_hs)
        return result


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
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    # model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    accelerator.print(f"Done. Creating curiosity module...")
    # Setting up curiosity module
    c_module = (
        CuriosityModule(llm_hidden_size=model.config.hidden_size)
        if config_args.rl_script_args.cur_coef != 0
        else None
    )
    critic = Critic(llm_hidden_size=model.config.hidden_size)
    # Instantiate environment with non LoRa, untrained LLM
    if config_args.rl_script_args.cohere_key:
        # TODO: investigate this versioning issue
        # version = "2022-12-06"
        co = cohere.Client(api_key=config_args.rl_script_args.cohere_key)
        env_llm = CohereWrapper(co)
        # accelerator.print(f"using cohere version {version} as env_llm")
    else:
        env_llm = StrToStrWrapper(model, tokenizer, accelerator=accelerator)
    env = LLMComEnvText(
        env_llm,
        config_args.rl_script_args.dataset_path,
        max_length=config_args.rl_script_args.max_new_tokens,
        batch_size=config_args.rl_script_args.batch_size,
        affix=config_args.rl_script_args.affixes,
    )

    accelerator.print(f"Done. Adding LoRa to {model_path} agent...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config_args.rl_script_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # wrap the LoRa model
    wmodel = StrToStrWrapper(model, tokenizer, accelerator, critic, c_module)

    optimizer = DummyOptim(wmodel.parameters(), lr=config_args.rl_script_args.lr)
    # optimizer graveyard, rip
    # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
    #     wmodel.parameters(), lr=config_args.rl_script_args.lr
    # )
    # torch.optim.Adam(model.parameters(), lr=config_args.rl_script_args.lr)

    # TODO: this exclusion only makes sense with the warmup decay scheduler
    # if other are attempted, a new condition should be used
    lr_scheduler = (
        DummyScheduler(
            optimizer,
            warmup_num_steps=config_args.rl_script_args.lr_warmup_steps,
            warmup_min_lr=config_args.rl_script_args.lr_warmup_min,
            warmup_max_lr=config_args.rl_script_args.lr_warmup_max,
            total_num_steps=config_args.rl_script_args.epochs
            * config_args.rl_script_args.ppo_epochs,
        )
        if config_args.rl_script_args.lr_warmup_steps != 0
        else None
    )
    # Set up experience buffer
    buf = PPOBuffer(
        config_args.rl_script_args.steps_per_epoch
        * config_args.rl_script_args.batch_size,
        config_args.rl_script_args.gamma,
        config_args.rl_script_args.lam,
    )

    accelerator.print(f"Done. Preparing...")
    wmodel, env.dataloader, optimizer, lr_scheduler = accelerator.prepare(
        wmodel, env.dataloader, optimizer, lr_scheduler
    )
    return wmodel, env, optimizer, lr_scheduler, buf, pad_token
