from gosh.model.gosh_model import GoshModel, DEFAULT_MEMORY_BLOCK_INDEX
from gosh.train.qlora import get_accelerate_model, ModelArguments, DataArguments, TrainingArguments, GenerationArguments, print_trainable_parameters, \
    smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, \
    find_all_linear_names
from gosh.datasets.code_repository import FILE_START_TOKEN
from transformers import AutoTokenizer

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

import torch
import os
import random
import numpy as np
import json
from transformers import AutoModelForCausalLM
import transformers
import argparse

KNN_MEMORY_ATTENTION_PARAMETERS_FILENAME = "knn_memory_attention_parameters.bin"
KNN_MEMORY_PARAMS_PREFIX = "attn.knn_attn"

def load_model(checkpoint_dir: str):
    GoshModel.register_model()

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    return model

def device_map_for_a16():
    device_map = {
        "transformer.wte": 0,
        "lm_head": 0,
        "transformer.wpe": 0,
        "transformer.drop": 0,
        "transformer.h.0": 1,
        "transformer.h.1": 1,
        "transformer.h.2": 1,
        "transformer.h.3": 1,
        "transformer.h.4": 1,
        "transformer.h.5": 1,
        "transformer.h.6": 2,
        "transformer.h.7": 2,
        "transformer.h.8": 2,
        "transformer.h.9": 2,
        "transformer.h.10": 2,
        "transformer.h.11": 2,
        "transformer.h.12": 3,
        "transformer.h.13": 3,
        "transformer.h.14": 3,
        "transformer.h.15": 3,
        "transformer.h.16": 3,
        "transformer.h.17": 3,
        "transformer.h.18": 4,
        "transformer.h.19": 4,
        "transformer.h.20": 4,
        "transformer.h.21": 4,
        "transformer.h.22": 4,
        "transformer.h.23": 4,
        "transformer.h.24": 5,
        "transformer.h.25": 5,
        "transformer.h.26": 5,
        "transformer.h.27": 5,
        "transformer.h.28": 6,
        "transformer.h.29": 6,
        "transformer.h.30": 6,
        "transformer.h.31": 6,
        "transformer.h.32": 6,
        "transformer.h.33": 6,
        "transformer.h.34": 7,
        "transformer.h.35": 7,
        "transformer.h.36": 7,
        "transformer.h.37": 7,
        "transformer.h.38": 7,
        "transformer.h.39": 7,
        "transformer.ln_f": 7
    }

    return device_map

def make_qlora_config(checkpoint_dir: str, output_dir: str, device_map = "cpu"):
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args = \
        hfparser.parse_dict({

        "model_name_or_path": checkpoint_dir,
        "output_dir": output_dir,
        "device_map": device_map,

        "max_memory_MB": 16000,

        "logging_steps": 10,
        "save_strategy": "steps",
        "data_seed": 42,
        "save_steps": 500,
        "save_total_limit": 40,
        "evaluation_strategy": "steps",
        "eval_dataset_size": 1024,
        "max_eval_samples": 1000,
        "per_device_eval_batch_size": 1,
        "max_new_tokens": 32,
        "dataloader_num_workers": 3,
        "group_by_length": True,
        "logging_strategy": "steps",
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": True,
        "do_mmlu_eval": True,
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_modules": all,
        "double_quant": True,
        "quant_type": "nf4",
        "bf16": True,
        "bits": 4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "constant",
        "gradient_checkpointing": True,
        "dataset": "oasst1",
        "source_max_len": 16,
        "target_max_len": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_steps": 1875,
        "eval_steps": 187,
        "learning_rate": 0.0002,
        "adam_beta2": 0.999,
        "max_grad_norm": 0.3,
        "lora_dropout": 0.05,
        "weight_decay": 0.0,
        "seed": 0
        }, allow_extra_keys=True)
        # hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    return args

def load_qlora_model(config, checkpoint_dir: 'str | None'):
    GoshModel.register_model()
    model = get_accelerate_model(config, checkpoint_dir)
    model.config.use_cache = False
    if checkpoint_dir is not None:
        device = config.device_map if type(config.device_map) == str else f"cuda:{config.device_map[f'transformer.h.{DEFAULT_MEMORY_BLOCK_INDEX}']}"
        load_knn_memory_parameters(model, os.path.join(checkpoint_dir, KNN_MEMORY_ATTENTION_PARAMETERS_FILENAME), device)
    print_trainable_parameters(config, model)
    print('loaded model')
    return model

def load_LORA_model(model, qlora_config, checkpoint_dir: 'str | None'):
    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    else:
        print(f'adding LoRA modules...')
        modules = find_all_linear_names(qlora_config, model)
        print(f"LoRA modules {modules}")
        config = LoraConfig(
            r=qlora_config.lora_r,
            lora_alpha=qlora_config.lora_alpha,
            target_modules=modules,
            lora_dropout=qlora_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if qlora_config.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        # if 'lm_head' in name or 'embed_tokens' in name:
        if 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if qlora_config.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

def load_qlora_tokenizer(model, model_path: str, cache_dir: str, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        padding_side=padding_side,
        use_fast=False # Fast tokenizer giving issues.
    ) # tokenizer.pad_token = tokenizer.eos_token
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if FILE_START_TOKEN not in tokenizer.encoder:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=[FILE_START_TOKEN]),
            tokenizer=tokenizer,
            model=model,
        )
    return tokenizer

def save_qlora_model(model, output_dir: str):
    model.save_pretrained(output_dir)
    save_knn_memory_parameters(model, os.path.join(output_dir, KNN_MEMORY_ATTENTION_PARAMETERS_FILENAME))

def load_knn_memory_parameters(model, file_path: str, device_name: str, knn_memory_prefix: str = KNN_MEMORY_PARAMS_PREFIX):
    knn_memory_parameters = torch.load(file_path, map_location=torch.device(device_name))

    model_state_dict = model.state_dict()
    for key in knn_memory_parameters.keys():
        assert knn_memory_prefix in key
        layer = model_state_dict[key]
        layer.data = knn_memory_parameters[key].data
        layer.requires_grad = True

def save_knn_memory_parameters(model, file_path: str, knn_memory_prefix: str = KNN_MEMORY_PARAMS_PREFIX):
    knn_memory_parameters = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if knn_memory_prefix in key:
            knn_memory_parameters[key] = model_state_dict[key]
    
    torch.save(knn_memory_parameters, file_path)

def get_random_state():
    def convert_to_python_structure(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, tuple):
            return tuple(convert_to_python_structure(item) for item in data)
        elif isinstance(data, list):
            return [convert_to_python_structure(item) for item in data]
        else:
            return data
    return dict(
        random = random.getstate(),
        np = convert_to_python_structure(np.random.get_state()),
        torch = torch.get_rng_state().tolist()
    )

def save_random_state(file_path: str, state = get_random_state()):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(state, file)
