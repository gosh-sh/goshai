#%%
from gosh.model.knn_memory import KNNMemoryList, DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
from gosh.datasets.code_repository import CodeRepository, FILE_START_TOKEN
from gosh.datasets.text_based_batching import TextBasedBatching
from gosh.datasets.load_dataset import get_repos
from gosh.train.save_load_model import device_map_for_a16, make_qlora_config, load_qlora_model, load_model, load_LORA_model, load_qlora_tokenizer, save_random_state
from gosh.train.qlora import get_last_checkpoint, set_seed, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN
from gosh.checkpoints.rolling_checkpoints_saver import RollingCheckpointsSaver
from gosh.model.knn_memory_utils import knn_memories_context
from gosh.logging_config import configure_logging

import os
import torch
import argparse
import json
import datetime
import math
from pydantic import BaseModel
import tqdm
import random
from pathlib import Path
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np

import wandb

import logging

logger = logging.getLogger(__name__)

class Config(BaseModel):
    dataset_path: str
    model_path: str
    output_dir: str
    input_size: int
    batch_size: int
    openai_api_key: str

    @staticmethod
    def load(config_path):
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)     
            return Config(**config_data)

def datetime_string():
    return datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')

class TrainException(Exception):
    pass

#%%
def train(model, tokenizer, train_dataset: Dataset, val_dataset: Dataset, config: Config, input_size: int, steps: int, device: str):
    batch_size = config.batch_size
    train_start_datetime = datetime_string()

    MAX_GRAD_CLIP_NORM = 0.5
    learing_rate = 10**-5

    wandb.init(
        project="llm",
        config={
            "learning_rate": learing_rate,
            "batch_size": batch_size,
            "steps": steps,
        }
    )

    train_checkpoints_saver = RollingCheckpointsSaver()
    val_checkpoints_saver = RollingCheckpointsSaver()

    text_based_batching = TextBasedBatching(train_dataset, batch_size, input_size, tokenizer)
    text_chunks_iterator = text_based_batching.text_chunks_in_batches(loop=True)

    knn_memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
    val_knn_memories_directory = f"{DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY}_val"
    num_memory_layers = 1
    knn_mem_kwargs = dict(
        dim = model.config.n_embd // model.config.n_head,
        max_memories = 2**14,
        multiprocessing = False
    )

    optim = torch.optim.Adam(model.parameters(), lr = learing_rate)

    with knn_memories_context(knn_memories_directory, num_memory_layers, batch_size, **knn_mem_kwargs) as knn_memories:
        # clear memory for batch index on text end
        with text_based_batching.on_text_end.subscribe(
            on_next=lambda batch_indices: knn_memories.clear_memory(batch_indices=batch_indices)
        ):
            for step in tqdm.tqdm(range(steps), mininterval = 10., desc = 'training'):
                model.train()
            
                try: 
                    text_chunks = next(text_chunks_iterator)
                except StopIteration:
                    break
                input = text_chunks[..., :-1]
                labels = text_chunks[..., 1:]
                output = model(input_ids=torch.from_numpy(input).to(device), labels=torch.from_numpy(labels).to(device), knn_memories=knn_memories)
                loss = output.loss
                loss.backward()

                loss_value = loss.item()
                train_checkpoints_saver.save_checkpoint_with_loss(model, os.path.join(config.output_dir, f"train_{train_start_datetime}"), step, loss_value)

                if torch.isnan(loss).any():
                    logger.warning("loss is NaN")
                    raise TrainException("loss is NaN")

                logger.info(f"loss {loss.item()}")
                wandb.log({"loss": loss})

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_CLIP_NORM)
                optim.step()

                if torch.isnan(model.base_model.model.transformer.wte.weight).any():
                    logger.warning("wte has NaN after back prop")
                    raise TrainException("wte has NaN after back prop")

                optim.zero_grad()
                    
                steps_to_evaluate = 500 // batch_size
                if step % steps_to_evaluate == 0:
                    model.eval()
                    with torch.no_grad():
                        val_checkpoints_saver.save_checkpoint(model, os.path.join(config.output_dir, f"val_{train_start_datetime}"), step)

                        val_text_based_batching = TextBasedBatching(val_dataset, batch_size, input_size, tokenizer)

                        if True:
                            val_text_chunks_iterator = val_text_based_batching.text_chunks_in_batches()
                            with knn_memories_context(val_knn_memories_directory, num_memory_layers, batch_size, **knn_mem_kwargs) as val_knn_memories:
                                # clear memory for batch index on val text end
                                with val_text_based_batching.on_text_end.subscribe(
                                    on_next=lambda batch_indices: val_knn_memories.clear_memory(batch_indices=batch_indices)
                                ):
                                    total_loss = 0
                                    total_chunks_count = 0
                                    for text_chunks in tqdm.tqdm(val_text_chunks_iterator, mininterval = 10., desc = 'val loss'):
                                        input = text_chunks[..., :-1]
                                        labels = text_chunks[..., 1:]
                                        output = model(input_ids=torch.from_numpy(input).to(device), labels=torch.from_numpy(labels).to(device), knn_memories=val_knn_memories)
                                        loss = output.loss
                                        loss_value = loss.item()
                                        total_loss += loss_value
                                        total_chunks_count += 1
                                    total_loss /= total_chunks_count
                                    logger.info(f"val_loss {total_loss}")
                                    wandb.log({ "val_loss": total_loss })

    wandb.finish()

#%%
def load_dataset_from_repos(repos, train_dataset_ratio: float = 0.9):
    random.shuffle(repos)
    train_length = int(math.ceil(len(repos) * train_dataset_ratio))
    train_dirs, val_dirs = repos[:train_length], repos[train_length:]

    extentions = [".sol", ".tsol", ".md"]
    train_dataset = CodeRepository()
    train_dataset.set_extentions(extentions)
    for dir in train_dirs:
        train_dataset.add_path(f"file:{dir}")

    val_dataset = CodeRepository()
    val_dataset.set_extentions(extentions)
    for dir in val_dirs:
        val_dataset.add_path(f"file:{dir}")

    return train_dataset, val_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file")
    return parser.parse_args()    

def main():
    configure_logging("train")

    args = parse_args()
    config = Config.load(args.config_path)

    device = "cuda" # "cpu"
    device_map = device_map_for_a16() # device

    qlora_config = make_qlora_config(config.model_path, config.output_dir, device_map)

    checkpoint_dir, completed_training = get_last_checkpoint(qlora_config.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    is_train_finished = False

    set_seed(qlora_config.seed)

    input_size = config.input_size

    while not is_train_finished:
        model = load_qlora_model(qlora_config, checkpoint_dir)
        tokenizer = load_qlora_tokenizer(model, config.model_path, qlora_config.cache_dir)

        dataset_path = config.dataset_path
        train_dataset, val_dataset = load_dataset_from_repos(get_repos(dataset_path))

        save_random_state(
            os.path.join(
                config.output_dir,
                "train",
                f"random_state_{datetime_string()}.json"
            )
        )

        try:
            train(model, tokenizer, train_dataset, val_dataset, config, input_size, 100*1000, device)
            is_train_finished = True
        except TrainException as e:
            logger.exception(e)

if __name__ == "__main__":
    main()
