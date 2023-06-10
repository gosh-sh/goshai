from gosh.datasets.code_repository import CodeRepository
from gosh.logging_config import configure_logging
from gosh.datasets.load_dataset import get_repos

from transformers import AutoTokenizer
import json
from pydantic import BaseModel, Extra
import argparse
import os
import logging

logger = logging.getLogger(__name__)

class Config(BaseModel, extra=Extra.allow):
    dataset_path: str
    model_path: str

    @staticmethod
    def load(config_path):
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)     
            return Config.parse_obj(config_data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file")
    return parser.parse_args()    

def open_dataset(dataset_path: str):
    repos = get_repos(dataset_path)
    
    dataset = CodeRepository()
    dataset.set_extentions([".sol", ".tsol, .md"])
    for dir in repos:
        dataset.add_path(f"file:{dir}")

    return dataset

def dataset_size(dataset: CodeRepository, tokenizer):
    char_size = 0
    token_size = 0
    number = 1
    for i in range(len(dataset)):
        text, address = dataset[i]
        if len(text) == 0:
            logger.warning(f"Empty text for {address}")
        else:
            text_encoding = tokenizer(text)
            text_tokens = text_encoding.input_ids
            logger.debug(f"{number} {address} length {len(text)}, token length {len(text_tokens)}")
            char_size += len(text)
            token_size += len(text_tokens)
            number += 1

    return char_size, token_size

def main():
    configure_logging("dataset_size")

    args = parse_args()
    config = Config.load(args.config_path)
    
    dataset = open_dataset(config.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    char_size, token_size = dataset_size(dataset, tokenizer)

    print(f"dataset char length {char_size}, token length {token_size}")

if __name__ == "__main__":
    main()
