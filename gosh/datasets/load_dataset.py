import os
import random
import math
from gosh.datasets.code_repository import CodeRepository

def get_repos(dataset_path):
    def get_sub_dirs(dir_path: str):
        sub_dirs = []
        for sub_dir_name in os.listdir(dir_path):
            sub_dataset_path = os.path.join(dir_path, sub_dir_name)
            if os.path.isdir(sub_dataset_path):
                sub_dirs.append(sub_dataset_path)
        return sub_dirs

    repos = sum(
        [get_sub_dirs(sub_dir) for sub_dir in get_sub_dirs(dataset_path)],
        []
    )
    return repos

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
