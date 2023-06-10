from gosh.train.save_load_model import save_qlora_model

import os
import shutil

class RollingCheckpointsSaver:
    def __init__(self, max_checkpoints: int = 30) -> None:
        self.checkpoints_paths = []
        self.index = 0
        self.max_checkpoints = max_checkpoints
        self.loss = None
        self.metric = None

    def save_checkpoint(self, model, output_path: str, step: int):
        checkpoint_path = os.path.join(output_path, f"checkpoint-{step}")
        save_qlora_model(model, checkpoint_path)
        self.index += 1
        self.checkpoints_paths.append(checkpoint_path)
        while len(self.checkpoints_paths) > self.max_checkpoints:
            checkpoint_path_to_remove = self.checkpoints_paths.pop(0)
            shutil.rmtree(checkpoint_path_to_remove)

    def save_checkpoint_with_loss(self, model, output_path: str, step: int, loss: float):
        if self.loss is None or self.loss > loss:
            self.save_checkpoint(model, output_path, step)
            self.loss = loss

    def save_checkpoint_with_metric(self, model, output_path: str, step: int, metric: float):
        if self.metric is None or self.metric < metric:
            self.save_checkpoint(model, output_path, step)
            self.metric = metric
