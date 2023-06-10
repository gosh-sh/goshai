import random
import numpy as np

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from reactivex import Subject

import logging

logger = logging.getLogger(__name__)

class TextBasedBatching:
    def __init__(self, dataset: Dataset, batch_size: int, input_length: int, tokenizer: PreTrainedTokenizer) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_length = input_length
        self.tokenizer = tokenizer

        self.loop_counter = 0 # to regenerate random numbers in random_iterator

        self.on_text_end = Subject()

    def random_iterator(self, l: "list[str]", loop: bool):
        while True:
            indecies = list(range(len(l)))
            random.shuffle(indecies)
    
            self.loop_counter += 1
    
            for index in indecies:
                yield l[index]
            
            if not loop:
                break

    def tokenize_and_split_to_chunks(self, text_data: str):
        text_encoding = self.tokenizer(text_data)
        text_tokens = text_encoding.input_ids
        text_input_length = len(text_tokens)
        text_output_length = text_input_length + 1 # eos token at the end
        chunks_count = text_output_length // self.input_length + (1 if text_output_length % self.input_length > 0 else 0)
        
        text_chunks = []
        for i in range(chunks_count):
            chunk = np.full([self.input_length + 1], self.tokenizer.eos_token_id, dtype=int)
            chunk_start_pos = i * self.input_length
            chunk_end_pos = min(len(text_tokens), chunk_start_pos + self.input_length + 1)
            text_part = text_tokens[chunk_start_pos:chunk_end_pos]
            chunk[:len(text_part)] = text_part[:]
            text_chunks.append(chunk)
        text_chunks = np.stack(text_chunks)
        return text_chunks

    def text_chunks_in_batches(self, loop: bool = False):
        dataset_iterator = self.random_iterator(self.dataset, loop)

        batch_text_chunks = [[] for _ in range(self.batch_size)]

        chunk_indecies = [0 for _ in range(self.batch_size)]
        while True:
            batch_data: 'list[list[int]]' = []
            for batch_index in range(self.batch_size):
                chunk_index = chunk_indecies[batch_index]
                if len(batch_text_chunks[batch_index]) <= chunk_index:
                    self.on_text_end.on_next([batch_index])
                    try:
                        text, address = next(dataset_iterator)
                    except StopIteration:
                        return
                    while len(text) == 0:
                        logger.warn(f"Empty text for {address}")
                        try:
                            text, address = next(dataset_iterator)
                        except StopIteration:
                            return
                    text_chunks = self.tokenize_and_split_to_chunks(text)
                    batch_text_chunks[batch_index] = text_chunks
                    chunk_indecies[batch_index] = 0
                    chunk_index = chunk_indecies[batch_index]
                text_chunk = batch_text_chunks[batch_index][chunk_index]
                chunk_indecies[batch_index] += 1

                batch_data.append(text_chunk)
            batch_data = np.stack(batch_data)
            yield batch_data
