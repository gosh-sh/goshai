from gosh.model.gosh_model import GoshModel
from gosh.datasets.text_based_batching import TextBasedBatching
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList

import torch
import logging

logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids) -> None:
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
def generate(model: GoshModel, knn_memories, tokenizer: PreTrainedTokenizer, text: str, input_size: int, device) -> str:
    text_based_batching = TextBasedBatching(None, -1, input_size, tokenizer)
    chunks = text_based_batching.tokenize_and_split_to_chunks(text)
    input_chunks = [chunk[:-1] for chunk in chunks]

    # fill knn memory with embeddings from every input_chunk except last
    for input_chunk in input_chunks[:-1]:
        input = torch.unsqueeze(torch.from_numpy(input_chunk).to(device), 0)
        logger.debug(f"add to memory {input_chunk}")
        model(input, knn_memories = knn_memories, add_knn_memory = True)
    
    answer_chunks = []

    stop_token_ids = [0]

    generation_config = GenerationConfig(
        # temperature=0.2,
        # top_k=50,
        # top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=input_size,
        stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    )

    input_chunk = input_chunks[-1]
    input = torch.unsqueeze(torch.from_numpy(input_chunk).to(device), 0)
    output = model.generate(inputs=input, generation_config = generation_config, knn_memories = knn_memories, add_knn_memory = False)
    answer_chunks.append(output)
    logger.debug(f"output {output}")

    while len(output[0]) < input_size and output[0][-1].item() != tokenizer.eos_token_id:
        logger.debug(f"add to memory {input}")
        model(input, knn_memories = knn_memories, add_knn_memory = True)
        input = output
        output = model.generate(inputs=input, generation_config = generation_config, knn_memories = knn_memories, add_knn_memory = False)
        logger.debug(f"output {output}")
        answer_chunks.append(output[0])
    
    answer = torch.cat(answer_chunks)
    answer_text = tokenizer.decode(answer[0], skip_special_tokens = True)
    return answer_text
