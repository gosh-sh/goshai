from gosh.model.knn_memory import KNNMemoryList

from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock

def create_knn_memories(
        knn_memories_directory,
        num_memory_layers,
        batch_size,
        **knn_mem_kwargs
    ):
        return KNNMemoryList.create_memories(
            batch_size = batch_size,
            num_memory_layers = num_memory_layers,
            memories_directory = knn_memories_directory,
        )(**knn_mem_kwargs)

@contextmanager
def knn_memories_context(
    knn_memories_directory,
    num_memory_layers,
    batch_size,
    **knn_mem_kwargs
):
    knn_dir = Path(knn_memories_directory)
    knn_dir.mkdir(exist_ok = True, parents = True)
    lock = FileLock(str(knn_dir / 'mutex'))

    with lock:
        knn_memories = create_knn_memories(knn_memories_directory, num_memory_layers, batch_size, **knn_mem_kwargs)
        yield knn_memories
        knn_memories.cleanup()
