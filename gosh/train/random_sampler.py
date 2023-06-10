from torch.utils.data import Sampler
import numpy as np

class RandomSampler(Sampler):
    def __init__(self, data_source_length: int):
        self.data_source_length = data_source_length
        self.seed = np.random.randint(0, 2**32)
        self.indices = self.generate_indices()
        self.current_idx = 0

    def generate_indices(self):
        state = np.random.get_state()
        np.random.seed(self.seed)
        
        indices = list(range(self.data_source_length))
        np.random.shuffle(indices)
        
        np.random.set_state(state)  # restore state
        return indices

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        else:
            idx = self.indices[self.current_idx]
            self.current_idx += 1
            return idx

    def state_dict(self):
        return { 'seed': self.seed, 'current_idx': self.current_idx }

    def load_state_dict(self, state_dict):
        self.seed = state_dict['seed']
        self.indices = self.generate_indices()
        self.current_idx = state_dict['current_idx']