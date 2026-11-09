"""Opt-in source-balanced sampling, independent of musical/polyphony filtering."""
from collections import Counter
import torch
from torch.utils.data import Sampler


class SourceBalancedSampler(Sampler):
    def __init__(self, sources, seed=42):
        if not sources or any(not source for source in sources):
            raise ValueError('Source-balanced sampling requires every row source identity')
        counts = Counter(sources)
        self.weights = torch.tensor([1 / counts[source] for source in sources], dtype=torch.double)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.weights)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.multinomial(self.weights, len(self), replacement=True, generator=generator).tolist()
