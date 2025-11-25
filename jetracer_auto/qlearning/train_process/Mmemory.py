import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # Zamień map na list comprehension i usuń Variable
        return [torch.cat(x, dim=0) for x in samples]

    def __len__(self):
        return len(self.memory)
