import numpy as np


class Memory:  # stored as ( s, a, r, s_ )

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.count = 0
        self.n_data = 0

    def add(self, sample):
        self.data[self.n_data] = sample
        self.n_data += 1

    def sample(self, n):
        batch = []
        for i in range(n):
            data = self.data[self.count]
            self.count += 1
            batch.append(data)
            if self.count == self.n_data:
                self.count = 0
                break
        return batch
