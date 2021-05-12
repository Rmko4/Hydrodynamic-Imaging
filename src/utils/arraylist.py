import numpy as np

class ArrayList:
    def __init__(self, shape=(0,), dtype=float):
        """First item of shape is ingnored, the rest defines the shape"""
        self.shape = shape
        self.data = np.empty((100,*shape[1:]),dtype=dtype)
        self.capacity = 100
        self.size = 0

    def __getitem__(self, key):
        return self.data[key]

    def append(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.empty((self.capacity,*self.data.shape[1:]))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        return self.data[:self.size]