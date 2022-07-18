from collections import namedtuple

import numpy as np


Datapoint = namedtuple('Datapoint',
                       ('input', 'target'))


class FIFOMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a datapoint."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Datapoint(*args)
        if self.position == 0:
            x = self.memory[0][0]
            y = self.memory[0][1]
            self.batch_in_sizes = {}
            self.n_inputs = len(x)
            for dim in range(len(x)):
                self.batch_in_sizes[dim] = np.array(x[dim]).shape
            self.batch_out_sizes = {}
            self.n_outputs = len(y)
            for dim in range(len(y)):
                self.batch_out_sizes[dim] = np.array(y[dim]).shape

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch"""

        batch = {}
        inputs = []
        outputs = []
        
        for dim in range(self.n_inputs):
            inputs.append(np.concatenate(batch_size *
                                    [np.expand_dims(np.zeros(
                                        self.batch_in_sizes[dim]
                                    ), 0)]))

        for dim in range(self.n_outputs):
            outputs.append(np.concatenate(batch_size *
                                     [np.expand_dims(np.zeros(
                                         self.batch_out_sizes[dim]
                                     ), 0)]))

        indices = np.random.choice(len(self.memory), batch_size, replace=False)

        count = 0
        for i in indices:
            x = self.memory[i][0]
            y = self.memory[i][1]

            for dim in range(len(x)):
                inputs[dim][count] = np.array(x[dim])

            for dim in range(len(y)):
                outputs[dim][count] = np.array(y[dim])

            count += 1

        return (inputs, outputs)

    def __len__(self):
        return len(self.memory)
