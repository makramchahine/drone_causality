import h5py
import numpy as np

class DataLoader:
    def __init__(self, cache, input, output):
        f = h5py.File(cache, 'r')

        # load data directly into RAM for speed
        self._x = (f[input][:]/255.).astype(np.float32)
        self._y = (f[output][:]).astype(np.float32)

        self.num_samples = self._x.shape[0]
        self.output_dim = self._y.shape[1]

    def get_batch(self, batch_size):
        ind = np.random.choice(self.num_samples, batch_size)
        return self._x[ind], self._y[ind]

    def get_num_samples(self):
        return self.num_samples

    def get_output_dim(self):
        return self.output_dim
