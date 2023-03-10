from torch.utils.data import Dataset
import numpy as np


class FruitDataset(Dataset):
    def __init__(self, path,):
        # Grab the marco made text file and separate the different columns,
        self.data = np.loadtxt(path, dtype=str, delimiter="\t").T
        

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        return (
            self.data[1][item], # question
            self.data[0][item], # context
            self.data[2][item], # answer
        )
