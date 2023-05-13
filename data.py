from torch.utils.data import Dataset
import numpy as np


class FruitDataset(Dataset):
    def __init__(
        self,
        path,
    ):
        # Grab the marco made text file and separate the different columns,
        self.data = np.loadtxt(path, dtype=str, delimiter="\t").T

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        return (
            self.data[1][item].replace("network", "person", 1).capitalize(),  # question
            self.data[0][item].capitalize(),  # context
            self.data[2][item].capitalize(),  # answer
        )
