import torchvision
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class CustomMnistDataset(Dataset):
    """
    Reads the MNIST data from csv file given file path.
    """

    def __init__(self, csv_path, num_datapoints=None):
        super(CustomMnistDataset, self).__init__()

        self.df = pd.read_csv(csv_path)

        # Will be useful later while evaluating
        if num_datapoints is not None:
            self.df = self.df.iloc[0:num_datapoints]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Read
        img = self.df.iloc[index].filter(regex='pixel').values
        img = np.reshape(img, (28, 28)).astype(np.uint8)

        # Convert to Tensor
        img_tensor = torchvision.transforms.ToTensor()(img)  # [0, 1]
        img_tensor = 2 * img_tensor - 1  # [-1, 1]

        return img_tensor