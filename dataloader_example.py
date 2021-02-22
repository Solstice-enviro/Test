"""
Here is a super simple dataloader example.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def create_fake_data_and_write_to_disk(output_directory: Path):
    """Simple method to create some fake data to play width."""
    np.random.seed(0)

    features = np.random.rand(width_dimension, height_dimension, number_of_channels)
    labels = np.random.choice(unique_labels, (width_dimension, height_dimension))

    features_file_path = output_directory.joinpath("features.npz")
    labels_file_path = output_directory.joinpath("labels.npz")
    np.savez(features_file_path, features)
    np.savez(labels_file_path, labels)
    return features_file_path, labels_file_path


class DemoDataset(Dataset):
    def __init__(self, features_file_path, labels_file_path):

        self._features_array = np.load(features_file_path)["arr_0"]
        self._labels_array = np.load(labels_file_path)["arr_0"]
        self._width, self._height = self._labels_array.shape

    def __len__(self):
        return 4

    def _get_slice_for_integer_index(self, integer_index: int):
        """Assume the input is a single integer between 0 and 3. This splitting is
        a bit hacky and only meant to be a quick and dirty demo. In the real code,
        we would need some serious logic here.
        """

        xmid = self._width // 2
        ymid = self._height // 2

        if integer_index == 0:
            x = self._features_array[:xmid, :ymid, :]
            y = self._labels_array[:xmid, :ymid]
        elif integer_index == 1:
            x = self._features_array[xmid:, :ymid, :]
            y = self._labels_array[xmid:, :ymid]
        elif integer_index == 2:
            x = self._features_array[:xmid, ymid:, :]
            y = self._labels_array[:xmid, ymid:]
        elif integer_index == 3:
            x = self._features_array[xmid:, ymid:, :]
            y = self._labels_array[xmid:, ymid:]
        else:
            raise ValueError("Something is wrong with the input index")

        return x, y

    def __getitem__(self, idx):
        """In theory, we could put all our code in this method, but I like to keep things nice
        and separated for readability. 
        """
        x, y = self._get_slice_for_integer_index(idx)
        return torch.from_numpy(x), torch.from_numpy(y)


width_dimension = 100
height_dimension = 50
number_of_channels = 7

unique_labels = [0, 1, 2, 3]

current_directory = Path(__file__).parent

if __name__ == "__main__":

    # let's create some fake data and write it to disk
    features_file_path, labels_file_path = create_fake_data_and_write_to_disk(
        current_directory
    )

    # create a Dataset object. This could be our "training" data.
    dataset = DemoDataset(features_file_path, labels_file_path)

    # it's very easy to get a dataloader from the dataset
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Let's check what kind of arrays the dataloader returns
    for counter, (features, labels) in enumerate(dataloader):
        print(f"Iteration {counter}: ")
        print(
            f"    - Features dimension [batch, width, height, channels] : {features.shape}"
        )
        print(f"    - Labels dimension [batch, width, height] : {labels.shape}")
