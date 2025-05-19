import os
import torch
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, Subset, DataLoader
from .common import ImageFolderWithPaths, SubsetSampler

class CustomDataset(Dataset):
    def __init__(self, data_dir, split, transform):
        self.data_dir = os.path.join(data_dir, "waterbird_complete95_forest2water2")
        self.split = split
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.confounder_array = self.metadata_df['place'].values
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values

        self.targets = torch.tensor(self.y_array)
        self.targets_group = torch.tensor(self.group_array)
        self.targets_spurious = torch.tensor(self.confounder_array)

        self.transform = transform

        self.n_classes = 2
        self.n_groups = 4

        # Attribute for noisy label detection
        self.noise_or_not = np.abs(self.y_array - self.confounder_array)  # 1 if minor (noisy)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.targets[idx].item()
        y_group = self.targets_group[idx].item()
        y_spurious = self.targets_spurious[idx].item()

        return {
            'images': img,
            'labels': label,
            'groups': y_group,
            'spurious': y_spurious,
        }

class Waterbirds:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser('~/data'),
        batch_size=32,
        num_workers=32,
        # classnames='openai',
        # custom=False,
    ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = ["landbird", "waterbird"]

        self.train_dataset = CustomDataset(location, 'train', preprocess)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True
        )
        self.test_dataset = CustomDataset(location, 'test', preprocess)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False
        )






