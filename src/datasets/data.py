import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
from tqdm import tqdm
import h5py as h5

import braceexpand
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

 
from typing import Sized, Optional, Iterator

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize, get_tokenizer

class CustomRandomSampler(RandomSampler):
    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, epoch: int=0) -> None:
        super(CustomRandomSampler, self).__init__(data_source, replacement, num_samples, generator)
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item()) + self.epoch # add epoch
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]   

class CsvDataset(Dataset):
    def __init__(self,
                 input_filename,
                 transforms,
                 img_key,
                 caption_key,
                 spurious_path=None,
                 bs=None,
                 seed=42,
                 sep="\t",
                 label_key="label",
                 tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.labels = list(map(int, df[label_key].tolist()))

        self.spurious = None
        if spurious_path:
            self.spurious = torch.load(spurious_path)

        self.bs = bs

        self.transforms = transforms

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = tokenize

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = str(self.captions[idx])
        labels = self.labels[idx]

        if self.spurious:

            self.spurious_template = random.choice(self.spurious)

            spurious_texts = self.spurious_template.format(texts[:-1])

            texts = self.tokenizer([texts])[0]
            spurious_texts = self.tokenizer([spurious_texts])[0]
            return images, texts, labels, spurious_texts

        texts = self.tokenizer([texts])[0]
        return images, texts, labels


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler,
                                                   DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2,
                                    transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)



def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.ft_data if is_train else args.val_data
    assert input_filename
    if args.get_labeled_csv:
        label_key = args.supervised_label_key

    else:
        label_key = None

    tokenizer =None
    label_key = "label"
    dataset = CsvDataset(input_filename,
                         preprocess_fn,
                         img_key=args.csv_img_key,
                         caption_key=args.csv_caption_key,
                         spurious_path=args.spurious_path,
                         bs=args.batch_size,
                         seed=args.seed,
                         sep=args.csv_separator,
                         label_key=label_key,
                         tokenizer=tokenizer)
    num_samples = len(dataset)
    # sampler = DistributedSampler(dataset) if args.distributed and is_train else None

    sampler = None
    if epoch:
        sampler = CustomRandomSampler(dataset, epoch=epoch) # use custom random sampler rather than shuffle=true

    shuffle = is_train and sampler is None 

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples #102493360
    dataloader.num_batches = len(dataloader) #200183

    return DataInfo(dataloader, sampler)




def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}."
            )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    data["train_ft"] = get_dataset_fn(args.ft_data,
                                      args.dataset_type)(args,
                                                         preprocess_train,
                                                         is_train=True,
                                                         epoch=epoch)

    return data
