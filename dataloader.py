import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import numpy as np
from PIL import Image  # Import the PIL library to convert NumPy arrays to PIL images
from fedlab.utils.dataset.partition import (
    CIFAR10Partitioner,
    CIFAR100Partitioner,
    MNISTPartitioner,
)
from avalanche.benchmarks.datasets import TinyImagenet
import pathlib


class TinyImageNetPartitioner(CIFAR10Partitioner):
    num_classes = 200


class EMNISTPartitioner(MNISTPartitioner):
    num_classes = 62


def load_data(args):
    """
    Load dataset and perform non-IID splitting.

    Args:
        args (argparse.Namespace): Arguments containing dataset parameters.

    Returns:
        tuple: Train and test DataLoaders.
    """

    if args.dataset == "emnist":

        # Define transformations for the training and testing sets
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),  # EMNIST has 28x28 images
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1751,), (0.3333,)
                ),  # Example mean and std for EMNIST (single channel)
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3333,)),  # Normalization for EMNIST
            ]
        )

        # Load EMNIST dataset for training and testing
        train_dataset = datasets.EMNIST(
            root=args.data_dir,
            split="byclass",
            train=True,
            download=True,
            transform=transform_train,
        )

        test_dataset = datasets.EMNIST(
            root=args.data_dir,
            split="byclass",
            train=False,
            download=True,
            transform=transform_test,
        )

        # Extract the targets from the train_dataset
        targets = np.array(train_dataset.targets)

        # Check if the targets are 1D
        if targets.ndim != 1:
            raise ValueError(f"Expected targets to be 1D but got {targets.ndim}D array")

        partitioner = EMNISTPartitioner(
            targets,
            args.num_clients,
            partition="noniid-labeldir",
            dir_alpha=args.noniid,
            seed=args.seed,
        )

    elif args.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform
        )

        targets = np.array(train_dataset.targets)

        if targets.ndim != 1:
            raise ValueError(f"Expected targets to be 1D but got {targets.ndim}D array")

        partitioner = CIFAR10Partitioner(
            targets,
            args.num_clients,
            balance=None,
            partition="dirichlet",
            dir_alpha=args.noniid,
            min_require_size=1,
            seed=args.seed,
        )

    elif args.dataset == "cifar100":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_dataset = datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        targets = np.array(train_dataset.targets)

        partitioner = CIFAR100Partitioner(
            targets,
            args.num_clients,
            balance=None,
            partition="dirichlet",
            dir_alpha=args.noniid,
            min_require_size=1,
            seed=args.seed,
        )

    elif args.dataset == "tinyimagenet":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = TinyImagenet(
            root=args.data_dir, train=True, transform=transform, download=True
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_dataset = TinyImagenet(
            root=args.data_dir, train=False, transform=transform, download=True
        )

        targets = np.array(train_dataset.targets)

        if targets.ndim != 1:
            raise ValueError(f"Expected targets to be 1D but got {targets.ndim}D array")

        partitioner = TinyImageNetPartitioner(
            targets,
            args.num_clients,
            balance=None,
            partition="dirichlet",
            dir_alpha=args.noniid,
            min_require_size=1,
            seed=args.seed,
        )

    else:
        raise ValueError("Unsupported dataset!")

    client_train_loaders = [
        DataLoader(
            DatasetSubset(train_dataset, partitioner.client_dict[i], args.batch_size),
            batch_size=args.batch_size,
            shuffle=True,
        )
        for i in range(args.num_clients)
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 10,
        shuffle=False,
    )
    sample, target = train_dataset[0]
    input_channels = sample.shape[0] if len(sample.shape) == 3 else 1
    num_classes = len(set(targets))
    image_size = sample.shape[1] if len(sample.shape) == 3 else sample.shape[0]
    args.input_channels = input_channels
    args.num_classes = num_classes
    args.image_size = image_size
    return client_train_loaders, test_loader


class DatasetSubset(Dataset):
    """
    Subset of a dataset containing only the corresponding data samples and targets.

    Args:
        dataset (Dataset): The whole dataset.
        indices (list): List of indices to include in the subset.
        batch_size (int): The size of the batches.
    """

    def __init__(self, dataset, indices, batch_size):
        self.data = [dataset.data[i] for i in indices]  # Store the actual data subset
        self.targets = [
            dataset.targets[i] for i in indices
        ]  # Store the actual targets subset
        self.transform = (
            dataset.transform
        )  # Store the transformation from the original dataset
        self.batch_size = batch_size
        self._balance_dataset()

    def _balance_dataset(self):
        num_samples = len(self.data)
        remainder = num_samples % self.batch_size
        if remainder != 0:
            num_to_duplicate = self.batch_size - remainder
            indices_to_duplicate = range(num_to_duplicate)
            # Extend the data and targets lists to balance the batches
            self.data.extend([self.data[i % num_samples] for i in indices_to_duplicate])
            self.targets.extend(
                [self.targets[i % num_samples] for i in indices_to_duplicate]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the data and target
        data_item = self.data[idx]
        target_item = self.targets[idx]

        if isinstance(
            data_item, (str, pathlib.Path)
        ):  # Handle file paths (TinyImageNet)
            data_item = Image.open(data_item).convert(
                "RGB"
            )  # Load image from path and convert to RGB
        elif not isinstance(data_item, Image.Image):  # Handle preloaded arrays/tensors
            if isinstance(data_item, torch.Tensor):
                data_item = (
                    data_item.cpu().numpy()
                )  # Convert tensor to NumPy array if on GPU
            data_item = Image.fromarray(data_item)

        # Apply transformation, if any
        if self.transform:
            data_item = self.transform(data_item)

        return data_item, target_item
