"""
utils.py

General utility functions, including a stratified DataLoader generator.
"""

import math
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_stratified_dataloader(x_train: torch.Tensor,
                                 y_train: torch.Tensor,
                                 batch_size: int) -> DataLoader:
    """
    Create a stratified DataLoader that ensures class proportions
    are maintained in each batch.

    Parameters
    ----------
    x_train : torch.Tensor
        Input features tensor of shape (N, D).
    y_train : torch.Tensor
        Labels tensor of shape (N, 1) or (N,).
    batch_size : int
        Desired batch size.

    Returns
    -------
    DataLoader
        A custom DataLoader that yields stratified batches according
        to the class proportions.
    """
    labels = y_train.squeeze()
    unique_labels = labels.unique()

    # Count samples per class
    class_counts = {
        label.item(): (labels == label).sum().item()
        for label in unique_labels
    }
    total_samples = len(labels)
    class_proportions = {
        label: count / total_samples
        for label, count in class_counts.items()
    }

    # Allocate samples per class in each batch
    samples_per_class = {}
    remainders = {}
    total_samples_in_batch = 0

    for label, proportion in class_proportions.items():
        exact_samples = proportion * batch_size
        samples = int(math.floor(exact_samples))
        remainder = exact_samples - samples

        samples_per_class[label] = samples
        remainders[label] = remainder
        total_samples_in_batch += samples

    remaining_slots = batch_size - total_samples_in_batch
    sorted_labels = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    for i in range(remaining_slots):
        label = sorted_labels[i % len(sorted_labels)][0]
        samples_per_class[label] += 1

    # Collect indices for each class
    class_indices = {
        label.item(): (labels == label).nonzero(as_tuple=True)[0]
        for label in unique_labels
    }
    # Shuffle indices
    for label in class_indices:
        idx = class_indices[label]
        class_indices[label] = idx[torch.randperm(len(idx))]

    def stratified_batches(class_indices_dict, samples_dict, bs):
        batches = []
        class_cursors = {l: 0 for l in class_indices_dict}
        num_samples = sum(len(ix) for ix in class_indices_dict.values())
        num_batches = math.ceil(num_samples / bs)

        for _ in range(num_batches):
            batch = []
            for label, indices in class_indices_dict.items():
                cursor = class_cursors[label]
                needed = samples_dict[label]
                if cursor >= len(indices):
                    continue
                # Adjust if we don't have enough left
                needed = min(needed, len(indices) - cursor)
                batch_indices = indices[cursor: cursor + needed]
                batch.extend(batch_indices.tolist())
                class_cursors[label] += needed

            if batch:
                # Shuffle the combined batch
                batch_tensor = torch.tensor(batch)
                batch_tensor = batch_tensor[torch.randperm(len(batch_tensor))]
                batches.append(batch_tensor.tolist())
        return batches

    # Create final list of batches
    batches = stratified_batches(class_indices, samples_per_class, batch_size)

    class StratifiedBatchSampler(torch.utils.data.BatchSampler):
        def __init__(self, batch_indices):
            self.batch_indices = batch_indices

        def __iter__(self):
            for b in self.batch_indices:
                yield b

        def __len__(self):
            return len(self.batch_indices)

    dataset = TensorDataset(x_train, y_train)
    batch_sampler = StratifiedBatchSampler(batches)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    return data_loader
