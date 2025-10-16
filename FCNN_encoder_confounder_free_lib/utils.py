import torch
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset

class StratifiedDataset(Dataset):
    """Custom dataset that includes sample IDs"""
    def __init__(self, x_data, y_data, sample_ids=None):
        self.x_data = x_data
        self.y_data = y_data
        self.sample_ids = sample_ids
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        if self.sample_ids is not None:
            return self.x_data[idx], self.y_data[idx], self.sample_ids[idx]
        else:
            return self.x_data[idx], self.y_data[idx]

def create_stratified_dataloader(x_train, y_train, batch_size, pad_incomplete=True, sampleid=None):
    """
    Create a stratified DataLoader ensuring that each batch maintains the overall class distribution.
    
    Parameters:
      x_train (Tensor): Input features.
      y_train (Tensor): Target labels.
      batch_size (int): Desired number of samples per batch.
      pad_incomplete (bool): If True, pad batches to reach the batch_size by repeating random samples.
      sampleid (numpy.ndarray, optional): String sample IDs for each data point (used to record predictions results).
    
    Returns:
      DataLoader: A PyTorch DataLoader that returns (x_batch, y_batch) if sampleid is None,
                  or (x_batch, y_batch, sampleid_batch) if sampleid is provided.
    """
    # Squeeze y_train to ensure it is a 1D tensor.
    labels = y_train.squeeze()
    unique_labels = labels.unique()
    
    # Handle sampleid - convert numpy array to list of strings
    if sampleid is not None:
        assert len(sampleid) == len(labels), "sampleid must have the same length as y_train"
        sampleid = sampleid.tolist()  # Convert numpy array to list
    
    # Count and compute proportions for each unique label.
    class_counts = {label.item(): (labels == label).sum().item() for label in unique_labels}
    total_samples = len(labels)
    class_proportions = {label: count / total_samples for label, count in class_counts.items()}

    # Compute the number of samples per class in a batch.
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

    # Distribute any remaining slots based on the highest remainders.
    remaining_slots = batch_size - total_samples_in_batch
    sorted_labels = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    for i in range(remaining_slots):
        label = sorted_labels[i % len(sorted_labels)][0]
        samples_per_class[label] += 1

    # Get indices for each label and shuffle them.
    class_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}
    for label in class_indices:
        indices = class_indices[label]
        class_indices[label] = indices[torch.randperm(len(indices))]

    def stratified_batches(class_indices, samples_per_class, batch_size):
        batches = []
        
        # Set up cursors for each label.
        class_cursors = {label: 0 for label in class_indices}
        num_samples = sum([len(indices) for indices in class_indices.values()])
        num_batches = math.ceil(num_samples / batch_size)

        # Build each batch.
        for _ in range(num_batches):
            batch = []
            
            for label, indices in class_indices.items():
                cursor = class_cursors[label]
                samples = samples_per_class[label]
                if cursor >= len(indices):
                    continue
                # Adjust sample count if there are not enough samples left.
                if cursor + samples > len(indices):
                    samples = len(indices) - cursor
                batch_indices = indices[cursor:cursor+samples]
                batch.extend(batch_indices.tolist())
                class_cursors[label] += samples
            
            # Shuffle within the batch
            if len(batch) > 0:
                shuffle_order = torch.randperm(len(batch))
                batch = torch.tensor(batch)[shuffle_order].tolist()
            
            batches.append(batch)
                
        return batches

    batches = stratified_batches(class_indices, samples_per_class, batch_size)

    class StratifiedBatchSampler(torch.utils.data.BatchSampler):
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            for batch in self.batches:
                yield batch

        def __len__(self):
            return len(self.batches)

    # Create dataset with sample IDs integrated
    dataset = StratifiedDataset(x_train, y_train, sampleid)
    
    batch_sampler = StratifiedBatchSampler(batches)
    g = torch.Generator()
    g.manual_seed(42)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, shuffle=False, num_workers=4, 
                            worker_init_fn=lambda wid: torch.manual_seed(42 + wid), generator=g)

    return data_loader