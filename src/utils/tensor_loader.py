import torch

# In case there is enough vram, use this dataset for faster training
class FastTensorDataLoader:
    def __init__(self, dataset, batch_size=1024, shuffle=False, device='cuda'):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=shuffle)
        all_data, all_targets = next(iter(data_loader))
        
        self.data = all_data.to(device)
        self.targets = all_targets.to(device)
        
        self.batch_size = batch_size
        self.n_samples = self.data.size(0)
    
    def __iter__(self):
        self.curr_idx = 0
        return self
    
    def __next__(self):
        if self.curr_idx >= self.n_samples:
            raise StopIteration
        
        end_idx = min(self.curr_idx + self.batch_size, self.n_samples)
        
        batch_data = self.data[self.curr_idx:end_idx]
        batch_targets = self.targets[self.curr_idx:end_idx]
        
        self.curr_idx = end_idx
        return batch_data, batch_targets

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size