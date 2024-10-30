import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 256),  # Wider
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 64),  # Added layer
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class SyntheticDataset(Dataset):
    def __init__(self, size=100000):
        self.X = torch.randn(size, 10)
        self.y = torch.randint(0, 2, (size, 1)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    dist.init_process_group(
        backend='nccl',  # Changed from 'rccl' to 'nccl'
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def train():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Create model and move it to GPU
    model = SimpleModel().to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Create synthetic dataset and distributed sampler
    dataset = SyntheticDataset()
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        pin_memory=True
    )

    # Define loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # Important for proper shuffling
        total_loss = 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Only print from rank 0 to avoid duplicate outputs
            if rank == 0 and batch % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch} | Loss {loss.item():.4f}")
        
        # Calculate and print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch} complete | Average Loss: {avg_loss:.4f}")
    
    # Cleanup distributed training
    cleanup()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error during training: {e}")
        # Make sure to cleanup even if there's an error
        cleanup()