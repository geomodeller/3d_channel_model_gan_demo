import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random

# Reset random seeds
def reset_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

# Define generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, first_channel=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 2 * 8 * 8 * first_channel, bias=False),
            nn.BatchNorm1d(2 * 8 * 8 * first_channel),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (first_channel, 2, 8, 8)),
            nn.ConvTranspose3d(first_channel, first_channel // 2, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(first_channel // 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(first_channel // 2, first_channel // 4, (3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(first_channel // 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(first_channel // 4, first_channel // 8, (3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(first_channel // 8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(first_channel // 8, 1, (3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3), bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channel=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(input_channel, 128, (3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(128, 64, (3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(64, 32, (3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(32, 16, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(16 * 2 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Loss functions
def discriminator_loss(real_output, fake_output):
    loss_fn = nn.BCELoss()
    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)
    real_loss = loss_fn(real_output, real_labels)
    fake_loss = loss_fn(fake_output, fake_labels)
    return real_loss + fake_loss

def generator_loss(fake_output):
    loss_fn = nn.BCELoss()
    real_labels = torch.ones_like(fake_output)
    return loss_fn(fake_output, real_labels)

# Training loop
def train(generator, discriminator, dataloader, epochs, latent_dim, device):
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            noise = torch.randn(batch.size(0), latent_dim, device=device)
            
            # Train discriminator
            disc_optimizer.zero_grad()
            real_output = discriminator(batch)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            disc_loss = discriminator_loss(real_output, fake_output)
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train generator
            gen_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            gen_loss = generator_loss(fake_output)
            gen_loss.backward()
            gen_optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f}")

# Dataset and DataLoader
class EnsembleDataset(data.Dataset):
    def __init__(self, ensemble):
        self.data = torch.tensor(ensemble, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
