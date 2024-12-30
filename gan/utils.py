import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random
import matplotlib.pyplot as plt

# Dataset and DataLoader
class EnsembleDataset(data.Dataset):
    def __init__(self, ensemble):
        self.data = torch.tensor(ensemble, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Reset random seeds
def reset_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

    
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
def train(generator, discriminator, dataloader, epochs, latent_dim, device,
          save_every_epoch = None, 
          test_every_epoch = None,
          save_dir = 'saved_gan_models'):
    if (os.path.isdir(save_dir) == False):
        os.mkdir(save_dir)
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.5e-4)
    
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
            disc_loss = discriminator_loss(real_output, fake_output)/4
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train generator
            gen_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            gen_loss = generator_loss(fake_output)
            gen_loss.backward()
            gen_optimizer.step()

        if isinstance(save_every_epoch, int):
            if epoch % save_every_epoch == 0:
                torch.save(generator.state_dict(), os.path.join(save_dir,f"generator_weights_epoch_{epoch:05}.pth"))    

        if isinstance(test_every_epoch, int):
            if epoch % test_every_epoch == 0:
                noise = torch.randn(9, latent_dim, device=device)
                fake_images = generator(noise).detach().cpu().numpy().squeeze()
                plt.figure(figsize = (10,10))
                for i_subfigure in range(9):
                    plt.subplot(3,3,i_subfigure+1)
                    plt.imshow(fake_images[i_subfigure,0])
                plt.savefig(os.path.join(save_dir,f"outcomes_epoch_{epoch:05}.pdf"))
                plt.close()
                
        print(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss.item():.4f} | Disc Loss: {disc_loss.item():.10f}")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)