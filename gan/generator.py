import torch.nn as nn

# Define generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, first_channel=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 2 * 8 * 8 * first_channel, bias=False),
            nn.BatchNorm1d(2 * 8 * 8 * first_channel),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (first_channel, 2, 8, 8)),
            nn.ConvTranspose3d(first_channel, first_channel // 2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(first_channel // 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(first_channel // 2, first_channel // 4, (2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(first_channel // 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(first_channel // 4, first_channel // 8, (2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(first_channel // 8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(first_channel // 8, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)