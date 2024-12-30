import torch.nn as nn
# Define discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channel=1, first_channel = 128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(input_channel, first_channel, (3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d(first_channel, first_channel//2, (3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d( first_channel//2,  first_channel//4, (3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv3d( first_channel//4,  first_channel//8, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear( first_channel//8 * 2 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)