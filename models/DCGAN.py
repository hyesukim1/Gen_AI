from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        global BATCH_SIZE
        return input.view(input.size()[0], -1).to(DEVICE)

class UnFlatten(nn.Module):
    def forward(self, input):
        global BATCH_SIZE
        return input.view(-1, 128, 16, 16).to(DEVICE)


class VAE(nn.Module):
    def __init__(self, device, h_dim=128*16*16, z_dim=64):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128*16*16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten()
       )

        self.fc1 = nn.Linear(h_dim, z_dim).to(DEVICE)  # for mu right before reparameterization
        self.fc2 = nn.Linear(h_dim, z_dim).to(DEVICE)  # for logvar right before reparameterization

        self.fc3 = nn.Linear(z_dim, h_dim).to(DEVICE)  # right before decoding starts


        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64,  kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )