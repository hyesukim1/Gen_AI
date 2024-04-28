import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        global BATCH_SIZE
        return input.view(input.size()[0], -1).to(self.device)


class UnFlatten(nn.Module):
    def forward(self, input):
        global BATCH_SIZE
        return input.view(-1, 128, 16, 16).to(self.device)


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

        self.fc1 = nn.Linear(h_dim, z_dim).to(self.device)  # for mu right before reparameterization
        self.fc2 = nn.Linear(h_dim, z_dim).to(self.device)  # for logvar right before reparameterization

        self.fc3 = nn.Linear(z_dim, h_dim).to(self.device)  # right before decoding starts


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


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)  # be sure not to add activation functions here!
        logvar = torch.clamp(logvar, min=-4, max=4)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.bottleneck(h)

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD



# EPOCH = 500
# BATCH_SIZE = 64
#
# transform = transforms.Compose([transforms.Resize((128, 128)),
#                                 transforms.Grayscale(num_output_channels=1),
#                                 transforms.ToTensor()
#                                 ])
#
# pth = '/media/hskim/data/practice_data/'#'D:/data/kaggle/dataset'
# data = CustomDataset(pth, transform=transform)
# data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
#
# model = VAE()
# model = model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
# # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# view_data = [data_loader.dataset[i].unsqueeze(0).to(DEVICE) for i in range(5)]
#
#
# train_history = {
#     'total_loss': [],
#     'bce_loss': [],
#     'kld_loss': [],
#     'mu_range_min': [],
#     'mu_range_max': [],
#     'logvar_range_min': [],
#     'logvar_range_max': []
# }
#
# for epoch in range(1, EPOCH+1):
#     model.train()
#     for step, x in enumerate(data_loader):
#         images = x.float().to(DEVICE)
#         recon_images, mu, logvar = model(images)
#         loss, bce, kld = loss_function(recon_images, images, mu, logvar)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     scheduler.step()
#
#     print(f"Epoch [{epoch + 1}/{EPOCH}], total loss: {loss.item():.4f}, bce: {bce.item():.4f}, kld: {abs(kld.item()):.4f}")
#     print(f"Mu range: {torch.min(mu[0])} ~ {torch.max(mu[0])}, Logvar range: {torch.min(logvar[0])} ~ {torch.max(logvar[0])}")
#
#     if (epoch + 1) % 5 == 0:
#
#         torch.save(model.state_dict(), 'D:/result/kaggle/model/'+f'Epoch_{epoch}.h5')
#
#
#         train_history['total_loss'].append(loss.item())
#         train_history['bce_loss'].append(bce.item())
#         train_history['kld_loss'].append(abs(kld.item()))
#         train_history['mu_range_min'].append(torch.min(mu[0]).item())
#         train_history['mu_range_max'].append(torch.max(mu[0]).item())
#         train_history['logvar_range_min'].append(torch.min(logvar[0]).item())
#         train_history['logvar_range_max'].append(torch.max(logvar[0]).item())
#
#         np.savetxt('D:/result/kaggle/history/'+f'train_history_epoch_{epoch + 1}.csv',
#                    np.column_stack((train_history['total_loss'],
#                                     train_history['bce_loss'],
#                                     train_history['kld_loss'],
#                                     train_history['mu_range_min'],
#                                     train_history['mu_range_max'],
#                                     train_history['logvar_range_min'],
#                                     train_history['logvar_range_max'])),
#                    delimiter=',',
#                    header='total_loss,bce_loss,kld_loss,mu_range_min,mu_range_max,logvar_range_min,logvar_range_max',
#                    comments='')
#
#         f, a = plt.subplots(2, 5, figsize=(5, 2))
#
#
#         for i in range(5):
#             img = np.reshape(view_data[i].to("cpu").data.numpy(), (128, 128))
#             a[0][i].imshow(img, cmap='gray')
#             a[0][i].set_xticks(()); a[0][i].set_yticks(())
#
#         for ind, i in enumerate(view_data):
#             data, _, _  = model(i)
#             img = np.reshape(data.to("cpu").data.numpy(), (128, 128))
#             a[1][ind].imshow(img, cmap='gray')
#             a[1][ind].set_xticks(()); a[1][ind].set_yticks(())
#
#         plt.savefig('D:/result/kaggle/train_imgs/'+f'Epoch_{epoch}.png', bbox_inches='tight', pad_inches=0.1)








