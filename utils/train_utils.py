import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

class Trainer:
    def __init__(self, model, model_type, optimizer, device, loss_type, img_size, history_save_path, model_save_path):
        self.model = model
        self.model_type = model_type
        self.optimizer = optimizer
        self.device = device
        self.loss_type = loss_type
        self.img_size = img_size
        self.history_save_path = history_save_path
        self.model_save_path = model_save_path
        self.data_num = 5
        self.train_history = {
            'total_loss': [],
            'bce_loss': [],
            'kld_loss': [],
            'mu_range_min': [],
            'mu_range_max': [],
            'logvar_range_min': [],
            'logvar_range_max': []
        }
    def custom_loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

    def comput_loss(self, recon_x, x, mu=None, logvar=None):
        if self.loss_type =='custom':
            return self.custom_loss(recon_x, x, mu, logvar)
        elif self.loss_type == 'mse':
            return F.mse_loss(recon_x, x, reduction='sum')
        elif self.loss_type == 'cross_entropy':
            return F.cross_entropy(recon_x, x)
        else:
            raise ValueError("Unsupported loss function")

    def view_data(self, model_type, data, image_size, device, model):

        if model_type == 'autoencoder':
            view_data = [data.dataset[i].view(-1, 1, image_size ** 2).to(device) for i in
                         range(5)]  # view는 기존의 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게 함

            f, a = plt.subplots(2, self.data_num, figsize=(self.data_num, 2))

            for i, x in enumerate(view_data):
                img = np.reshape(view_data[i].to("cpu").data.numpy(), (image_size, image_size))
                a[0][i].imshow(img, cmap='gray')
                a[0][i].set_xticks(());
                a[0][i].set_yticks(())

                if model_type == 'autoencoder':
                    _, output = model(x)
                elif model_type == 'vae':
                    output, _, _ = model(x)

                img_de = np.reshape(output.to("cpu").data.numpy(), (image_size, image_size))
                img_de = np.clip(img_de, 0, 1)

                a[1][i].imshow(img_de, cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())

        elif model_type == 'vae':
            view_data = [data.dataset[i].unsqueeze(0).to(device) for i in range(5)]
            f, a = plt.subplots(2, self.data_num, figsize=(self.data_num, 2))

            for i in range(5):
                img = np.reshape(view_data[i].to("cpu").data.numpy(), (128, 128))
                a[0][i].imshow(img, cmap='gray')
                a[0][i].set_xticks(());
                a[0][i].set_yticks(())

            for ind, i in enumerate(view_data):
                data_de, _, _ = model(i)
                img = np.reshape(data_de.to("cpu").data.numpy(), (128, 128))
                a[1][ind].imshow(img, cmap='gray')
                a[1][ind].set_xticks(());
                a[1][ind].set_yticks(())
        plt.show()

    def update_history(self, epoch, loss, bce=None, kld=None, mu=None, logvar=None):
        self.train_history['total_loss'].append(loss)
        if bce is not None and kld is not None:
            self.train_history['bce_loss'].append(bce)
            self.train_history['kld_loss'].append(kld)
        if mu is not None and logvar is not None:
            self.train_history['mu_range_min'].append(torch.min(mu).item())
            self.train_history['mu_range_max'].append(torch.max(mu).item())
            self.train_history['logvar_range_min'].append(torch.min(logvar).item())
            self.train_history['logvar_range_max'].append(torch.max(logvar).item())

        if (epoch + 1) % 5 == 0 and self.history_save_path:
            np.savetxt(self.history_save_path + f'/train_history_epoch_{epoch + 1}.csv',
                       np.column_stack((self.train_history.values())),
                       delimiter=',',
                       header='total_loss,bce_loss,kld_loss,mu_range_min,mu_range_max,logvar_range_min,logvar_range_max',
                       comments='')

    def train(self, data_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for step, x in enumerate(data_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()

                if self.model_type == 'autoencoder':
                    _, x_recon = self.model(x)
                    x_recon = x_recon.view(-1, 1, self.img_size, self.img_size)
                    loss = self.comput_loss(x_recon, x)
                elif self.model_type == 'vae':
                    x_recon, mu, logvar = self.model(x)
                    loss, bce, kld = self.comput_loss(x_recon, x, mu, logvar)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(data_loader)
            print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')
            self.update_history(epoch, epoch_loss)
            self.view_data(self.model_type, data_loader, self.img_size, self.device, self.model)
            torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch_{epoch}.h5')
