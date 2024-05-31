import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

# 모델 모듈 임포트
from Gen_AI.models.Autoencoder import Autoencoder
from Gen_AI.models.Variational_Autoencoder import VAE
from Gen_AI.models.DCGAN import DCGAN

from Gen_AI.utils.main_utils import *

class Trainer:
    def __init__(self, model_type, learning_rate, device, loss_type, img_size, history_save_path, model_save_path):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if device else "cpu")
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

    def model_load(self):
        # print("Using Device1:", self.device)
        if self.model_type == 'autoencoder':
            model = Autoencoder().to(self.device)
        elif self.model_type == 'vae':
            model = VAE().to(self.device)
        elif self.model_type == 'dcgan':
            model = DCGAN().to(self.device)
        return model

    def set_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        return optimizer

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
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError("Unsupported loss function")

    def view_train_data(self, model_type, data, image_size, device, model):
        '''
        학습에 사용하는 걸 넣는게 아니라 data_loader에서 따로 뺀걸 넣어야함
        그래서 data.dataset[i].unsqueeze(0)을 to(device)로 넘겨줘야댐
        '''

        # 원래 이미지 5개 불러오기
        # view_data = [data.dataset[i].view(1, 1, -1).to(self.device) for i in range(5)]
        # view_data = [data.dataset[i].unsqueeze(0).to(self.device) for i in range(5)]
        data_iter = iter(data)
        view_data = next(data_iter) # torch.Size([8, 3, 512, 512]) 배치만큼 가져옴

        f, a = plt.subplots(2, 5, figsize=(14, 12))

        ori_imgs = []
        model_imgs = []
        for i in range(5):
            img = np.transpose(view_data[i].numpy(), (1, 2, 0))
            # ori_imgs.append(img)
            a[0][i].imshow(img)
            a[0][i].set_xticks(());
            a[0][i].set_yticks(())

            # 모델 통과한 이미지 5개 불러오기
            if model_type == 'autoencoder':
                output = model(view_data.to(device)) # torch.Size([8, 3, 512, 512]) .to("cuda")
                view_out = output[i].to("cpu").detach().numpy()
                out = np.transpose(view_out, (1, 2, 0))

                # model_imgs.append(out)
                a[1][i].imshow(out)
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())

            elif model_type == 'vae':
                output, _, _ = model(img)
                out = np.transpose(output[i].numpy(), (1, 2, 0))
                model_imgs.append(out)
                a[1][i].imshow(out)
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())
        plt.show()
        # 이미지 쇼 함수 호출
        # show_images

        # self.view_train_data(self.model_type, data_loader, self.img_size, self.device, model)

        # if model_type == 'autoencoder':
        #     view_data = [data.dataset[i].view(-1, 1, image_size ** 2).to(device) for i in range(5)]  # view는 기존의 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게 함
        #
        #     f, a = plt.subplots(2, self.data_num, figsize=(self.data_num, 2))
        #
        #     for i, x in enumerate(view_data):
        #         img = np.reshape(view_data[i].to("cpu").data.numpy(), (image_size, image_size))
        #         a[0][i].imshow(img, cmap='gray')
        #         a[0][i].set_xticks(());
        #         a[0][i].set_yticks(())
        #
        #         if model_type == 'autoencoder':
        #             _, output = model(x)
        #         elif model_type == 'vae':
        #             output, _, _ = model(x)
        #
        #         img_de = np.reshape(output.to("cpu").data.numpy(), (image_size, image_size))
        #         img_de = np.clip(img_de, 0, 1)
        #
        #         a[1][i].imshow(img_de, cmap='gray')
        #         a[1][i].set_xticks(());
        #         a[1][i].set_yticks(())
        #
        # elif model_type == 'vae':
        #     view_data = [data.dataset[i].unsqueeze(0).to(device) for i in range(5)]
        #     f, a = plt.subplots(2, self.data_num, figsize=(self.data_num, 2))
        #
        #     for i in range(5):
        #         img = np.reshape(view_data[i].to("cpu").data.numpy(), (128, 128))
        #         a[0][i].imshow(img, cmap='gray')
        #         a[0][i].set_xticks(());
        #         a[0][i].set_yticks(())
        #
        #     for ind, i in enumerate(view_data):
        #         data_de, _, _ = model(i)
        #         img = np.reshape(data_de.to("cpu").data.numpy(), (128, 128))
        #         a[1][ind].imshow(img, cmap='gray')
        #         a[1][ind].set_xticks(());
        #         a[1][ind].set_yticks(())
        # plt.show()

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
                       np.column_stack((self.train_history.values())), #  dict_values([[618419.5012454711, 392277.55536684784, 347877.01619112317, 323456.92232789856], [], [], [], [], [], []])
                       delimiter=',',
                       header='total_loss,bce_loss,kld_loss,mu_range_min,mu_range_max,logvar_range_min,logvar_range_max',
                       comments='')

    def train(self, data_loader, epochs):

        model = self.model_load()
        optimizer = self.set_optimizer(model)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for step, x in enumerate(data_loader):
                x = x.to(self.device) # 데이터를 같은 디바이스로 보내기
                optimizer.zero_grad()

                if self.model_type == 'autoencoder':
                    '''
                    loss: MSE
                    '''
                    x_recon = model(x)
                    loss = self.comput_loss(x_recon, x)
                # elif self.model_type == 'vae':
                #     '''
                #     loss: custom loss(KLD + BCE)
                #     '''
                #     x_recon, mu, logvar = self.model(x)
                #     loss, bce, kld = self.custom_loss(x_recon, x, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(data_loader)
            print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')
            self.update_history(epoch, epoch_loss)
            self.view_train_data(self.model_type, data_loader, self.img_size, self.device, model)
            # torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch_{epoch}.h5')
