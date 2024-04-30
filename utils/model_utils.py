import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt


def custom_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def comput_loss(loss_type, recon_x=None, x=None, mu=None, logvar=None):
    if loss_type =='custom':
        return custom_loss(recon_x, x, mu, logvar)
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'cross_entropy':
        return nn.MSELoss()
    else:
        raise ValueError("Unsupported loss function")

def view_data(data_num, model_type, data, image_size, device, model):

    if model_type == 'autoencoder':
        view_data = [data.dataset[i].view(-1, 1, image_size**2).to(device) for i in range(5)] # view는 기존의 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게 함

        f, a = plt.subplots(2, data_num, figsize=(data_num, 2))

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
        f, a = plt.subplots(2, data_num, figsize=(data_num, 2))

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

def update_history(epoch, train_history, loss, bce=None, kld=None, mu=None, logvar=None, history_save_path=None):
    # 학습 도중 모델의 성능 지표를 저장
    train_history['total_loss'].append(loss)

    if bce is not None and kld is not None:
        train_history['bce_loss'].append(bce)
        train_history['kld_loss'].append(kld)

    if mu is not None and logvar is not None:
        train_history['mu_range_min'].append(torch.min(mu).item())
        train_history['mu_range_max'].append(torch.max(mu).item())
        train_history['logvar_range_min'].append(torch.min(logvar).item())
        train_history['logvar_range_max'].append(torch.max(logvar).item())

    # 주어진 간격으로 히스토리를 파일로 저장
    if (epoch + 1) % 5 == 0 and history_save_path:
        np.savetxt(history_save_path + f'/train_history_epoch_{epoch + 1}.csv',
                   np.column_stack((train_history['total_loss'],
                                    train_history['bce_loss'],
                                    train_history['kld_loss'],
                                    train_history['mu_range_min'],
                                    train_history['mu_range_max'],
                                    train_history['logvar_range_min'],
                                    train_history['logvar_range_max'])),
                   delimiter=',',
                   header='total_loss,bce_loss,kld_loss,mu_range_min,mu_range_max,logvar_range_min,logvar_range_max',
                   comments='')


def train(model, model_type, data_loader, loss_type, optimizer, device, epochs, img_size, history_save_path, model_save_path):
    data_loader1 = data_loader
    train_history = {
        'total_loss': [],
        'bce_loss': [],
        'kld_loss': [],
        'mu_range_min': [],
        'mu_range_max': [],
        'logvar_range_min': [],
        'logvar_range_max': []
    }

    # 학습
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0

        for step, x in enumerate(data_loader):

            if model_type == 'autoencoder':
                train_x = x.to(device)  # torch.Size([64, 1, 128, 128]) #x.view(-1, 1, img_size, img_size).to(device)
                train_y = x.to(device)  # x.view(-1, 1, img_size, img_size).to(device)
                # autoencoder 기준
                encoded, decoded = model(train_x)
                decoded = decoded.view(-1, 1, img_size, img_size)
                loss = comput_loss(loss_type)
                loss = loss(decoded, train_y) # 손실 계산

            elif model_type == 'vae':
                # VAE 기준
                images = x.float().to(device)
                recon_images, mu, logvar = model(images)
                loss, bce, kld = comput_loss(loss_type, recon_x=recon_images, x=images, mu=mu, logvar=logvar) # custom_loss(recon_images, images, mu, logvar)

            optimizer.zero_grad()  # 기울기 초기화
            loss.backward() # 역전파
            optimizer.step() # 매개변수 업데이트

            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')
        update_history(epoch, train_history, loss, bce=None, kld=None, mu=None, logvar=None, history_save_path=None)

        # 학습 동안에 이미지 확인
        view_data(5, model_type, data_loader, img_size, device, model)

        # 모델 저장
        torch.save(model.state_dict(), model_save_path+f'epoch_{epoch}.h5')