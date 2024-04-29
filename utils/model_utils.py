import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

def custom_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def view_data(data_num, data, image_size, device, model):

    view_data = [data.dataset[i].view(-1, 1, image_size**2).to(device) for i in range(5)] # view는 기존의 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게 함

    f, a = plt.subplots(2, data_num, figsize=(data_num, 2))

    for i, x in enumerate(view_data):
        img = np.reshape(view_data[i].to("cpu").data.numpy(), (image_size, image_size))
        a[0][i].imshow(img, cmap='gray')
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())

        _, decoded = model(x)
        img_de = np.reshape(decoded.to("cpu").data.numpy(), (image_size, image_size))
        img_de = np.clip(img_de, 0, 1)

        a[1][i].imshow(img_de, cmap='gray')
        a[1][i].set_xticks(());
        a[1][i].set_yticks(())
    plt.show()

def train(model, data_loader, criterion, optimizer, device, epochs, img_size, history_save_path, model_save_path):
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
            train_x = x.to(device) # torch.Size([64, 1, 128, 128]) #x.view(-1, 1, img_size, img_size).to(device)
            train_y = x.to(device) #x.view(-1, 1, img_size, img_size).to(device)

            optimizer.zero_grad() # 기울기 초기화

            # autoencoder 기준
            encoded, decoded = model(train_x)
            decoded = decoded.view(-1, 1, img_size, img_size)
            loss = criterion(decoded, train_y) # 손실 계산

            # VAE 기준
            # images = x.float().to(device)
            # recon_images, mu, logvar = model(images)
            # loss, bce, kld = custom_loss(recon_images, images, mu, logvar)

            loss.backward() # 역전파
            optimizer.step() # 매개변수 업데이트

            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')

        # 학습 동안에 이미지 확인
        view_data(5, data_loader, img_size, device, model)

        # print(
        #     f"Epoch [{epoch + 1}/{epoch}], total loss: {loss.item():.4f}, bce: {bce.item():.4f}, kld: {abs(kld.item()):.4f}")
        # print(
        #     f"Mu range: {torch.min(mu[0])} ~ {torch.max(mu[0])}, Logvar range: {torch.min(logvar[0])} ~ {torch.max(logvar[0])}")
        #
        # if (epoch + 1) % 5 == 0:
        #     # history 저장
        #     train_history['total_loss'].append(loss.item())
        #     train_history['bce_loss'].append(bce.item())
        #     train_history['kld_loss'].append(abs(kld.item()))
        #     train_history['mu_range_min'].append(torch.min(mu[0]).item())
        #     train_history['mu_range_max'].append(torch.max(mu[0]).item())
        #     train_history['logvar_range_min'].append(torch.min(logvar[0]).item())
        #     train_history['logvar_range_max'].append(torch.max(logvar[0]).item())
        #
        #     np.savetxt('D:/result/kaggle/history/' + f'train_history_epoch_{epoch + 1}.csv',
        #                np.column_stack((train_history['total_loss'],
        #                                 train_history['bce_loss'],
        #                                 train_history['kld_loss'],
        #                                 train_history['mu_range_min'],
        #                                 train_history['mu_range_max'],
        #                                 train_history['logvar_range_min'],
        #                                 train_history['logvar_range_max'])),
        #                delimiter=',',
        #                header='total_loss,bce_loss,kld_loss,mu_range_min,mu_range_max,logvar_range_min,logvar_range_max',
        #                comments='')


        # 모델 저장
        torch.save(model.state_dict(), model_save_path+f'epoch_{epoch}.h5')


# def evaluate_model(model, data_loader, criterion, device):
#     """ 모델을 평가하는 함수 """
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     accuracy = 100. * correct / total
#     average_loss = total_loss / len(data_loader)
#     return accuracy, average_loss
