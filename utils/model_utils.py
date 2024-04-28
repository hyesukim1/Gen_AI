import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# def draw_train_image(data_loader, model, img_size):
#     model.eval()
#     with torch.no_grad():
#         data_num = min(len(data_loader.dataset), 10)
#         view_data = [data_loader.dataset[i].view(-1, 1, img_size**2).to('cpu') for i in range(data_num)]
#
#         f, a = plt.subplots(2, data_num, figsize=(data_num, 2))
#         for i, x in enumerate(view_data):
#             img = np.reshape(x.to("cpu").numpy(), (128, 128))
#             a[0][i].imshow(img, cmap='gray')
#             a[0][i].set_xticks(());
#             a[0][i].set_yticks(())
#
#         for i, x in enumerate(view_data):
#             encoded, decoded = model(x)
#             img = np.reshape(decoded.to("cpu").numpy(), (128, 128))
#             a[1][i].imshow(img, cmap='gray')
#             a[1][i].set_xticks(());
#             a[1][i].set_yticks(())
#         plt.show()


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

        for step, x in enumerate(data_loader.dataset):
            train_x = x.view(-1, 1, img_size**2).to(device)
            train_y = x.view(-1, 1, img_size**2).to(device)

            optimizer.zero_grad()

            # autoencoder 기준
            encoded, decoded = model(train_x)

            loss = criterion(decoded, train_y)
            loss.backward()
            optimizer.step()

            # draw_train_image(data_loader, model)
        print('loss:', loss)

        data_num = 5
        view_data = [data_loader.dataset[i].view(-1, 1, img_size).to('cpu') for i in
                     range(data_num)]

        f, a = plt.subplots(2, data_num, figsize=(data_num, 2))

        for i, x in enumerate(view_data):
            img = np.reshape(x.to("cpu").numpy(), (128, 128))
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(());
            a[0][i].set_yticks(())

        for i, x in enumerate(view_data):
            encoded, decoded = model(x)
            img = np.reshape(decoded.to("cpu").numpy(), (128, 128))
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(());
            a[1][i].set_yticks(())
        plt.show()

        # print(f"Epoch [{epoch + 1}/{epoch}], total loss: {loss.item():.4f}, bce: {bce.item():.4f}, kld: {abs(kld.item()):.4f}")
        # print(f"Mu range: {torch.min(mu[0])} ~ {torch.max(mu[0])}, Logvar range: {torch.min(logvar[0])} ~ {torch.max(logvar[0])}")

        # if (epoch + 1) % 5 == 0:
        #     # history 저장
        #     train_history['total_loss'].append(loss.item())
        #             # train_history['bce_loss'].append(bce.item())
        #             # train_history['kld_loss'].append(abs(kld.item()))
        #             # train_history['mu_range_min'].append(torch.min(mu[0]).item())
        #             # train_history['mu_range_max'].append(torch.max(mu[0]).item())
        #             # train_history['logvar_range_min'].append(torch.min(logvar[0]).item())
        #             # train_history['logvar_range_max'].append(torch.max(logvar[0]).item())
        #
        #             np.savetxt('D:/result/kaggle/history/'+f'train_history_epoch_{epoch + 1}.csv',
        #                        np.column_stack((train_history['total_loss'],
        #                                         train_history['bce_loss'],
        #                                         train_history['kld_loss'],
        #                                         train_history['mu_range_min'],
        #                                         train_history['mu_range_max'],
        #                                         train_history['logvar_range_min'],
        #                                         train_history['logvar_range_max'])),
        #                        delimiter=',',
        #                        header='total_loss,bce_loss,kld_loss,mu_range_min,mu_range_max,logvar_range_min,logvar_range_max',
        #                        comments='')
        #
        #     # 모델 저장
        #     torch.save(model.state_dict(), model_save_path+f'epoch_{epoch}.h5')


def evaluate_model(model, data_loader, criterion, device):
    """ 모델을 평가하는 함수 """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss

'''
학습동안 이미지 보고싶은지 여부 분기

def train(model, criterion, optimizer, train_loader, epoch):
    for epoch in range(epoch):
        model.train()
        loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad() # 기울기 초기화
            outputs = model(inputs) # 순전파
            loss = criterion(outputs, labels) # 손실계산
            loss.backward() # 역전파
            optimizer.step() # 매개변수 업데이트
            loss += loss.item() * input.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
'''
