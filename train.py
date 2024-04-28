import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from models.Autoencoder import Autoencoder
from models.Variational_Autoencoder import VAE
from utils.data_utils import CustomDataset
from utils.config_utils import ConfigManager
# from utils.model_utils import *

def train(model, data_loader, criterion, optimizer, device, epochs, img_size, history_save_path, model_save_path):
    # train_history = {
    #     'total_loss': [],
    #     'bce_loss': [],
    #     'kld_loss': [],
    #     'mu_range_min': [],
    #     'mu_range_max': [],
    #     'logvar_range_min': [],
    #     'logvar_range_max': []
    # }
    # 학습
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0

        for step, x in enumerate(data_loader.dataset):
            train_x = x.view(-1, 1, img_size**2).to(device)
            train_y = x.view(-1, 1, img_size**2).to(device)

            optimizer.zero_grad()

            # autoencoder 기준
            encoded, decoded = model(train_x)
            loss = criterion(decoded, train_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # draw_train_image(data_loader, model)
        epoch_loss /= len(data_loader)
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')

        data_num = 5
        view_data = [data_loader.dataset[i].view(-1, img_size, img_size).to(device) for i in
                     range(data_num)]

        f, a = plt.subplots(2, data_num, figsize=(data_num, 2))

        for i, x in enumerate(view_data):
            img = np.reshape(x.to("cpu").numpy(), (128, 128))
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(());
            a[0][i].set_yticks(())

        for i, x in enumerate(view_data):
            encoded, decoded = model(x)
            img = np.reshape(decoded.detach().to("cpu").numpy(), (128, 128))
            img = np.clip(img, 0, 1)
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(());
            a[1][i].set_yticks(())
        plt.show()

if __name__ == "__main__":
    # 데이터, 학습에 필요한 설정 파일 읽어오기
    config_manager = ConfigManager('./config/main.json')
    data_config, model_config = config_manager.check_list_of_config()

    # 학습 설정
    epochs = model_config["epoch"]
    batch_size = model_config['batch_size']

    # gpu 사용 여부
    device = torch.device("cuda:0" if model_config['use_gpu'] else "cpu")
    print("Using Device:", device)

    # model_config에서 channel이 3이면 color, 1이면 gray
    transform = transforms.Compose([
        transforms.Resize((model_config["image_size"], model_config["image_size"])),
        transforms.Grayscale(num_output_channels=1) if model_config["channel"] != 1 else lambda x: x,
        transforms.ToTensor()
    ])

    # load dataset
    path = data_config['data_path']
    data = CustomDataset(path, transform=transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # load model, loss
    model = Autoencoder().to(device)
    model2 = VAE(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), model_config['lr'])
    criterion = nn.MSELoss()

    # train
    train(model, data_loader, criterion, optimizer, device, epochs,  model_config["image_size"], data_config["save_path"][0], data_config["save_path"][1])

    # evaluation

    #

    k=10
