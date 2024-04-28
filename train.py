import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import json

from models.Autoencoder import Autoencoder
from utils.data_utils import CustomDataset
from utils.config_utils import ConfigManager

# 데이터, 학습에 필요한 설정 파일 읽어오기
config_manager = ConfigManager('./config/main.json')
data_config, model_config = config_manager.check_list_of_config()

# 학습 설정
epoch = model_config["epoch"]
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
optimizer = torch.optim.Adam(model.parameters(), model_config['lr'])
criterion = model_config['loss']

def train(autoencoder, train_loader):
    autoencoder.train()
    for step, x in enumerate(train_loader):
        train_x = x.view(-1, 1, 128*128).to(device)
        train_y = x.view(-1, 1, 128*128).to(device)

        encoded, decoded = autoencoder(train_x)

        loss = criterion(decoded, train_y) # 스텝별로 로스를 구함
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss:', loss)



view_data = [data_loader.dataset[i].view(-1, 1, 128*128).to(device) for i in range(5)]

for epoch in range(1, epoch+1):
    train(model, data_loader.dataset)

    decoded_dt = []
    for i in view_data:
        _, decoded_data = model(i)
        decoded_dt.append(decoded_data)

    f, a = plt.subplots(2, 5, figsize=(5, 2))
    print("[Epoch {}]".format(epoch))

    for i in range(5):
        img = np.reshape(view_data[i].to("cpu").data.numpy(), (128, 128))
        a[0][i].imshow(img, cmap='gray')
        a[0][i].set_xticks(()); a[0][i].set_yticks(())

    for ind, i in enumerate(view_data):
        _, decoded_data = model(i)
        img = np.reshape(decoded_data.to("cpu").data.numpy(), (128, 128))
        a[1][ind].imshow(img, cmap='gray')
        a[1][ind].set_xticks(()); a[1][ind].set_yticks(())
    plt.show()
