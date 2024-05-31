from torch import nn

### 테스트 코드 사용 ###
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
import os


class Encoder(nn.Module): # nn.Module 상속
    def __init__(self, latent_dim=128): # 모델에서 사용될 모듈과 다양한 함수등을 정의
        super(Encoder, self).__init__() # Autoencoder가 nn.Module(부모클래스)의 생성자 호출하여 부모 클래스의 속성과 메서드를 상속 받음

        # nn.Sequential => 신경망 정의: nn.Sequential 안의 모듈을 순차적으로 실행
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # torch.Size([3, 512, 512]) > torch.Size([16, 256, 256])
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # torch.Size([16, 256, 256]) > torch.Size([32, 128, 128])
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # torch.Size([32, 128, 128]) > torch.Size([64, 64, 64])
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # torch.Size([64, 64, 64]) > torch.Size([128, 32, 32])
            nn.ReLU(),
            nn.Flatten(), # torch.Size([128, 32, 32] > torch.Size([128*32*32])
            nn.Linear(128*32*32, latent_dim), # (128*32*32) > (128)
            nn.ReLU()
        )

    def forward(self, x):
        # x = torch.Size([27, 3, 512, 512])
        encoded = self.encoder(x)
        return encoded

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * 32 * 32), # torch.Size([128]) > torch.Size([128 * 32 * 32])
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 32, 32))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, 128, 32, 32)
        decoded = self.decoder(output)
        return decoded

class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

##### 테스트 코드 #####

# data load & transform
# class CustomDataset(Dataset):
#
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.image_paths = glob(os.path.join(data_dir, '*/*.png'))
#         self.class_names = os.listdir(self.data_dir)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         img = Image.open(image_path)
#         if self.transform:
#             img = self.transform(img)
#         return img
#
# device = torch.device("cuda:0" if True else "cpu")
#
# transform = transforms.Compose([transforms.Resize((512, 512)), ## 이미지 사이즈 작은게 1000*500정도 512*512으로 해도 될듯
#                                 # transforms.Grayscale(num_output_channels=1),
#                                 transforms.ToTensor()
#                                 ])
#
# pth = '/media/hskim/data/practice_data/'
# data = CustomDataset(pth, transform=transform)
# data_loader = DataLoader(data, batch_size=4, shuffle=True)


# for i in data_loader:
#     model = Autoencoder()
#     output = model(i)
#     print(output.shape)
k=10
# train