import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

import os
from glob import glob
from PIL import Image


class Generator(nn.Module):
    '''
    생성자
    - 잠재 공간 벡터 z를 학습 이미지와 같은 사이즈를 가진 이미지를 생성하는 것
    - 마지막 출력 계층에서 데이터 tanh 함수에 통과시키는데 출력값 [-1, 1] 사이의 범위로 조정하기 위해서
    - 학습의 안정화를위해서 배치 정규화는 필수
    '''
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    '''
    구분자
    - 입력 이미지가 진짜 이미지인지 가까이미지인지 판별하는 전통적인 이진 분류 신경망
    - 마지막 출력 Sigmoid로 0~1사이의 확률값으로 조정
    - DCGAN 논문에서는 보폭이 있는 합성곱 계층을 사용한느 것이 신경망 내에서 풀링 함수를 학습하기 때문에 데이터를 처리하는 과정에서
    직접적으로 풀링 계층을 사용하는 것보다 더 유리함
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128*16*16
            nn.BatchNorm2d(128),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class CustomDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = glob(os.path.join(data_dir, '*/*.png'))
        self.class_names = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img

device = torch.device("cuda:0" if True else "cpu")
print("Using Device:", device)

netG = Generator().to(device)
netD = Discriminator().to(device)

# 손실함수
criterion = nn.BCELoss()

# 옵티마이저
optimizerD = optim.Adam(netD.parameters(), lr=0.005)
optimizerG = optim.Adam(netG.parameters(), lr=0.005)

'''
학습
1. Discriminator 학습
- 입력이 진짜인지 가까인지 판별하는 것 
- 진짜 데이터 배치를 netD에 통과 시킴 > 출력값으로 로스 계산 및 역전파 > 가짜 데이터 배치를 netD에 통과 시킴 > 출력값으로 로스 계산 및 역전파

2. Generator 학습
- log(D(G(z)))를 최대화하는 방식으로 바꿔서 학습
- 구분자를 이용해 생성자의 출력값 판별 후 진짜 라벨값을 기용해 G의 손실값값을 구해줌 => 손실값으로 변화도를 구하고 옵티마이저를 이용해 G의 가중치를 업데이트 시켜줌 
'''

img_list = []
G_losses = []
D_losses = []
iters = 0
epochs = 5

transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()
                                ])

pth = '/media/hskim/data/practice_data/'#'D:/data/kaggle/dataset'

data = CustomDataset(pth, transform=transform)
data_loader = DataLoader(data, batch_size=64, shuffle=True)


for epoch in range(epochs):
    for i, data in enumerate(data_loader, 0):
        netD.zero_grad()

        real_labels = torch.ones(data, 1).to(device)
        fake_labels = torch.zeros(data, 1).to(device)

        real_data = data.to(device)
        b_size = real_data.size(0)

        output = netD(real_data).view(-1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 64, 1, 1, device=device)

        fake = netG(noise)
        output = netD(fake.detach()).view(-1)

        #링크: https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html#id13