from torch import nn

class Autoencoder(nn.Module): # nn.Module 상속
    def __init__(self): # 모델에서 사용될 모듈과 다양한 함수등을 정의
        super(Autoencoder, self).__init__() # Autoencoder가 nn.Module(부모클래스)의 생성자 호출하여 부모 클래스의 속성과 메서드를 상속 받음

        # nn.Sequential => 신경망 정의: nn.Sequential 안의 모듈을 순차적으로 실행
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 4096)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 8192)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 16384)),
            nn.Sigmoid(),
        )
    def forward(self, x): # 모델에서 실행되어야 하는 연산을 정의
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


