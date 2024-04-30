from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.Autoencoder import Autoencoder
from models.Variational_Autoencoder import VAE
from utils.data_utils import CustomDataset
from utils.config_utils import ConfigManager
from utils.model_utils import * # 얘는 클래스없이 진행한 것
from utils.train_utils import *


if __name__ == "__main__":

    '''
    if __name__ == "__main__"
    - __name__ 변수의 값이 __main__인지 확인하는 코드는 현재 스크립트 파일이 프로그램의 시작점이 맞는지 판단하는 작업
    '''

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
        transforms.Grayscale(num_output_channels=1) if model_config["channel"] == 1 else lambda x: x,
        transforms.ToTensor()
    ])

    # load dataset
    path = data_config['data_path']
    data = CustomDataset(path, transform=transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)


    # load model
    if model_config['model_type'] == 'autoencoder':
        model = Autoencoder().to(device)

    elif model_config['model_type'] == 'vae':
        model = VAE().to(device)

    # set loss
    optimizer = torch.optim.Adam(model.parameters(), model_config['lr'])

    # train
    # 이미지 그릴건지, 학습 모델 저장할 건지, 학습 히스토리 저장할건지
    # train(model, model_config['model_type'], data_loader, loss_type, optimizer, device, epochs,  model_config["image_size"], data_config["save_path"][0], data_config["save_path"][1])

    # train
    trainer = Trainer(model, model_config['model_type'], optimizer, device, model_config['loss_type'], model_config["image_size"], data_config["save_path"][0], data_config["save_path"][0])
    trainer.train(data_loader, epochs)

    k=10
