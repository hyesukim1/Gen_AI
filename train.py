# 파이토치 라이브러리 임포트
import torch
from torch.utils.data import DataLoader
from torchvision import transforms



# 유틸리티 모듈 임포트
from utils.data_utils import *
from utils.main_utils import *
from utils.train_utils import *


if __name__ == "__main__":

    '''
    if __name__ == "__main__"
    - __name__ 변수의 값이 __main__인지 확인하는 코드는 현재 스크립트 파일이 프로그램의 시작점이 맞는지 판단하는 작업
    '''

    # 데이터, 학습에 필요한 설정 파일 읽어오기
    main_conf = read_config('main.json')

    # 학습 설정
    epochs = main_conf["model_config"]["epoch"]
    batch_size = main_conf["model_config"]['batch_size']

    # model_config에서 channel이 3이면 color, 1이면 gray
    image_size = main_conf["model_config"]["image_size"]
    num_channels = main_conf["model_config"]["channel"]
    transform = transforms.Compose([
        transforms.Resize((main_conf["model_config"]["image_size"], main_conf["model_config"]["image_size"])),
        transforms.ToTensor()
    ])

    # load dataset
    path = main_conf["data_config"]["data_path"]
    data = CustomDataset(path, transform=transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # train
    trainer = Trainer(main_conf["model_config"]['model_type'],  main_conf["model_config"]['lr'], main_conf["model_config"], main_conf["model_config"]['loss_type'], main_conf["model_config"]["image_size"], main_conf["data_config"]["save_path"][0], main_conf["data_config"]["save_path"][1])
    trainer.train(data_loader, epochs)

    k=10


