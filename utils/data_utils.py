from torch.utils.data import Dataset

import os
from glob import glob
from PIL import Image

class CustomDataset(Dataset):
    """ 데이터 관리 클래스 """

    def __init__(self, data_dir, transform=None):
        """
        클래스 생성자

        :param data_dir: 데이터셋이 있는 디렉토리 경로
        :param transform: 이미지에 적용할 변환
        """
        self.data_dir = data_dir
        self.image_paths = glob(os.path.join(data_dir, '*/*.png'))
        self.class_names = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        """
        데이터 샘플 수를 반환하는 메서드

        :return: 데이터 샘플 수 반환
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터 샘플을 반환하는 메서드

        :param idx: 반환할 데이터 샘플의 인덱스
        :return: 변환된 이미지 데이터 샘플
        """
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img