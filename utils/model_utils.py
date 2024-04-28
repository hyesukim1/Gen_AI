import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt

class MyModel(nn.Module):
    """ 사용자 정의 모델 클래스 """

    def __init__(self, num_classes):
        """
        클래스 생성자

        :param num_classes: 클래스의 개수
        """
        super(MyModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        순전파 메서드

        :param x: 입력 데이터
        :return: 모델의 출력
        """
        return self.base_model(x)

def save_model(model, filepath):
    """ 모델을 파일에 저장하는 함수 """
    torch.save(model.state_dict(), filepath)

def load_model(filepath, num_classes):
    """ 파일에서 모델을 로드하는 함수 """
    model = MyModel(num_classes)
    model.load_state_dict(torch.load(filepath))
    return model

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

# train
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

def draw_train_image(data_loader, model, device, img_size):
    model.eval()
    with torch.no_grad():
        data_num = min(len(data_loader.dataset), 10)
        view_data = [data_loader.dataset[i].view(-1, 1, img_size**2).to(device) for i in range(data_num)]

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


def train(model, data_loader, criterion, optimizer, device, epochs, img_size):
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for step, x in enumerate(data_loader.dataset):
            train_x = x.view(-1, 1, img_size**2).to(device)
            train_y = x.view(-1, 1, img_size**2).to(device)

            optimizer.zero_grad()
            encoded, decoded = model(train_x)

            loss = criterion(decoded, train_y) # 스텝별로 로스를 구함
            loss.backward()
            optimizer.step()
        epoch_loss /= len(data_loader)
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')

        draw_train_image(data_loader, model, device)