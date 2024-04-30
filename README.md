# README

### config 
:: 사용자가 변경 할 수 있는 요소를 json파일로 관리
- data.json :: 데이터 관련 설정(데이터 경로, 데이터 형식)을 담은 파일
  - data_path
  - save_path
- model.json :: 모델 하이퍼파라미터 및 학습 관련 설정을 담은 파일
  - model_type
  - loss_type
  - epoch
  - batch_size
  - use_gpu
  - image_size
  - channel
  - lr
- main.json :: 각각의 컨피그를 연결해주는 설정 파일
  - data_config
  - model_config


### models 
:: 여러 모델이 구현된 파일
- Autoencoder.py => 구현 완료
- Variational_Autoencoder.py => 구현 완료
- DCGAN.py => 구현 중
- Cycle_GAN.py
- Style_GAN.py
- Diffusion.py

### result 
:: 학습 결과물을 저장하는 폴더
- history :: 학습 과정에 대한 정보를 저장하는 폴더(loss, 정확도 등)
- model :: 학습된 모델 가중치를 저장하는 폴더

### utils 
:: 데이터 로드 및 모델 관련 유틸리티를 포함하는 폴더
- config_utils.py
  - ConfigManager | Class 
    - read_config
    - check_list_of_config
- data_utils.py
  - CustomDataset | Class
    - __init__
    - __len__
    - __getitem__
- train_utils.py
  - Trainer | Class

### train.py 
:: 모델의 학습 및 검증 단계를 실행하고 결과를 저장

### 버전 정보
- cudnn 8.6
- cuda 11.3
- torch 1.12.1+cu113
- torchaudio 0.12.1+cu113
- torchvision 0.13.1+cu113

(+) argparse, 도커 이미지

---
정보
- 데이터 정적으로 사용
- 모델 학습 정적으로 사용 => 나중에 동적으로 변경할 것(how to 베이스 클래스에서 상속받아 각각의 모델 구현)

사용방법 
- config 폴더에 model.json에서  모델 타입, 로스 및 기타 파라미터 변경해서 사용

---
기타
- 텐서가 cpu상에 있으면 numpy 배열은 메모리 공간을 공유하므로 하나가 변하면 다른 하나도 변함
ex. 
a = torch.ones(1)
b = a.numpy()
a.add_(1) # _가 붙으면 inplace 연산임

.detach() 연산기록 추적을 안하고 분리 시켜줌
grad_fn: 미분값을 계산한 함수에 대한 정보 저장(어떤 함수에 대해서 backprop했는지)
