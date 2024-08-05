# README

### main.json 
  - data_config: 데이터 관련 설정(데이터 경로, 데이터 형식)을 담은 파일
    - data_path
    - save_path
  - model_config: 모델 하이퍼파라미터 및 학습 관련 설정을 담은 파일
    - model_type
    - loss_type
    - epoch
    - batch_size
    - use_gpu
    - image_size
    - channel
    - lr
    
### models 
:: 여러 모델이 구현된 파일
- Autoencoder.py
- Variational_Autoencoder.py

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


---
사용방법 
- config 폴더에 model.json에서  모델 타입, 로스 및 기타 파라미터 변경해서 사용
- 배치 사이즈는 최소 5이상으로 왜냐면 학습 시 그림 그리는 걸 배치 5개 이상일때 그려짐

추가 변경 예정 사항
- 모델 학습 정적으로 사용 => 나중에 동적으로 변경할 것(how to 베이스 클래스에서 상속받아 각각의 모델 구현)
- train.py 실행 시 학습 이미지 저장 여부, 학습 이미지 show 여부, 히스토리 저장 여부, show 여부로 수정
- args로 관리할 지 고민
- 이미지 배치에 따라 동적으로 그려지는걸로 수정

---
기타
