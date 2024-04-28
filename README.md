config :: 사용자가 변경 할 수 있는 요소를 json파일로 관리
- data.json :: 데이터 관련 설정(데이터 경로, 데이터 형식)을 담은 파일
- parameter.json :: 모델 하이퍼파라미터 및 학습 관련 설정을 담은 파일

models :: 여러 모델이 구현된 파일
- Autoencoder.py
- Variational_Autoencoder.py
- DCGAN.py
- Cycle_GAN.py
- Style_GAN.py
- Diffusion.py

result :: 학습 결과물을 저장하는 폴더
- history :: 학습 과정에 대한 정보를 저장하는 폴더(loss, 정확도 등)
- model :: 학습된 모델 가중치를 저장하는 폴더

utils :: 데이터 로드 및 모델 관련 유틸리티를 포함하는 폴더
- data_utils.py
- model_utils.py

train.py :: 모델의 학습 및 검증 단계를 실행하고 결과를 저장


버전 정보
cudnn 8.6
cuda 11.3
torch 1.12.1+cu113
torchaudio 0.12.1+cu113
torchvision 0.13.1+cu113
