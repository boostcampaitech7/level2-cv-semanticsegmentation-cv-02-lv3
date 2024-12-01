import torch
from model import *  # 모델 정의를 import
from init_weights import *
######가중치만 저장된 pt파일을 로드해서 모델에 가중치를 붙이기 위한 스크립트#######

#results_unetppp 폴더가 현재 실행 중인 스크립트의 디렉토리 안에 있다고 가정
#필요에 맞게 수정
state_dict_path = os.path.join("results_unetppp", "best_model.pt")           # 가중치 파일 경로
output_model_path = os.path.join("results_unetppp", "best_model_full.pt")    # 모델 경로 (새로 만드는 pt 파일)

# 모델 아키텍처 초기화
model = UNet_3Plus(
    in_channels=3,  # 입력 채널 (RGB 이미지)
    n_classes=29,   # 출력 클래스 수 (다중 클래스 분할)
    feature_scale=4, 
    is_deconv=True, 
    is_batchnorm=True
)

# 가중치 로드
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)

# 모델 저장
torch.save(model, output_model_path)
print(f"모델 객체가 {output_model_path}에 저장되었습니다.")
