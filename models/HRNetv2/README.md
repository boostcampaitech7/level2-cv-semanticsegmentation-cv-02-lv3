# HRNetv2(Semantic-Segmentation)

### **1. 필수 base 코드 및 pretrained weight 받아오기**

HRNetv2를 실행하기 위해서 HRNet-Semantic-Segmentation github에서 base code와 pretrained된 가중치를 다운받아 와야한다.
[text](https://github.com/HRNet/HRNet-Semantic-Segmentation)

### **2. HRNetv2만의 특징**

아래의 특징으로 높은 해상도에서 semantic segmentation을 할 수 있는 모델이다.

- Stage-by-Stage 구조로 각 Stage는 서로 다른 해상도의 Feature Map을 생성하고 이를 융합한다.
- 다양한 해상도의 Feature Map을 유지하기 위해 Parallel Streams을 사용합니다.
- High resolution을 유지하며 서로 다른 feature map이 정보를 공유하는 형태이다.


### **3. 사용 가이드**

- 모델 stage마다의 채널과 출력 class 수를 정하고 pretrained weight 파일 경로를 지정하는 config 파일을 만든다.

- train.py 스크립트를 실행하여 모델을 학습시킵니다.
```bash
python train.py --epochs 100 --batch_size 2 --valid_batch_size 2 --val_every 10 --model_name hrnet2 --image_resize 1024 --saved_dir output_path --image_root IMAGE_PATH --lable_root LABEL_PATH
```

인자 설명:
    - --image_root: train 이미지가 저장된 경로
    - --label_root: masking 정보가 들어있는 경로
    - --epochs: 학습 에포크 수 (기본값: 100)
    - --batch_size: 배치 크기 (기본값: 2)
    - --image_resize: 입력 이미지 크기 (기본값: 1024)
    - --valid_batch_size: 검증 시 배치크기 (기본값 :2)
    - --model_name: 학습 시 wandb에 기록될 이름
    - --saved_dir: 학습된 가중치가 저장되는 곳

- 학습 결과:
    - 학습된 모델 가중치는 우리가 정한 output_path에 저장된다.

- 학습된 HRNetv2 모델로 추론을 수행합니다.
```bash
python inference.py --image_root IMAGE_PATH --model_name hrnet2 --saved_dir output_path --image_resize 1024 --batch_size 2
```
- 인자 설명:
    - --image_root: 테스트 이미지가 저장된 경로
    - --model_name: csv 저장할때 쓸 이름
    - --saved_dir: 추론 결과를 저장할 CSV 파일 경로
    - --batch_size: 배치 크기 (기본값: 2)
    - --image_resize: 입력 이미지 크기 (기본값: 1024)

- 추론 결과:
    - 추론 결과는 output_path에 저장됩니다.
    - 결과 파일에는 image_name(이미지 이름), class(클래스), rle(RLE 인코딩)가 포함됩니다.
