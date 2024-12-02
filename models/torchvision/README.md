# FCN: 모델 실행 방법

### **1. FCN 모델 특징**

### **2. Train**
- train.py 스크립트를 실행하여 모델을 학습시킵니다.
```bash
python train.py --epochs 100 --batch_size 2 --saved_dir output_dir --image_root IMAGE_PATH --label_path LABEL_PATH --valid_batch_size 2 --val_every 10 --model_name model_name
```

- 인자 설명:
    - --epochs: 학습 에포크 수 (기본값: 100)
    - --batch_size: 배치 크기 (기본값: 2)
    - --saved_dir: 모델 저장하는 경로
    - --imgae_root: train 데이터가 저장된 경로
    - --label_root: masking 정보가 저장된 경로
    - --valid_batch_size: 검증시 배치 크기
    - --val_every: 지정된 에폭마다 검증하게 하는 크기
    - --model_name: 모델 이름 정하기


- 학습 결과:
    - 학습된 모델 가중치는 saved_dir에서 지정한 디렉토리에 저장됩니다.

### **3. Inference**
- 학습된 FCN 모델로 추론을 수행합니다.
```bash
python inference.py --image_root image_path --saved_dir output_path --model_name model_name --batch_size 2
```
- 인자 설명:
    - --image_root: test 이미지가 저장된 경로
    - --saved_dir: 추론한 결과를 저장할 위치
    - --model_name: 출력 csv의 이름 정하기(train과 맞춰줘야한다.)
    - --batch_size: 추론시 배치크기(2)

- 추론 결과:
    - saved_dir 경로에 추론 결과가 저장됩니다.
    - 결과 파일에는 image_name(이미지 이름), class(클래스), rle(RLE 인코딩)가 포함됩니다.
