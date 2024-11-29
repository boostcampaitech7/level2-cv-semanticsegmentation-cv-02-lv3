# YOLO11: 모델 실행 방법

이 문서는 YOLO11 모델을 설치하고 실행하는 방법을 안내합니다.

---

### **1. 필수 패키지 설치**
ultralytics 라이브러리 실행을 위한 패키지를 설치합니다.
```bash
pip install ultralytics
```

설치 후에 아래 에러가 발생한다면<br>
ImportError: libGL.so.1: cannot open shared object file: No such file or directory<br>
**libgl1** 패키지를 설치해주세요
```bash
apt-get install libgl1
```

libgl1 설치 후에도 아래 에러가 발생한다면 
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
추가로 libglib2.0-0를 설치해주세요
```bash
apt-get install libglib2.0-0

```

### **2. 데이터 준비**
- 학습 데이터는 YOLO 포맷으로 준비해야 합니다. ```<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>```
- 이후 dataset.yaml 파일에 데이터 경로 및 클래스 정보를 추가해주세요.


### **3. Train**
- train.py 스크립트를 실행하여 모델을 학습시킵니다.
```bash
python train.py --pretrained_weight "path/to/pretrained_model.pt" --data_path "path/to/dataset.yaml" --epochs 100 --batch_size 8 --img_size 640
```

- 인자 설명:
    - --pretrained_weight: 사전 학습된 YOLO 모델 가중치 경로 (필수)
    - --data_path: 학습에 사용할 데이터셋 YAML 파일 경로 (필수)
    - --epochs: 학습 에포크 수 (기본값: 100)
    - --batch_size: 배치 크기 (기본값: 8)
    - --img_size: 입력 이미지 크기 (기본값: 640)

- 학습 결과:
    - 훈련 로그는 runs/segment/train/ 디렉토리에 저장됩니다.
    - 학습된 모델 가중치는 train 폴더 안의 weights/ 디렉토리에 저장됩니다.

### **4. Inference**
- 학습된 YOLO 모델로 추론을 수행합니다.
```bash
python inference.py --pretrained_weight "path/to/trained_model.pt" --data_path "path/to/test/images" --output_csv "submission.csv"
```
- 인자 설명:
    - --pretrained_weight: 학습된 YOLO 모델 가중치 경로 (필수)
    - --data_path: 테스트 이미지가 있는 디렉토리 경로 (필수)
    - --output_csv: 추론 결과를 저장할 CSV 파일 경로 (기본값: submission.csv)

- 추론 결과:
    - submission.csv 파일에 추론 결과가 저장됩니다.
    - 결과 파일에는 image_name(이미지 이름), class(클래스), rle(RLE 인코딩)가 포함됩니다.