import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1  # RLE는 1부터 시작
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

# 1. 예측 CSV 파일 불러오기
file_paths = [
    'file_path1',
    'file_path2',
    'file_path3'
]

# 2. 모든 예측 결과를 DataFrame으로 로드
predictions = [pd.read_csv(file_path) for file_path in file_paths]

# 3. 모든 RLE 디코딩
decoded_masks = []
for df in predictions:
    masks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Decoding RLE"):
        mask = rle_decode(row["rle"], (2048, 2048))  # 이미지 크기에 맞게 조정
        masks.append((row["image_name"], row["class"], mask))
    decoded_masks.append(masks)

# 4. 하드 보팅 수행 (사용자 정의 투표 기준)
threshold = 3  # 최소 몇 표를 받아야 채택할지 결정

ensemble_results = []
for i in tqdm(range(len(decoded_masks[0])), desc="Performing Custom Voting"):
    image_id = decoded_masks[0][i][0]
    class_id = decoded_masks[0][i][1]

    # 각 모델의 해당 클래스 마스크를 GPU 텐서로 변환
    masks = torch.stack([torch.tensor(decoded_masks[j][i][2], device="cuda") for j in range(len(decoded_masks))])

    # GPU에서 픽셀별 투표 계산
    votes = torch.sum(masks, dim=0)

    # 사용자 정의 기준으로 픽셀 채택 (threshold 이상의 표를 받은 픽셀만 1로 설정)
    final_mask = (votes >= threshold).int().cpu().numpy()

    # 결과를 RLE로 인코딩
    encoded_mask = rle_encode(final_mask)
    ensemble_results.append({"image_name": image_id, "class": class_id, "rle": encoded_mask})

# 5. 최종 결과 저장
ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df.to_csv("output_name", index=False)
