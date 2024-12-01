import torch
import json

# 1. 모델 로드
model = torch.load('/data/ephemeral/home/Jihwan/ducknet/model.pth')

# 2. 모델 구조를 문자열로 변환
model_structure = str(model)  # 모델 구조를 문자열로 변환

# 3. JSON 파일로 저장
json_path = 'model_structure.json'
with open(json_path, 'w') as f:
    json.dump({"model_structure": model_structure}, f, indent=4)

print(f"Model structure saved to {json_path}")