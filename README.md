<div align='center'>
  <h2>🏆 Hand Bone Image Segmentation</h2>
</div>


<div align="center">

[👀Model](#final-model) |
[🤔Issues](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-02-lv3/issues) | 
[🚀SMP](https://github.com/qubvel-org/segmentation_models.pytorch) |
[🤗Transformers](https://huggingface.co/docs/transformers/en/index) |
[💎Ultralytics](https://github.com/ultralytics/ultralytics)
</div>

## Introduction
뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다. Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

**Goal :** 손 뼈를 정확히 분할하는 모델을 개발하여 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육 등에 사용<br>
**Data :** 2048x2048 크기의 hand bone x-ray 객체가 담긴 이미지 (Train Data 총 800장, Test Data 총 288장)<br>
**Metric :** Dice coefficient

## Project Overview
- 초기 단계에서는 EDA와 베이스라인 코드에 대한 기초적인 분석을 진행한 후, segmentation 태스크를 잘 수행하는 다양한 모델들을 탐색하고 단일 모델들의 성능을 강화시키기 위해 다양한 실험을 진행했습니다.
- 최종적으로는 성능이 잘 나오는 모델을 선정한 후 각 모델에 tta와 k-fold ensemble을 진행하였으며, 각 모델들의 추론된 output들을 hard voting으로 앙상블하여 최종 모델 아키덱쳐를 구성하였습니다.
- 결과적으로 private dice coefficient 점수 **0.9760**을 달성하여 리더보드에서 7위를 기록하였습니다.

<img width="962" alt="최종 public 리더보드 순위" src="https://github.com/user-attachments/assets/11fca078-8725-42e1-9bf3-ddc6147bc68b">

## Final Model
최종 모델은 U-Net++, HRNetv2, DeepLabv3+, U-Net3+, YOLO11, UperNet, SegFormer, BEiT 앙상블로 구성되었습니다. <br> 각 모델의 예측 결과를 바탕으로 hard voting을 적용하였고 그 결과, 최종 성능으로 **dice coefficient 0.9760**를 달성했습니다.<br>


Model | tta | 5-fold ensemble | Public score
-- | -- | -- | --
U-Net++ | o | soft-voting | 0.9734
HRNetv2 | o | soft-voting | 0.9681
DeepLabv3+ | o | soft-voting | 0.9702
U-Net3+ | x | soft-voting | 0.9574
YOLO11 | x | hard-voting | 0.9442
transformers(UperNet,SegFormer) | x | soft-voting | 0.9728
BEiT | o | soft-voting | 0.9723

## Data
```
├── data
      ├── test
            └── DCM
                  └── ID001 # 사람 고유 아이디
                         ├── 오른손 뼈 이미지 파일
                         └── 왼손 뼈 이미지 파일
     └── train
            ├── DCM
                  └── ID001
                         ├── 오른손 뼈 이미지 파일
                         └── 왼손 뼈 이미지 파일      
            └── outputs_json
                  └── ID001
                         ├── 오른손 뼈 annotation 파일
                         └── 왼손 뼈 annotation 파일   
``` 

## File Tree
```
├── .github
├── datasets
├── ensemble
├── models
         ├── HRNetv2
         ├── SMP
         ├── torchvision
         ├── transformers
         ├── ultralytics
         ├── UNet3+
└── README.md
```

## Environment Setting
<table>
  <tr>
    <th colspan="2">System Information</th> <!-- 행 병합 -->
    <th colspan="2">Tools and Libraries</th> <!-- 열 병합 -->
  </tr>
  <tr>
    <th>Category</th>
    <th>Details</th>
    <th>Category</th>
    <th>Details</th>
  </tr>
  <tr>
    <td>Operating System</td>
    <td>Linux 5.4.0</td>
    <td>Git</td>
    <td>2.25.1</td>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10.13</td>
    <td>Conda</td>
    <td>23.9.0</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>Tesla V100-SXM2-32GB</td>
    <td>Tmux</td>
    <td>3.0a</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>12.2</td>
    <td></td>
    <td></td>
  </tr>
</table>
<br>

<p align='center'>© 2024 LuckyVicky Team.</p>
<p align='center'>Supported by Naver BoostCamp AI Tech.</p>

---

<div align='center'>
  <h3>👥 Team Members of LuckyVicky</h3>
  <table width="80%">
    <tr>
      <td align="center" valign="top" width="15%"><a href="https://github.com/jinlee24"><img src="https://avatars.githubusercontent.com/u/137850412?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/stop0729"><img src="https://avatars.githubusercontent.com/u/78136790?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/yjs616"><img src="https://avatars.githubusercontent.com/u/107312651?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/sng-tory"><img src="https://avatars.githubusercontent.com/u/176906855?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/Soojeoong"><img src="https://avatars.githubusercontent.com/u/100748928?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/cyndii20"><img src="https://avatars.githubusercontent.com/u/90389093?v=4"></a></td>
    </tr>
    <tr>
      <td align="center">🍀이동진</td>
      <td align="center">🍀정지환</td>
      <td align="center">🍀유정선</td>
      <td align="center">🍀신승철</td>
      <td align="center">🍀김소정</td>
      <td align="center">🍀서정연</td>
    </tr>
    <tr>
      <td align="center">서버 관리, <br> 모델링(transformers 라이브러리)</td>
      <td align="center">기법 정리, <br> 모델링(UNet3+, DuckNet) </td>
      <td align="center">EDA, <br> 모델링(UNet3+, DuckNet) </td>
      <td align="center">WandB 관리, <br> HRNetv2, <br> 앙상블 </td>
      <td align="center">스케줄링, <br>문서화, <br>모델링(SMP 라이브러리) </td>
      <td align="center">깃 관리, <br>모델링(Ultralytics 라이브러리) </td>
    </tr>
  </table>
</div>
