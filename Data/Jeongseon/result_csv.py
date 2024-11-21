import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# 색상 팔레트
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)

def create_color_palette(num_classes):
    """커스텀 색상 팔레트 생성"""
    return np.array(PALETTE[:num_classes]) / 255.0

def visualize_segmentation(image, masks, color_palette):
    """
    이미지에 세분화 마스크 시각화
    
    :param image: 원본 이미지
    :param masks: 세분화 마스크 배열
    :param color_palette: 클래스별 색상 팔레트
    :param border_only: True이면 테두리만, False이면 전체 마스크 채우기
    :return: 시각화된 이미지
    """
    vis_image = image.copy()
    
    for idx, mask in enumerate(masks):
        if mask.sum() > 0:  # 비어있지 않은 마스크만 처리
            color = color_palette[idx]
                     
            # 윤곽선 찾아 테두리 그리기
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                vis_image, 
                contours, 
                -1, 
                color * 255, #color * 255
                thickness=5
            )
  
    return vis_image

def main():
    st.title('Segmentation 시각화')
    
    # 데이터셋 경로 설정
    test_data_path = '/data/ephemeral/home/data/test/DCM'
    
    # ID 폴더 선택
    id_folders = [f for f in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, f))]
    selected_id = st.selectbox('ID 선택', id_folders)
    
    # 이미지 선택
    image_path = os.path.join(test_data_path, selected_id)
    images = [f for f in os.listdir(image_path) if f.endswith('.png')]
    selected_image = st.selectbox('이미지 선택', images)
    
    # CSV 업로드
    uploaded_csv = st.file_uploader("세분화 CSV 업로드", type=['csv'])   #accept_multiple_files=True
    
    if selected_image and uploaded_csv:
        # 이미지 읽기
        full_image_path = os.path.join(image_path, selected_image)
        image = cv2.imread(full_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # CSV 읽기
        df = pd.read_csv(uploaded_csv)
        
        # 현재 이미지에 대한 행 필터링
        image_df = df[df['image_name'] == selected_image]
        
        # RLE를 마스크로 디코딩
        masks = [
            decode_rle_to_mask(row['rle'], height, width) 
            for _, row in image_df.iterrows()
        ]
        
        # 색상 팔레트 생성
        color_palette = create_color_palette(len(masks))
        
        # 시각화
        visualization = visualize_segmentation(image, masks, color_palette)
        
        # 보여주기
        st.image(visualization)
        
        # 선택적: 클래스 이름 표시
        st.subheader('세분화된 클래스')
        st.dataframe(image_df[['class']])

if __name__ == '__main__':
    main()