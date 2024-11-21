import streamlit as st
import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import albumentations as A
import cv2
import json

# Streamlit 화면 설정
st.set_page_config(layout="wide")

# 현재 스크립트 디렉토리
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로 설정
IMAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/train/DCM"))
ANNOTATION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/train/outputs_json"))

# 디버깅 출력
#print("SCRIPT_DIR:", SCRIPT_DIR)
#print("IMAGE_ROOT:", IMAGE_ROOT)
#print("ANNOTATION_ROOT:", ANNOTATION_ROOT)


###################json & 이미지 로드 ##############################3###3
# JSON 로드 함수
def load_annotations(annotation_root, image_id, image_name):
    """JSON 파일에서 이미지 이름에 해당하는 세그멘테이션 데이터를 로드"""
    annotation_path = os.path.join(annotation_root, image_id, f"{image_name}.json")
    if not os.path.exists(annotation_path):
        st.error(f"JSON 파일이 존재하지 않습니다: {annotation_path}")
        return []

    # JSON 파일 로드
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        #st.write(f"JSON 파일 로드 성공: {annotation_path}")
        #st.write("JSON 데이터 샘플:", data)
        return data.get("annotations", [])  # annotations 필드 반환
    except json.JSONDecodeError as e:
        st.error(f"JSON 파일을 파싱할 수 없습니다: {annotation_path}, 에러: {str(e)}")
        return []

# 이미지 로딩 함수 (EXIF 방향 정보 자동 적용)
def load_image_with_orientation(image_path):
    """Load image and apply EXIF orientation if available"""
    image = ImageOps.exif_transpose(Image.open(image_path))
    return image.convert("RGB")


######################**증 강 함 수**#############################################
def get_crop_parameters(original_image):
    apply_crop = st.sidebar.checkbox("Center Crop 적용")
    if apply_crop:
        maintain_aspect_ratio = st.sidebar.checkbox("원본 비율 유지", value=True)
        
        if maintain_aspect_ratio:
            # 원본 비율을 유지하는 경우
            original_width, original_height = original_image.size
            aspect_ratio = original_width / original_height
            
            # 더 작은 차원을 기준으로 crop 크기 설정
            base_size = min(original_width, original_height)
            crop_size = st.sidebar.slider(
                "Crop Size (%)", 
                min_value=10, 
                max_value=100, 
                step=5, 
                value=50
            )
            
            # 실제 크기 계산
            crop_height = int(base_size * (crop_size / 100))
            crop_width = int(crop_height * aspect_ratio)
            
            # 큰 쪽이 원본을 넘지 않도록 조정
            if crop_width > original_width:
                crop_width = original_width
                crop_height = int(crop_width / aspect_ratio)
                
        else:
            # 정사각형 crop
            crop_size = st.sidebar.slider(
                "Crop Size", 
                min_value=10, 
                max_value=min(original_image.size), 
                step=10, 
                value=min(original_image.size) // 2
            )
            crop_width = crop_height = crop_size
    else:
        crop_width = crop_height = 0
        
    return crop_width, crop_height


def create_augmentation_transforms(h_flip, v_flip, brightness, blur, crop_width, crop_height,
                                   hue_shift, sat_shift, val_shift, grid_num_steps, grid_distort_limit,
                                   alpha, sigma, scale, rotate_limit, dilation_kernel_size, erosion_kernel_size):
    """증강 옵션 리스트 생성"""
    transforms = []
    
    # CenterCrop을 첫 번째 transform으로 이동
    if crop_width > 0 and crop_height > 0:
        transforms.append(A.CenterCrop(height=crop_height, width=crop_width, p=1.0))
        
    if h_flip:
        transforms.append(A.HorizontalFlip(p=1.0))
    if v_flip:
        transforms.append(A.VerticalFlip(p=1.0))
    if brightness > 0:
        transforms.append(A.CLAHE(clip_limit=max(brightness, 1), p=1.0))
    if blur > 0:
        transforms.append(A.GaussianBlur(blur_limit=(blur, blur), p=1.0))
    if hue_shift != 0 or sat_shift != 0 or val_shift != 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=(hue_shift, hue_shift),
            sat_shift_limit=(sat_shift, sat_shift),
            val_shift_limit=(val_shift, val_shift),
            p=1.0
        ))
    if grid_num_steps > 0 and grid_distort_limit > 0:
        transforms.append(A.GridDistortion(num_steps=grid_num_steps, distort_limit=grid_distort_limit, p=1.0))
    if alpha > 0 and sigma > 0:
        transforms.append(A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha * 0.1, p=1.0))
    if scale > 0:
        transforms.append(A.Perspective(scale=(scale, scale), p=1.0))
    if rotate_limit != 0:
        transforms.append(A.Rotate(limit=rotate_limit, p=1.0))
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        transforms.append(A.Lambda(image=lambda img, **kwargs: cv2.dilate(img, kernel), p=1.0))
    if erosion_kernel_size > 0:
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        transforms.append(A.Lambda(image=lambda img, **kwargs: cv2.erode(img, kernel), p=1.0))
    
    return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def augment_image_and_segmentation(image, segmentation_coords, transform):
    """이미지와 세그멘테이션 데이터를 함께 증강"""
    if not transform:
        return image, segmentation_coords

    # 원본 이미지 크기 저장
    original_height, original_width = image.shape[:2]
    
    # Segmentation Keypoints 변환
    keypoints = []
    original_lengths = []  # 각 세그먼트의 원본 길이 저장
    
    for points in segmentation_coords:
        original_lengths.append(len(points))
        keypoints.extend(points)

    try:
        # Albumentations 변환 적용
        augmented = transform(image=image, keypoints=keypoints)
        augmented_image = augmented['image']
        augmented_keypoints = augmented['keypoints']

        # 크롭된 이미지 크기
        new_height, new_width = augmented_image.shape[:2]
        
        # Center crop 오프셋 계산
        offset_x = (original_width - new_width) // 2
        offset_y = (original_height - new_height) // 2

        # Keypoints를 원래 세그멘테이션 형식으로 변환하고 유효한 좌표만 유지
        augmented_segmentation = []
        start_idx = 0
        
        for length in original_lengths:
            segment_points = augmented_keypoints[start_idx:start_idx + length]
            # 유효한 좌표만 필터링 (이미지 경계 내에 있는 점들)
            valid_points = []
            for point in segment_points:
                x, y = point
                if 0 <= x <= new_width and 0 <= y <= new_height:
                    valid_points.append(point)
            
            # 최소 3개의 점이 있는 세그먼트만 추가
            if len(valid_points) >= 3:
                augmented_segmentation.append(valid_points)
            
            start_idx += length

        return augmented_image, augmented_segmentation

    except ValueError as e:
        print(f"Augmentation failed: {str(e)}")
        return image, segmentation_coords

# 세그멘테이션 시각화 함수
def draw_segmentation(image, segmentation_coords, line_width=5):
    """이미지 위에 세그멘테이션 폴리곤 그리기"""
    draw = ImageDraw.Draw(image)

    # 고정된 색상 팔레트 (최대 29개)
    COLORS = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", 
        "#800000", "#808000", "#008000", "#800080", "#008080", "#000080",
        "#FF4500", "#32CD32", "#1E90FF", "#FFD700", "#EE82EE", "#20B2AA", 
        "#A52A2A", "#B8860B", "#556B2F", "#6A5ACD", "#7B68EE", "#48D1CC",
        "#FF6347", "#40E0D0", "#FF69B4", "#DC143C", "#00CED1"
    ]

    for i, points in enumerate(segmentation_coords):
        points = [(float(x), float(y)) for x, y in points]
        if len(points) > 2:  # 최소 3개의 점이 있어야 폴리곤 생성 가능
            color = COLORS[i % len(COLORS)]
            for j in range(len(points)):
                start = points[j]
                end = points[(j + 1) % len(points)]  # 폐곡선을 위해 마지막 점 연결
                draw.line([start, end], fill=color, width=line_width)
    return image



#####################streamlit UI######################################
# 메인 UI 부분
def main(): 
    ####################### 메인 UI############################################33
    st.title("Data Augmentation Editor")

    # ID 선택 (DCM 하위 디렉토리 탐색)
    available_ids = [d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d))]
    selected_id = st.selectbox("ID 선택", options=available_ids)

    # ID 경로 설정
    id_folder = os.path.join(IMAGE_ROOT, selected_id)                 #img
    id_annotation_folder = os.path.join(ANNOTATION_ROOT, selected_id) #json
    image_files = [f for f in os.listdir(id_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 이미지 선택
    selected_image_file = st.selectbox("이미지 선택", options=image_files)
    selected_image_path = os.path.join(id_folder, selected_image_file)

    # JSON 파일 로드
    image_name = selected_image_file.split('.')[0]  # 확장자 제거
    annotations = load_annotations(ANNOTATION_ROOT, selected_id, image_name)

    # 원본 이미지 로드
    original_image = load_image_with_orientation(selected_image_path)
    original_image_np = np.array(original_image)

    # Segmentation 좌표 가져오기
    segmentation_coords = [annotation.get("points", []) for annotation in annotations]


    ###########################사이드 바#################################################

    # 증강 파라미터 설정
    st.sidebar.title("Data Augmentation Parameters")

    # Segmentation 표시 여부 설정
    show_segmentation = st.sidebar.checkbox("Segmentation 표시하기", value=True)

    # Crop 파라미터 가져오기 (새로운 함수 사용)
    crop_width, crop_height = get_crop_parameters(original_image)

    h_flip = st.sidebar.checkbox("Horizontal Flip")
    v_flip = st.sidebar.checkbox("Vertical Flip")
    brightness = st.sidebar.slider("Brightness (Contrast Adjustment)", min_value=1, max_value=10, step=1, value=1)
    blur = st.sidebar.slider("Blur Level", min_value=1, max_value=31, step=2, value=1)

    #apply_crop = st.sidebar.checkbox("Center Crop 적용")
    #crop = st.sidebar.slider("Crop Size", min_value=50, max_value=min(original_image.size), step=10, value=100) if apply_crop else 0
    # crop 크기를 이미지 크기에 맞게 제한
    #apply_crop = st.sidebar.checkbox("Center Crop 적용")
    #crop = st.sidebar.slider(
    #    "Crop Size", 
    #    min_value=10, 
    #    max_value=min(original_image.size), 
    #    step=10, 
    #    value=min(original_image.size) // 2
    #) if apply_crop else 0

    hue_shift = st.sidebar.slider("색조 변화", min_value=-30, max_value=30, step=1, value=0)
    sat_shift = st.sidebar.slider("채도 변화", min_value=-50, max_value=50, step=1, value=0)
    val_shift = st.sidebar.slider("밝기 변화", min_value=-50, max_value=50, step=1, value=0)

    # Grid Distortion Parameters
    apply_grid_distortion = st.sidebar.checkbox("Grid Distortion 적용")
    grid_num_steps = st.sidebar.slider("Grid Distortion Steps", min_value=2, max_value=10, step=1, value=5) if apply_grid_distortion else 0
    grid_distort_limit = st.sidebar.slider("Grid Distortion Limit", min_value=0.0, max_value=0.5, step=0.05, value=0.3) if apply_grid_distortion else 0

    # Additional Augmentation Parameters
    apply_elastic = st.sidebar.checkbox("Elastic Transformation 적용")
    alpha = st.sidebar.slider("Elastic Alpha", min_value=1, max_value=1000, step=10, value=50) if apply_elastic else 0
    sigma = st.sidebar.slider("Elastic Sigma", min_value=1, max_value=100, step=1, value=10) if apply_elastic else 0

    apply_perspective = st.sidebar.checkbox("Perspective Transformation 적용")
    scale = st.sidebar.slider("Perspective Scale", min_value=0.1, max_value=0.5, step=0.05, value=0.2) if apply_perspective else 0

    apply_rotation = st.sidebar.checkbox("Rotation 적용")
    rotate_limit = st.sidebar.slider("Rotation Angle", min_value=-90, max_value=90, step=1, value=0) if apply_rotation else 0

    apply_dilation = st.sidebar.checkbox("Dilation 적용")
    # Custom dilation 적용
    dilation_kernel_size = st.sidebar.slider("Dilation Kernel Size", min_value=1, max_value=31, step=2, value=3) if apply_dilation else 0

    apply_erosion = st.sidebar.checkbox("Erosion 적용")
    erosion_kernel_size = st.sidebar.slider("Erosion Kernel Size", min_value=1, max_value=31, step=2, value=3) if apply_erosion else 0


    # 증강 적용
    transform = create_augmentation_transforms(
        h_flip=h_flip,
        v_flip=v_flip,
        brightness=brightness,
        blur=blur,
        crop_width=crop_width,
        crop_height=crop_height,
        hue_shift=hue_shift,
        sat_shift=sat_shift,
        val_shift=val_shift,
        grid_num_steps=grid_num_steps,
        grid_distort_limit=grid_distort_limit,
        alpha=alpha,
        sigma=sigma,
        scale=scale,
        rotate_limit=rotate_limit,
        dilation_kernel_size=dilation_kernel_size,
        erosion_kernel_size=erosion_kernel_size
    )

    # 이미지와 세그멘테이션 증강
    augmented_image, augmented_segmentation = augment_image_and_segmentation(original_image_np, segmentation_coords, transform)

    # 증강된 이미지 Pillow로 변환
    augmented_image = Image.fromarray(augmented_image)

    # 원본 이미지와 증강된 이미지 나란히 표시
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("원본 이미지")
        if show_segmentation:
            original_with_segmentation = draw_segmentation(original_image.copy(), segmentation_coords)
            st.image(original_with_segmentation)
        else:
            st.image(original_image)

    with col2:
        st.subheader("증강 이미지")
        if show_segmentation:
            augmented_with_segmentation = draw_segmentation(augmented_image.copy(), augmented_segmentation)
            st.image(augmented_with_segmentation)
        else:
            st.image(augmented_image)

if __name__ == "__main__":
    main()