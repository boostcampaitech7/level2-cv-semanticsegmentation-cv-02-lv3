import json
import os
import shutil
import json

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

def load_split_data(split_path, images_path, labels_path, image_root, label_root):
    with open(split_path, 'r') as f:
        split_data = json.load(f)
        
    data = []
    for split_type in ['train', 'valid']:
        # 파일 목록 가져오기
        image_files = split_data[f'{split_type}_filenames']
        label_files = split_data[f'{split_type}_labelnames']

        data.append((
            os.path.join(images_path, split_type), image_files, image_root
        ))
        data.append((
            os.path.join(labels_path, split_type), label_files, label_root
        ))

    return data

def move_images(destination_path, dataset, root):
    os.makedirs(destination_path, exist_ok=True)

    for file_path in dataset:
        new_file_path = file_path[6:]
        source_file = os.path.join(root, file_path)
        destination_file = os.path.join(destination_path, new_file_path)
        shutil.copy(source_file, destination_file)
            
    print("파일 이동 완료!")

def normalize_coordinates(data, image_width, image_height):
    normalized_lines = []
    
    width_factor = 1.0 / image_width
    height_factor = 1.0 / image_height

    for line in data:
        normalized_numbers = [
            max(0.0, min(1.0, coord * width_factor if i % 2 == 0 else coord * height_factor))
            for i, coord in enumerate(line)
        ]

        normalized_lines.append(" ".join(map(str, normalized_numbers)))
        
    return normalized_lines

def base_to_yolo(base_dir, yolo_dir):
    os.makedirs(yolo_dir, exist_ok=True)

    for json_file in os.listdir(base_dir):
        annotation_path = os.path.join(base_dir, json_file)
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        for annotation in annotations['annotations']:
            class_idx = CLASS2IND[annotation['label']]
            norm_points = normalize_coordinates(annotation['points'], 2048, 2048)

            yolo_annotation = f"{class_idx} " + " ".join(
                point for norm_point in norm_points for point in norm_point.split()
            )

            yolo_file_path = os.path.join(yolo_dir, json_file.replace('.json', '.txt'))

            with open(yolo_file_path, "a") as file:
                file.write(yolo_annotation + "\n")
            
if __name__ == "__main__":
    
    split_path = 'your split data'
    image_root = "your image data"
    label_root = "your json data"

    destination_image_path = 'your destination path for split images'
    destination_label_path = 'your destination path for split labels'
    
    # 파일 이동
    split_data = load_split_data(split_path, destination_image_path, destination_label_path, image_root, label_root)
    for destination_path, dataset, root in split_data:
        move_images(destination_path, dataset, root)

    yolo_train_path = 'your yolo train label path'
    yolo_valid_path = 'yout yolo valid label path'
    
    # 기존 json 파일을 YOLO 형식으로 변환
    for base, yolo in [(f'{destination_label_path}/train', yolo_train_path), 
                       (f'{destination_label_path}/valid', yolo_valid_path)]:
        base_to_yolo(base, yolo)