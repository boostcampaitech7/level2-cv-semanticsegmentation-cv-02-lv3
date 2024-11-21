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

def move_images(destination_path, dataset, image_root):

    os.makedirs(destination_path, exist_ok=True)

    for file_path in dataset:
        new_file_path = file_path[6:]
        source_file = os.path.join(image_root, file_path)
        destination_file = os.path.join(destination_path, new_file_path)
        shutil.copy(source_file, destination_file)
            
    print("파일 이동 완료!")

def normalize_coordinates(data, image_width, image_height):
    
    normalized_lines = []
    for line in data:
        normalized_numbers = []
        for i, num in enumerate(line):
            if i % 2 == 0:
                normalized_num = num / image_width
            else:
                normalized_num = num / image_height

            normalized_num = max(0.0, min(1.0, normalized_num))
            normalized_numbers.append(normalized_num)

        normalized_lines.append(" ".join(map(str, normalized_numbers)))
    return normalized_lines


def baseToYolo(base_dir, yolo_dir): # YOLO annotation 포맷 : <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    
    for json_path in os.listdir(base_dir):
        annotation_path = os.path.join(base_dir, json_path)
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        for annotation in annotations['annotations']:
            yolo_annotation = ""
            
            class_idx = CLASS2IND[annotation['label']]
            yolo_annotation += str(class_idx) + " "
            
            norm_points = normalize_coordinates(annotation['points'], 2048, 2048)
            
            for point in norm_points:
                yolo_annotation += " ".join(str(p) for p in point.split()) + " "
        
            file_path = f'{yolo_dir}/{json_path[:-4]}txt'
            os.makedirs(file_path, exist_ok=True)
            
            # 파일에 yolo_annotation 데이터 추가
            with open(file_path, "a") as file:
                line = yolo_annotation + "\n"
                file.write(line)
            
if __name__ == "__main__":
    
    split_path = 'your split data path'
    image_root = "your image data"
    json_root = "your json data"

    # train, valid 데이터 분리
    with open(split_path, 'r') as f:
        split_data = json.load(f)
        train_images = split_data['train_filenames']
        train_jsons = split_data['train_labelnames']
        valid_images = split_data['valid_filenames']
        valid_jsons = split_data['valid_labelnames']

    move_images("your destination_path", train_images, image_root)
    move_images("your destination_path", train_jsons, json_root)
    move_images("your destination_path", valid_images, image_root)
    move_images("your destination_path", valid_jsons, json_root)
    
    base_json_dir_train = 'your base train json dir'
    yolo_json_dir_train = 'your yolo train json dir'
    base_json_dir_valid = 'your base valid json dir'
    yolo_json_dir_valid = 'your yolo valid json dir'
    
    baseToYolo(base_json_dir_train, yolo_json_dir_train)
    baseToYolo(base_json_dir_valid, yolo_json_dir_valid)