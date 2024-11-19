from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/data/ephemeral/home/Jungyeon/Yolo/datasets/datasets.yaml", 
                      epochs=100, 
                      batch = 4,
                      imgsz=1024)