from ultralytics import YOLO

# Load a model
model = YOLO("your pretrained model")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="your dataset yaml file", epochs=100, batch = 4, imgsz=2048)