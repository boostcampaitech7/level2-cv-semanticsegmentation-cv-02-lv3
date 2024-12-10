import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with custom arguments")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to the pretrained model file (e.g., yolo11x.pt)")
    parser.add_argument("--dataset_yaml", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.pretrained_model)

    # Train the YOLO model
    results = model.train(
        data=args.dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )

    print("Training completed successfully")