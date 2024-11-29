import os
import torch
from gradcam import GradCAM
from grad_dataset import XRayInferenceDataset
import matplotlib.pyplot as plt
import albumentations as A

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "model path.pt"

model = torch.load(model_path)
model = model.to(device)

model.eval()

# Debugging: list model layers
print("Model layers:")
for name, module in model.named_modules():
    print(name)

# Load dataset
image_root = "your image root"
tf = A.Compose([A.Resize(1024, 1024)])
dataset = XRayInferenceDataset(image_root=image_root, transforms=tf)

# Select an image
image, image_name = dataset[0]
image = image.to(device)

# Verify target layer
target_layer = "last_layer"

print(f"Using target layer: {target_layer}")

'''
target_layer = "encoder.model.incre_modules.3.0.act3" 
model.encoder.model.incre_modules[3][0].act3.inplace = False
'''


# Grad-CAM execution for all classes
num_classes = 29  # Get the number of output classes
print(f"Number of classes: {num_classes}")

# Grad-CAM visualization setup
fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))  # +1 for original image

# Display original image in the first subplot
image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
axes[0].imshow(image_np)
axes[0].axis("off")
axes[0].set_title("Original Image")

# Process each class and overlay Grad-CAM
for target_class in range(num_classes):
    print(f"Processing Grad-CAM for class {target_class}")
    grad_cam = GradCAM(model, target_layer)
    try:
        cam = grad_cam.generate(image, target_class)
    except ValueError as e:
        print(f"Grad-CAM generation error for class {target_class}: {e}")
        continue

    # Overlay Grad-CAM on the original image
    overlayed = grad_cam.overlay(image_np, cam)

    # Display the Grad-CAM result in the subplot
    axes[target_class + 1].imshow(overlayed)
    axes[target_class + 1].axis("off")
    axes[target_class + 1].set_title(f"Class {target_class}")

# Save and show the combined result
output_dir = "/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"Grad-CAM_{image_name}_{target_layer}_combined.png")
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"Grad-CAM combined result saved to {output_path}")

# python run_gradcam.py