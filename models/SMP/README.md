# Segmentation Models: U-Net++ and DeepLabv3+ with Segmentation Models PyTorch (SMP)

Our project employs the [Segmentation Models PyTorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch) library to implement **U-Net++** and **DeepLabv3+**, two advanced architectures for image segmentation tasks.

## 1. Why U-Net++ and DeepLabv3+?

### U-Net++
- **Detailed Feature Representation**: Enhanced feature representation with dense skip connections, enabling the model to capture small and intricate structures.
- **Improved Learning of Small Features**: The nested architecture allows for better learning of local details and fine-grained segmentation.

### DeepLabv3+
- **Multi-Scale Feature Capture**: Utilizes Atrous Spatial Pyramid Pooling (ASPP) to effectively aggregate multi-scale contextual information.
- **Robust on Diverse Scales**: Excels at segmenting objects of varying sizes, maintaining consistent performance across complex images.

## 2. About the SMP Library

[SMP](https://github.com/qubvel-org/segmentation_models.pytorch) is a versatile library providing state-of-the-art segmentation architectures and pretrained backbones. It simplifies implementation, experimentation, and deployment of segmentation models with:
- Multiple segmentation architectures (e.g., U-Net++, DeepLabv3+, etc.).
- Pretrained encoders for efficient feature extraction.
- An intuitive and well-documented API.

## 3. Implementation Details

- **Architectures**:
  - **U-Net++**: Nested U-Net structure with dense skip connections.
  - **DeepLabv3+**: Atrous convolution-based architecture for robust multi-scale segmentation.
- **Backbones**: (Specify backbones, e.g., ResNet34, EfficientNet-B0, or others used for each model).
- **Pretrained Weights**: Both models use ImageNet pretrained weights for backbones to enhance performance.
- **Loss Function**: (Specify, e.g., Dice Loss, Cross-Entropy Loss, or a combination).
- **Optimizer**: (Specify, e.g., Adam, SGD).

## 4. User Guide

Follow the steps below to install and train the segmentation models.

### Installation

To install the SMP library, run the following command:

```bash
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

### Training and Inference: U-Net++

```
cd ../level2-cv-semanticsegmentation-cv-02-lv3/models/SMP/U-Net++
# Ensure the split_file path is correctly set in your configuration:

# Grant execution permission to the training script
chmod +x ./unetplusplus_train.sh 
# Start the training
./unetplusplus_train.sh

# Grant execution permission to the inference script
chmod +x ./unetplusplus_inference.sh
# Start the inference
./unetplusplus_inference.sh
```

### Training: DeepLabv3+

To train the **DeepLabv3+** model, you can utilize the `deeplabv3plus_train.sh` script. Ensure the configuration and paths match your dataset settings.

Steps:
1. Navigate to the same directory where U-Net++ scripts are located.
2. Grant execution permission to the training script:
   ```bash
   chmod +x ./deeplabv3plus_train.sh
   ```
3. Start the training:
    ```bash
   ./deeplabv3plus_train.sh
   ```
