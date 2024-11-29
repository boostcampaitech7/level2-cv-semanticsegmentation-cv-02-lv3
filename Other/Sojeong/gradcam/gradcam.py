import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            print(f"Forward hook triggered for {self.target_layer}")
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            print(f"Backward hook triggered for {self.target_layer}")
            self.gradients = grad_out[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                print(f"Hooks registered for layer: {name}")

    def generate(self, input_image, target_class):
        self.model.eval()
        input_image = input_image.unsqueeze(0).to(next(self.model.parameters()).device)

        # Forward pass
        output = self.model(input_image)
         # 출력이 리스트인 경우 첫 번째 요소 선택 (필요에 따라 조정)
        
        if isinstance(output, list):
            output = output[0]  # 원하는 출력으로 변경

        
        if output.dim() > 1:
            class_score = output[:, target_class].mean()
        else:
            class_score = output

        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        # Check gradients
        if self.gradients is None:
            raise ValueError("Gradients were not computed. Check hooks and layer registration.")

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        grad_cam = F.relu(grad_cam).detach().cpu().numpy()

        # Normalize and resize Grad-CAM
        grad_cam = cv2.resize(grad_cam, (input_image.size(2), input_image.size(3)))
        grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))
        return grad_cam

    @staticmethod
    def overlay(image, mask, alpha=0.4, colormap=cv2.COLORMAP_JET):
        if image.dtype == np.float32 or image.max() <= 1.0:
            image = np.uint8(255 * image)

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        overlayed = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
        return overlayed
