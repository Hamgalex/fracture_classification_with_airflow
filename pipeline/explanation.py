import os
import cv2
import numpy as np
import torch
from torchvision import models


class GradCAMGenerator:
    def __init__(self, model_path="model/modelo_fractura_resnet50.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.gradients = None
        self.activations = None

        target_layer = self.model.layer4[-1].conv3
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.HEATMAP_DIR = os.path.join(base_dir, "data", "heatmaps")
        os.makedirs(self.HEATMAP_DIR, exist_ok=True)

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_path):
        from PIL import Image

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        orig = cv2.resize(img, (224, 224))

        pil_img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)

        fractured_class_idx = 0
        target_score = output[0, fractured_class_idx]

        self.model.zero_grad()
        target_score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        filename = os.path.basename(image_path)
        heatmap_filename = f"{os.path.splitext(filename)[0]}_gradcam.png"
        heatmap_path = os.path.join(self.HEATMAP_DIR, heatmap_filename)

        cv2.imwrite(heatmap_path, superimposed)

        print(f"Grad-CAM generado: {heatmap_path}")
        print(f"CAM stats: min={cam.min():.4f}, max={cam.max():.4f}, mean={cam.mean():.4f}")

        return heatmap_path
