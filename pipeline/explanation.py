import os
import cv2
import numpy as np
import torch
from torchvision import models


class GradCAMGenerator:
    """
    Genera un heatmap Grad-CAM sobre imágenes limpias
    usando la última capa convolucional de ResNet50.
    """

    def __init__(self, model_path="model/modelo_fractura_resnet50.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cargar modelo EXACTO como en evaluación
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        # Gradients y activaciones serán guardadas aquí
        self.gradients = None
        self.activations = None

        # Ultima capa convolucional de ResNet50
        target_layer = self.model.layer4[-1].conv3
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

        # Carpeta donde se guardarán heatmaps
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.HEATMAP_DIR = os.path.join(base_dir, "data", "heatmaps")
        os.makedirs(self.HEATMAP_DIR, exist_ok=True)

        # Transformación para GradCAM (igual que evaluación)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # =====================================================
    # HOOKS
    # =====================================================

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # =====================================================
    # 🔥 MÉTODO NECESARIO PARA EL DAG
    # =====================================================
    def generate(self, image_path):
        """
        Genera un heatmap GradCAM y devuelve la ruta del PNG guardado.
        """
        from PIL import Image

        # 1) Cargar imagen limpia
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        orig = cv2.resize(img, (224, 224))

        # 2) Transformar a tensor
        pil_img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # 3) Forward pass
        output = self.model(input_tensor)

        # 4) Obtener la clase "fractured" (índice 0 según orden alfabético)
        fractured_class_idx = 0
        target_score = output[0, fractured_class_idx]

        # 5) Backward pass para obtener gradientes
        self.model.zero_grad()
        target_score.backward()

        # 6) Calcular Grad-CAM
        # Gradientes de la última capa convolucional
        gradients = self.gradients.cpu().data.numpy()[0]  # [C, H, W]
        # Activaciones de la última capa convolucional
        activations = self.activations.cpu().data.numpy()[0]  # [C, H, W]

        # Pesos (promedio de gradientes por canal)
        weights = np.mean(gradients, axis=(1, 2))  # [C]

        # Combinar pesos con activaciones
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 7) Aplicar ReLU y normalizar
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # 8) Resize a tamaño original
        cam = cv2.resize(cam, (224, 224))

        # 9) Convertir a heatmap con colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # 10) Superponer heatmap en imagen original
        superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        # 11) Guardar resultado
        filename = os.path.basename(image_path)
        heatmap_filename = f"{os.path.splitext(filename)[0]}_gradcam.png"
        heatmap_path = os.path.join(self.HEATMAP_DIR, heatmap_filename)

        cv2.imwrite(heatmap_path, superimposed)

        print(f"✅ Grad-CAM generado: {heatmap_path}")
        print(f"🔍 CAM stats: min={cam.min():.4f}, max={cam.max():.4f}, mean={cam.mean():.4f}")

        return heatmap_path
