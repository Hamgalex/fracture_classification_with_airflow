import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ModelEvaluator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Orden de clases ALFABÉTICO
        self.classes = ["fractured", "not_fractured"]

        # ==========================
        # ⚡ Cargar ResNet50 exactamente como fue entrenada
        # ==========================
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        # ==========================
        # ⚡ Transformaciones correctas (entrenamiento)
        # ==========================
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def evaluate(self, clean_path):
        # Abrir imagen segmentada
        img = Image.open(clean_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Predicción
        with torch.no_grad():
            outputs = self.model(tensor)
            probas = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        label = self.classes[preds.item()]
        score = float(probas[0][preds.item()].item())

        # ====================================
        # 🔍 LOGGING DETALLADO PARA DEBUG
        # ====================================
        print(f"\n{'='*60}")
        print(f"🔍 EVALUACIÓN DE MODELO - DEBUG INFO")
        print(f"{'='*60}")
        print(f"📁 Imagen evaluada: {clean_path}")
        print(f"🖼️  Tensor shape: {tensor.shape}")
        print(f"🖼️  Tensor stats: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        print(f"\n📊 OUTPUT RAW del modelo:")
        print(f"   Logits: {outputs[0].cpu().numpy()}")
        print(f"\n📊 PROBABILIDADES (después de softmax):")
        print(f"   fractured:     {probas[0][0].item():.6f} ({probas[0][0].item()*100:.2f}%)")
        print(f"   not_fractured: {probas[0][1].item():.6f} ({probas[0][1].item()*100:.2f}%)")
        print(f"\n🎯 PREDICCIÓN FINAL:")
        print(f"   Clase predicha: {label}")
        print(f"   Índice: {preds.item()}")
        print(f"   Confianza: {score:.6f} ({score*100:.2f}%)")
        print(f"{'='*60}\n")

        return label, score, tensor
