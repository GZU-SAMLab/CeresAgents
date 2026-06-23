import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Optional
from .EfficientNet.config import cfg
from .EfficientNet.model import build_model

class DiseaseClassService:
    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        cfg.NUM_CLASSES = len(self.class_names)
        self.model = build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(int(round(cfg.IMG_SIZE / 0.875))),
            transforms.CenterCrop(cfg.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def parse_class_name(self, name: str) -> Dict[str, str]:
        parts = name.split('_', 1)
        if len(parts) == 2:
            return {"disease": name, "plant": parts[0], "condition": parts[1]}
        return {"disease": name, "plant": "unknown", "condition": "unknown"}

    def infer(self, image: Image.Image, topk: int = 5) -> Dict[str, object]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            topk = min(topk, probs.numel())
            values, indices = torch.topk(probs, k=topk)

        results: List[Dict[str, object]] = []
        for prob, idx in zip(values.tolist(), indices.tolist()):
            name = self.class_names[idx]
            info = self.parse_class_name(name)
            results.append({
                "label": idx,
                "prob": prob,
                "disease": info["disease"],
                "plant": info["plant"],
                "condition": info["condition"],
            })

        return {
            "topk": results,
            "pred": results[0] if results else None,
        }

    def infer_from_path(self, image_path: str, topk: int = 5) -> Dict[str, object]:
        image = Image.open(image_path)
        return self.infer(image, topk=topk)
