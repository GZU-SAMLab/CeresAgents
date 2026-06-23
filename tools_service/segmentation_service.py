import os
import numpy as np
from .segformer_para.model import SegFormerSeg
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from typing import Dict, Optional

def severity_level(r: float) -> int:
    if r < 0.01:
        return 0
    elif r < 0.05:
        return 1
    elif r < 0.15:
        return 2
    elif r < 0.30:
        return 3
    else:
        return 4

class SegmentationSeverityService:
    def __init__(
        self,
        leaf_model_path: str,
        lesion_model_path: str,
        image_size: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.leaf_model = SegFormerSeg(num_classes=2, model_name='nvidia/mit-b2')
        leaf_state = torch.load(leaf_model_path, map_location=self.device)
        self.leaf_model.load_state_dict(leaf_state)
        self.leaf_model.to(self.device)
        self.leaf_model.eval()

        self.lesion_model = SegFormerSeg(num_classes=2, model_name='nvidia/mit-b2')
        lesion_state = torch.load(lesion_model_path, map_location=self.device)
        self.lesion_model.load_state_dict(lesion_state)
        self.lesion_model.to(self.device)
        self.lesion_model.eval()

    def _predict_masks(self, image: Image.Image) -> Dict[str, np.ndarray]:
        img_np = np.array(image.convert("RGB"))
        tensor = self.transform(image=img_np)['image'].unsqueeze(0).to(self.device)

        with torch.inference_mode():
            leaf_logits = self.leaf_model(tensor)
            leaf_pred = leaf_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            lesion_logits = self.lesion_model(tensor)
            lesion_pred = lesion_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        return {
            "leaf": leaf_pred,
            "lesion": lesion_pred,
        }

    @staticmethod
    def render_overlay_on_image(
        base_image: Image.Image,
        leaf: np.ndarray,
        lesion: np.ndarray,
        leaf_gray: int = 140,
        lesion_color: tuple = (255, 255, 0),
        lesion_alpha: float = 0.6,
        leaf_alpha: float = 0.5,
        background_dim: float = 0.3,
    ) -> Image.Image:
        base_image = base_image.resize((leaf.shape[1], leaf.shape[0]), Image.BILINEAR)
        base_np = np.asarray(base_image.convert("RGB"), dtype=np.float32)
        
        leaf_bool = leaf > 0.5
        lesion_bool = lesion > 0.5

        brightness = np.full(leaf_bool.shape, background_dim, dtype=np.float32)
        if leaf_bool.any():
            brightness[leaf_bool] = 1.0

        brightness = brightness[:, :, np.newaxis]
        processed_base = np.clip(base_np * brightness, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(processed_base, mode="RGB").convert("RGBA")

        if lesion_bool.any():
            h, w = leaf_bool.shape
            overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
            overlay_np[lesion_bool, 0] = lesion_color[0]
            overlay_np[lesion_bool, 1] = lesion_color[1]
            overlay_np[lesion_bool, 2] = lesion_color[2]
            overlay_np[lesion_bool, 3] = int(255 * lesion_alpha)

            overlay_img = Image.fromarray(overlay_np, mode="RGBA")
            result_img = Image.alpha_composite(result_img, overlay_img)

        return result_img

    def infer(self, image: Image.Image) -> Dict[str, object]:
        masks = self._predict_masks(image)
        leaf_pred = masks["leaf"]
        lesion_pred = masks["lesion"]

        leaf_area = leaf_pred.sum()
        overlap_area = (leaf_pred & lesion_pred).sum()

        ratio = 0.0 if leaf_area == 0 else float(overlap_area) / float(leaf_area)
        level = severity_level(ratio)

        leaf_mask_img = Image.fromarray((leaf_pred * 255).astype(np.uint8), mode="L")
        lesion_mask_img = Image.fromarray((lesion_pred * 255).astype(np.uint8), mode="L")
        
        overlay_img = self.render_overlay_on_image(image, leaf_pred, lesion_pred)
        
        return {
            "ratio": ratio,
            "level": level,
            "leaf_area": int(leaf_area),
            "lesion_area": int(lesion_pred.sum()),
            "leaf_mask": leaf_mask_img,
            "lesion_mask": lesion_mask_img,
            "overlay": overlay_img,
        }

    def infer_from_path(self, image_path: str) -> Dict[str, object]:
        image = Image.open(image_path)
        return self.infer(image)

    @staticmethod
    def save_masks(
        leaf_mask: Image.Image,
        lesion_mask: Image.Image,
        overlay: Optional[Image.Image],
        output_dir: str,
        base_name: str,
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        leaf_path = os.path.join(output_dir, f"{base_name}_leaf.png")
        lesion_path = os.path.join(output_dir, f"{base_name}_lesion.png")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")

        leaf_mask.save(leaf_path)
        lesion_mask.save(lesion_path)
        if overlay is not None:
            overlay.save(overlay_path)

        return {
            "leaf": leaf_path,
            "lesion": lesion_path,
            "overlay": overlay_path if overlay is not None else "",
        }
