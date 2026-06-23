# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormerSeg(nn.Module):
    def __init__(self, num_classes=2, model_name="nvidia/mit-b2"):
        super(SegFormerSeg, self).__init__()
        # Load config only (no pretrained weights) to avoid mismatched-size warnings.
        # Fine-tuned weights are loaded externally via load_state_dict().
        config = SegformerConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits  # [B, num_classes, H, W]