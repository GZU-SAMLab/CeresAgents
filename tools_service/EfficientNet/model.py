# model.py
import warnings
import torch.nn as nn
from torchvision import models

try:
    from .config import cfg
except ImportError:
    from config import cfg


def _weights_for_model_name(model_name: str):
    """
    尽量兼容不同 torchvision 版本的 weights 传参方式。
    [修复] 兜底时返回 None 而非字符串 "IMAGENET1K_V1"，
           避免新版 torchvision 不接受字符串导致报错。
    """
    if not cfg.PRETRAINED:
        return None

    weights_enum_name = None
    if model_name.startswith("resnet"):
        suffix = model_name.replace("resnet", "")
        if suffix.isdigit():
            weights_enum_name = f"ResNet{suffix}_Weights"
    elif model_name.startswith("efficientnet_b"):
        # e.g. efficientnet_b0 -> EfficientNet_B0_Weights
        b = model_name.split("_")[-1].upper()  # B0/B1/...
        weights_enum_name = f"EfficientNet_{b}_Weights"
    elif model_name.startswith("efficientnet_v2_"):
        # e.g. efficientnet_v2_s -> EfficientNet_V2_S_Weights
        parts = model_name.split("_")
        if len(parts) == 3:
            v2_variant = parts[-1].upper()  # S/M/L
            weights_enum_name = f"EfficientNet_V2_{v2_variant}_Weights"

    weights_enum = getattr(models, weights_enum_name, None) if weights_enum_name else None
    if weights_enum is None:
        # [修复] 兜底返回 None，由 build_model 的 try/except 处理旧版兼容
        warnings.warn(
            f"未找到模型 '{model_name}' 对应的预训练权重枚举，"
            f"将尝试使用旧版 pretrained=True 参数加载。",
            UserWarning
        )
        return None

    return getattr(weights_enum, "IMAGENET1K_V1", None) or getattr(weights_enum, "DEFAULT", None)


def _infer_head_prefix(model) -> str:
    if hasattr(model, "fc"):
        return "fc"
    if hasattr(model, "classifier"):
        return "classifier"
    return ""


def _get_head_in_features(model, head_prefix: str) -> int:
    if head_prefix == "fc" and hasattr(model, "fc"):
        return model.fc.in_features
    if head_prefix == "classifier" and hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            return classifier.in_features
        if isinstance(classifier, nn.Sequential):
            for module in reversed(classifier):
                if isinstance(module, nn.Linear):
                    return module.in_features
    raise ValueError("无法推断分类头输入维度（in_features）")


def _replace_head(model, head_prefix: str, in_features: int):
    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, cfg.NUM_CLASSES),
    )

    if head_prefix == "fc":
        model.fc = new_head
        return model
    if head_prefix == "classifier":
        model.classifier = new_head
        return model

    raise ValueError(f"不支持的分类头前缀: {head_prefix}")


def build_model():
    """
    构建模型：
      - 加载 ImageNet 预训练权重
      - 可选冻结 Backbone
      - 替换分类头（含 Dropout）
    """
    model_dict = {
        # ResNet
        "resnet18":  models.resnet18,
        "resnet34":  models.resnet34,
        "resnet50":  models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        # EfficientNet
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "efficientnet_v2_m": models.efficientnet_v2_m,
        "efficientnet_v2_l": models.efficientnet_v2_l,
    }
    assert cfg.MODEL_NAME in model_dict, \
        f"不支持的模型: {cfg.MODEL_NAME}，可选: {list(model_dict.keys())}"

    model_fn = model_dict[cfg.MODEL_NAME]
    weights  = _weights_for_model_name(cfg.MODEL_NAME)

    try:
        model = model_fn(weights=weights)
    except (TypeError, ValueError):
        # 兼容旧版 torchvision：使用 pretrained 参数
        model = model_fn(pretrained=cfg.PRETRAINED)

    # 可选：冻结 Backbone，只训练分类头
    if cfg.FEATURE_EXTRACT:
        for param in model.parameters():
            param.requires_grad = False
        print("特征提取模式：Backbone 已冻结")

    # 替换分类头
    head_prefix = _infer_head_prefix(model)
    in_features = _get_head_in_features(model, head_prefix)
    model = _replace_head(model, head_prefix, in_features)

    model = model.to(cfg.DEVICE)

    # 打印参数量
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"模型        : {cfg.MODEL_NAME}")
    # print(f"总参数量    : {total / 1e6:.2f} M")
    # print(f"可训练参数  : {trainable / 1e6:.2f} M")
    # print(f"分类头类型  : {head_prefix}")
    # print(f"分类头输入  : {in_features}\n")

    return model


def build_optimizer(model):
    """
    分层学习率：
      - Backbone → LR_BACKBONE（较小）
      - 分类头   → LR_HEAD（较大）
    """
    import torch.optim as optim

    head_prefix = _infer_head_prefix(model)

    if cfg.FEATURE_EXTRACT:
        # 只优化分类头
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            params,
            lr=cfg.LR_HEAD,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        if not head_prefix:
            # [修复] 兜底时所有参数统一使用 LR_HEAD，避免分类头用小学习率训练效果差
            warnings.warn(
                "无法识别分类头前缀，所有参数将使用 LR_HEAD 统一学习率。",
                UserWarning
            )
            all_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                all_params,
                lr=cfg.LR_HEAD,
                weight_decay=cfg.WEIGHT_DECAY,
            )
        else:
            backbone_params = [p for name, p in model.named_parameters()
                               if not name.startswith(head_prefix)]
            head_params     = [p for name, p in model.named_parameters()
                               if name.startswith(head_prefix)]

            param_groups = [{"params": backbone_params, "lr": cfg.LR_BACKBONE}]
            if head_params:
                param_groups.append({"params": head_params, "lr": cfg.LR_HEAD})

            optimizer = optim.AdamW(
                param_groups,
                weight_decay=cfg.WEIGHT_DECAY,
            )

    return optimizer


def build_scheduler(optimizer):
    """余弦退火学习率调度器"""
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS,
        eta_min=cfg.ETA_MIN,
    )
