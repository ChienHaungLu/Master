# ---- workaround: stub torchvision::nms before importing torchvision ----
try:
    import torch
    from torch.library import Library
    _tv_stub = Library("torchvision", "DEF")
    _tv_stub.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass
# -----------------------------------------------------------------------

# 忽略 torchvision 圖像擴充警告（不影響執行）
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    category=UserWarning,
    module="torchvision.io.image",
)

from contextlib import nullcontext, contextmanager
import os, time, random, argparse, csv
from glob import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

# 繪圖（Headless）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator   # 讓 x 軸顯示整數

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50

# -----------------------------
# AMP 自動混合精度（以工廠函數 amp_ctx() 回傳 context）
# -----------------------------
use_cuda = torch.cuda.is_available()
try:
    from torch import amp
    autocast = amp.autocast
    GradScaler = amp.GradScaler
    def amp_ctx():
        return autocast(device_type='cuda') if use_cuda else nullcontext()
    def make_scaler():
        return GradScaler(enabled=use_cuda)
except Exception:
    # 舊版相容
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    autocast = _autocast
    GradScaler = _GradScaler
    def amp_ctx():
        return autocast() if use_cuda else nullcontext()
    def make_scaler():
        return GradScaler(enabled=use_cuda)

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    # 固定隨機性，啟用 cudnn 最佳化
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: str):
    # 目錄不存在就建立
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def list_images(root: str, exts=('.png', '.jpg', '.jpeg')) -> List[str]:
    # 遞迴列出影像檔
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(root, f'**/*{ext}'), recursive=True))
    return sorted(files)

def colorize_mask(mask: np.ndarray) -> Image.Image:
    # 將灰階 mask 依類別著色（僅視覺化用）
    palette = [
        (0,0,0),(128,0,0),(0,128,0),(128,128,0),
        (0,0,128),(128,0,128),(0,128,128),(128,128,128),
        (64,0,0),(192,0,0),(64,128,0),(192,128,0),
        (64,0,128),(192,0,128),(64,128,128),(192,128,128),
    ]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(palette):
        rgb[mask == i] = col
    return Image.fromarray(rgb)

# -----------------------------
# EMA（指數移動平均）權重（FP32 影子、裝置/型別自動對齊）
# -----------------------------
class ModelEMA:
    """
    維護模型浮點參數的滑動平均；可放 CPU 省顯存
    - 影子權重統一以 FP32 儲存（AMP 更穩）
    - update() 會把當前權重搬到影子權重的裝置與 dtype 再做 lerp
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = 'cpu'):
        self.decay = float(decay)
        self.device = device
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v):
                    data = v.detach().clone()
                    if data.dtype != torch.float32:
                        data = data.to(dtype=torch.float32)
                    self.shadow[k] = data.to(device if device else v.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        # 影子權重 ← lerp(當前權重)
        d = 1.0 - self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and torch.is_floating_point(v):
                s = self.shadow[k]
                v_det = v.detach().to(device=s.device, dtype=s.dtype)
                s.lerp_(v_det, d)

    @torch.no_grad()
    def store(self, model: nn.Module):
        # 暫存原權重
        self.backup = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.backup[k] = v.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        # 將 EMA 權重覆蓋到模型（對齊 device + dtype）
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and torch.is_floating_point(sd[k]):
                sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        # 還原原權重
        if not self.backup:
            return
        sd = model.state_dict()
        for k, v in self.backup.items():
            if k in sd and torch.is_floating_point(sd[k]):
                sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
        self.backup = {}

    def state_dict(self):
        # 儲存 EMA 內部狀態
        return {
            'decay': self.decay,
            'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state):
        # 載入 EMA 狀態
        self.decay = float(state['decay'])
        self.shadow = {k: v.clone() for k, v in state['shadow'].items()}
        self.backup = {}

    def full_state_dict_like(self, model: nn.Module):
        """
        以當前 model.state_dict() 為底，將浮點張量換成 EMA 張量
        """
        base = model.state_dict()
        out = {}
        for k, v in base.items():
            if torch.is_floating_point(v) and (k in self.shadow):
                out[k] = self.shadow[k].to(v.device, dtype=v.dtype)
            else:
                out[k] = v
        return out

@contextmanager
def use_ema_weights(ema: ModelEMA, model: nn.Module):
    # 以 EMA 權重做推論／驗證的便捷上下文
    ema.store(model)
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model)

# -----------------------------
# Dataset（不改原圖尺寸；影像/標註同步增強）
# -----------------------------
class UAVSegDataset(Dataset):
    def __init__(self,
                 img_roots: List[str],
                 mask_roots: Optional[List[str]] = None,
                 image_size: int = 512,   # 參數保留但不使用
                 is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.samples = []  # (img_path, mask_path or None)

        # 收集影像路徑
        img_files = []
        for r in img_roots:
            img_files.extend(list_images(r))
        img_files = sorted(img_files)

        # 建立檔名→mask 對應
        mask_map: Dict[str, str] = {}
        if mask_roots is not None:
            for r in mask_roots:
                for p in list_images(r):
                    key = os.path.basename(p)
                    mask_map[key] = p

        # 配對 (img, mask)
        for ip in img_files:
            fname = os.path.basename(ip)
            mp = mask_map.get(fname, None) if mask_roots is not None else None
            self.samples.append((ip, mp))

        # 基本轉換
        self.color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
        self.to_tensor   = transforms.ToTensor()
        self.normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img = Image.open(ip).convert('RGB')

        if mp is None:
            # Test：保留原圖尺寸與檔名
            orig_w, orig_h = img.size
            img = self.to_tensor(img)
            img = self.normalize(img)
            return img, os.path.basename(ip), (orig_w, orig_h), ip

        # Train/Val：不 resize，確保與 mask 尺寸一致
        mask = Image.open(mp).convert('L')

        if self.is_train:
            # 同步水平翻轉
            if random.random() < 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            # 顏色增強（僅影像）
            img = self.color_jitter(img)

        img  = self.to_tensor(img)
        img  = self.normalize(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask

# -----------------------------
# Collate：同批 padding，到「最大尺寸且可被 stride 整除」
# -----------------------------
def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

def _pad_to(img: torch.Tensor, target_h: int, target_w: int, value: float = 0.0) -> torch.Tensor:
    _, h, w = img.shape
    pad_h = target_h - h
    pad_w = target_w - w
    return F.pad(img, (0, pad_w, 0, pad_h), value=value)  # (left, right, top, bottom)

def _pad_mask_to(mask: torch.Tensor, target_h: int, target_w: int, ignore_val: int = 255) -> torch.Tensor:
    h, w = mask.shape[-2], mask.shape[-1]
    pad_h = target_h - h
    pad_w = target_w - w
    return F.pad(mask, (0, pad_w, 0, pad_h), value=ignore_val)

# Windows 的 DataLoader 使用 spawn，多程序需要可 pickle 的 collate
class CollateTrain:
    def __init__(self, stride: int = 16):
        self.stride = stride

    def __call__(self, batch):
        imgs, masks = zip(*batch)
        h_max = max(t.shape[-2] for t in imgs)
        w_max = max(t.shape[-1] for t in imgs)
        target_h = _round_up(h_max, self.stride)
        target_w = _round_up(w_max, self.stride)
        padded_imgs = [_pad_to(img, target_h, target_w, 0.0) for img in imgs]
        padded_masks = [_pad_mask_to(mask, target_h, target_w, 255) for mask in masks]
        return torch.stack(padded_imgs, 0), torch.stack(padded_masks, 0)

class CollateTest:
    def __init__(self, stride: int = 16):
        self.stride = stride

    def __call__(self, batch):
        # (img, fname, (W,H), path)
        imgs, fnames, orig_sizes, paths = zip(*batch)
        h_max = max(t.shape[-2] for t in imgs)
        w_max = max(t.shape[-1] for t in imgs)
        target_h = _round_up(h_max, self.stride)
        target_w = _round_up(w_max, self.stride)
        padded_imgs = [_pad_to(img, target_h, target_w, 0.0) for img in imgs]
        sizes_tensor = torch.tensor([[w, h] for (w, h) in orig_sizes], dtype=torch.int32)
        return torch.stack(padded_imgs, 0), list(fnames), sizes_tensor, list(paths)

# -----------------------------
# Torchvision 後備包裝（回傳 logits，對齊 smp 介面）
# -----------------------------
class TVDeeplabWrapper(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        m = deeplabv3_resnet50(weights='DEFAULT' if pretrained else None)
        in_ch = m.classifier[-1].in_channels
        m.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.m = m

    def forward(self, x):
        out = self.m(x)['out']  # NxCxH'×W'
        return out

# -----------------------------
# Model（smp DeepLabV3+，含別名/自動回退/OS 防呆）
# -----------------------------
def build_model(num_classes: int = 16, pretrained: bool = True,
                encoder_name: str = 'resnet50',
                encoder_weights: Optional[str] = 'imagenet',
                encoder_output_stride: int = 16):
    """
    優先使用 segmentation_models_pytorch.DeepLabV3Plus；
    若 smp 不可用，回退到 torchvision DeepLabV3。
    - 自動將 timm-resnet*d 別名映射到 resnet50/101
    - 對不支援 dilation 的骨幹（resnest/convnext），強制 OS=32
    """
    try:
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.encoders import encoders

        alias = {
            'timm-resnet50d': 'resnet50',
            'timm-resnet101d': 'resnet101',
        }
        name_in = encoder_name
        encoder_name = alias.get(str(encoder_name).lower(), encoder_name)

        # 不支援 dilation 的骨幹 → OS=32
        if any(k in str(encoder_name).lower() for k in ['resnest', 'convnext']) and encoder_output_stride != 32:
            print("[WARN] Encoder does not support dilated mode; forcing encoder_output_stride=32")
            encoder_output_stride = 32

        if not pretrained:
            encoder_weights = None

        if encoder_name not in encoders:
            print(f"[WARN] Encoder `{name_in}` not found in this smp version. Falling back to `resnet50`.")
            encoder_name = 'resnet50'

        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
            encoder_output_stride=encoder_output_stride
        )
        return model
    except Exception as e:
        print("[WARN] DeepLabV3+ (smp) unavailable:", e)
        print("[WARN] Fallback to torchvision DeepLabV3.")
        return TVDeeplabWrapper(num_classes=num_classes, pretrained=pretrained)

# -----------------------------
# Metrics（mIoU / PixelAcc；忽略 255）
# -----------------------------
def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int = 16, ignore_index: int = 255) -> Tuple[float, List[float]]:
    ious = []
    eps = 1e-6
    valid = (targets != ignore_index)
    for c in range(num_classes):
        p = (preds == c) & valid
        t = (targets == c) & valid
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(inter / (union + eps))
    miou = np.nanmean(ious).item()
    return miou, ious

def compute_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> float:
    valid = targets != ignore_index
    correct = (preds[valid] == targets[valid]).sum().item()
    total = valid.sum().item()
    if total == 0:
        return float('nan')
    return correct / total

def compute_precision_recall_f1(preds: torch.Tensor,
                                targets: torch.Tensor,
                                num_classes: int = 16,
                                ignore_index: int = 255):
    """
    以混淆矩陣計算多類別 Macro Precision / Recall / F1（忽略 255）
    只在 GT 出現過的類別上平均
    """
    valid = targets != ignore_index
    if valid.sum() == 0:
        return float('nan'), float('nan'), float('nan')

    t = targets[valid].view(-1).cpu().numpy()
    p = preds[valid].view(-1).cpu().numpy()

    cm = np.bincount(num_classes * t + p, minlength=num_classes**2).reshape(num_classes, num_classes)

    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    present = cm.sum(axis=1) > 0  # GT 出現過的類別

    precision_c = TP / (TP + FP + 1e-12)
    recall_c    = TP / (TP + FN + 1e-12)
    f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-12)

    if present.any():
        precision = precision_c[present].mean()
        recall    = recall_c[present].mean()
        f1        = f1_c[present].mean()
    else:
        precision = recall = f1 = float('nan')

    return float(precision), float(recall), float(f1)

# -----------------------------
# Loss: CE + Dice（支援 ignore_index=255）
# -----------------------------
class CombinedLoss(nn.Module):
    def __init__(self, num_classes=16, ignore_index=255, ce_weight=1.0, dice_weight=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        total = 0.0
        if self.ce_weight:
            total += self.ce_weight * self.ce(logits, targets)
        if self.dice_weight:
            total += self.dice_weight * multiclass_dice_loss(
                logits, targets,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index
            )
        return total

@torch.no_grad()
def _one_hot_targets(targets, num_classes):
    # 僅供 Dice 計算用；輸入已先過濾 ignore_index
    V = targets.numel()
    one_hot = torch.zeros((V, num_classes), dtype=torch.float32, device=targets.device)
    one_hot.scatter_(1, targets.view(-1, 1), 1.0)
    return one_hot

def multiclass_dice_loss(logits, targets, num_classes=16, ignore_index=255, eps=1e-6):
    # 多類別 Dice，忽略 255
    probs = torch.softmax(logits, dim=1)
    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return logits.new_tensor(0.0, dtype=torch.float32)

    # 壓平成 (V, C) 與 (V,)
    probs = probs.permute(0, 2, 3, 1)[valid]           # (V, C)
    tgt   = targets[valid].long()                       # (V,)
    gt    = _one_hot_targets(tgt, num_classes).to(dtype=probs.dtype, device=probs.device)

    inter = (probs * gt).sum(dim=0)                     # (C,)
    card  = probs.sum(dim=0) + gt.sum(dim=0)            # (C,)
    dice_c = (2.0 * inter + eps) / (card + eps)

    # 只在 GT 出現過的類別上平均
    present = gt.sum(dim=0) > 0
    if present.any():
        dice_c = dice_c[present]

    return 1.0 - dice_c.mean()

# -----------------------------
# Train / Val
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, criterion, ema=None):
    model.train()
    total_loss = 0.0
    correct_pix = 0
    total_pix = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            out = model(imgs)  # NxCxHxW（smp or wrapper 皆回 logits）
            loss = criterion(out, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 參數更新後再做 EMA 更新
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)

        with torch.no_grad():
            pred = out.argmax(dim=1)
            valid = (masks != 255)
            correct_pix += (pred[valid] == masks[valid]).sum().item()
            total_pix   += valid.sum().item()

    train_acc = (correct_pix / total_pix) if total_pix > 0 else float('nan')
    return total_loss / len(loader.dataset), train_acc

@torch.no_grad()
def validate(model, loader, device, num_classes: int = 16, criterion=None):
    model.eval()
    total_loss = 0.0
    correct_pix = 0
    total_pix = 0
    preds_all, targs_all = [], []

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with amp_ctx():
            out = model(imgs)
        if criterion is not None:
            loss = criterion(out, masks)
            total_loss += loss.item() * imgs.size(0)

        pred = out.argmax(dim=1)
        preds_all.append(pred.cpu())
        targs_all.append(masks.cpu())

        valid = (masks != 255)
        correct_pix += (pred[valid] == masks[valid]).sum().item()
        total_pix   += valid.sum().item()

    preds_all = torch.cat(preds_all, dim=0)
    targs_all = torch.cat(targs_all, dim=0)

    miou, _ = compute_miou(preds_all, targs_all, num_classes=num_classes, ignore_index=255)
    val_acc = (correct_pix / total_pix) if total_pix > 0 else float('nan')
    precision, recall, f1 = compute_precision_recall_f1(preds_all, targs_all, num_classes=num_classes, ignore_index=255)

    return ((total_loss / len(loader.dataset)) if criterion is not None else float('nan'),
            miou, val_acc, precision, recall, f1)

# -----------------------------
# TTA: 多尺度 + 左右翻轉（推論用）
# -----------------------------
@torch.no_grad()
def _predict_logits_with_tta(model, imgs, scales=(0.75, 1.0, 1.25)):
    """
    輸出 NxCxH×W 的 logits（與輸入 imgs 同大小）
    - 每個 scale 先將尺寸補到 32 的倍數，滿足 smp 的輸入限制（OS=16/32 皆可）
    - 做左右翻轉 TTA
    - 以 softmax 機率平均，再取 log 再平均
    """
    logits_sum = None
    H, W = imgs.shape[-2:]

    for s in scales:
        # 1) 計算縮放後尺寸，並補到 32 的倍數
        Hs_raw = max(1, int(round(H * s)))
        Ws_raw = max(1, int(round(W * s)))
        Hs = ((Hs_raw + 31) // 32) * 32
        Ws = ((Ws_raw + 31) // 32) * 32

        # 2) 重新取樣到 (Hs, Ws) —— 已是 32 的倍數
        x = F.interpolate(imgs, size=(Hs, Ws), mode='bilinear', align_corners=False)

        # 3) 前向
        with amp_ctx():
            out = model(x)  # N×C×Hs'×Ws'（與 Hs,Ws 相同或更小，smp會處理）
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # 4) flip TTA
        x_flip = torch.flip(x, dims=[-1])
        with amp_ctx():
            out_flip = model(x_flip)
        out_flip = torch.flip(out_flip, dims=[-1])
        out_flip = F.interpolate(out_flip, size=(H, W), mode='bilinear', align_corners=False)

        # 5) 機率平均 → 取對數 → 對所有 scale 做平均
        prob = (F.softmax(out, dim=1) + F.softmax(out_flip, dim=1)) * 0.5
        logits = torch.log(prob.clamp_min(1e-8))

        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / len(scales)

# -----------------------------
# 推論：輸出原始尺寸灰階 mask（像素=類別ID）
# -----------------------------
@torch.no_grad()
def predict_and_save_masks(model,
                           loader,
                           device,
                           save_dir: str,
                           num_classes: int = 16,
                           test_img_roots: Optional[List[str]] = None,
                           tta_scales: Optional[Tuple[float, ...]] = None):
    """
    以原圖尺寸存灰階 mask；檔名與原圖相同
    tta_scales: None 則不用 TTA；否則使用多尺度+flip TTA
    """
    ensure_dir(save_dir)
    model.eval()
    for batch in loader:
        # 支援 (imgs, fnames, sizes, paths) 或 (imgs, fnames, sizes)
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            imgs, fnames, sizes_tensor, _ = batch
        else:
            imgs, fnames, sizes_tensor = batch

        if isinstance(fnames, tuple):
            fnames = list(fnames)

        imgs = imgs.to(device, non_blocking=True)

        # 使用 TTA 或單次前向（上采樣回 padded 尺寸）
        if tta_scales is not None:
            out  = _predict_logits_with_tta(model, imgs, scales=tta_scales)   # N x C x H x W（已對齊）
        else:
            with amp_ctx():
                out  = model(imgs)                                            # N x C x h x w
            out  = F.interpolate(out, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

        pred = out.argmax(dim=1).cpu().numpy()  # N x H_pad x W_pad
        sizes = sizes_tensor.cpu().numpy().tolist()  # [[W,H], ...]

        for i in range(pred.shape[0]):
            w, h = int(sizes[i][0]), int(sizes[i][1])  # (W,H) = 原圖尺寸
            pred_crop = pred[i][:h, :w]  # padding 是往右/下補，原圖在左上
            mask_img = Image.fromarray(pred_crop.astype(np.uint8), mode='L')
            mask_img.save(os.path.join(save_dir, fnames[i]))

# -----------------------------
# 路徑助手
# -----------------------------
def default_paths(root='./machine_learning/UAV_dataset'):
    train_img_roots = [
        os.path.join(root, 'train', 'imgs', 'Fallen'),
        os.path.join(root, 'train', 'imgs', 'Normal'),
        os.path.join(root, 'train', 'imgs', 'Rain'),
        os.path.join(root, 'train', 'imgs', 'Snow'),
    ]
    train_mask_roots = [
        os.path.join(root, 'train', 'masks', 'Fallen'),
        os.path.join(root, 'train', 'masks', 'Normal'),
        os.path.join(root, 'train', 'masks', 'Rain'),
        os.path.join(root, 'train', 'masks', 'Snow'),
    ]
    test_img_roots = [
        os.path.join(root, 'test', 'imgs', 'Dust'),
        os.path.join(root, 'test', 'imgs', 'Fallen'),
        os.path.join(root, 'test', 'imgs', 'Fog'),
        os.path.join(root, 'test', 'imgs', 'Normal'),
        os.path.join(root, 'test', 'imgs', 'Rain'),
        os.path.join(root, 'test', 'imgs', 'Snow'),
    ]
    return train_img_roots, train_mask_roots, test_img_roots

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./machine_learning/UAV_dataset')
    parser.add_argument('--image_size', type=int, default=512)  # 保留參數，不使用
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_pretrain', action='store_true', help='不載入 ImageNet 預訓練（encoder_weights=None）')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--out_dir', type=str, default='./outputs_hw2')
    parser.add_argument('--only_infer', action='store_true', help='只推論 best.ckpt')
    parser.add_argument('--resume', type=str, default='', help='從檢查點續訓')
    parser.add_argument('--save_best_metric', type=str, default='miou', choices=['miou','loss'])
    # Dice + CE 權重
    parser.add_argument('--ce_weight', type=float, default=1.0, help='CE loss 權重')
    parser.add_argument('--dice_weight', type=float, default=0.7, help='Dice loss 權重')

    # 測試輸出 mask 路徑（灰階像素=類別ID 0..15）
    parser.add_argument('--save_masks_dir', type=str, default='./test_data',
                        help='輸出 test 預測的灰階 mask 到此資料夾')

    # EMA 相關
    parser.add_argument('--ema', action='store_true', help='啟用 EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA 衰減（越近 1 越平滑）')
    parser.add_argument('--ema_start', type=int, default=1, help='自第幾個 epoch 開始套用 EMA（1-based）')
    parser.add_argument('--ema_device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='EMA 權重放置位置')

    # TTA 相關
    parser.add_argument('--tta', action='store_true', help='推論時啟用 TTA（多尺度+flip）')
    parser.add_argument('--tta_scales', type=str, default='0.75,1.0,1.25',
                        help='TTA 尺度列表，逗號分隔，例如：0.75,1.0,1.25')

    # smp DeepLabV3+ 相關
    parser.add_argument('--encoder_name', type=str, default='resnet50',
                        help='smp encoder：如 resnet50/resnet101/efficientnet-b4/timm-resnest50d 等（本版未必支援全部）')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='smp encoder 預訓練權重：imagenet / ssl / noisy-student / None')
    parser.add_argument('--encoder_output_stride', type=int, default=16, choices=[8,16,32],
                        help='DeepLabV3+ encoder output stride；ResNeSt/ConvNeXt 請用 32')

    args = parser.parse_args()

    # CUDA 不可用時，--ema_device=cuda 自動退回 CPU
    if args.ema and args.ema_device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] --ema_device=cuda requested but CUDA not available; falling back to CPU.")
        args.ema_device = 'cpu'

    # 解析 TTA 尺度
    tta_scales_tuple: Optional[Tuple[float, ...]] = None
    if args.tta:
        try:
            tta_scales_tuple = tuple(float(x) for x in args.tta_scales.split(',') if x.strip())
        except Exception:
            print("[WARN] Invalid --tta_scales, fallback to (0.75, 1.0, 1.25)")
            tta_scales_tuple = (0.75, 1.0, 1.25)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_dir)

    train_img_roots, train_mask_roots, test_img_roots = default_paths(args.root)

    # --- 先建立完整 dataset 以利切分，再各自建 train/val Subset ---
    full_ds_for_split = UAVSegDataset(train_img_roots, train_mask_roots, image_size=args.image_size, is_train=True)
    n_total = len(full_ds_for_split)
    if n_total == 0:
        raise FileNotFoundError(
            f"No training images found under '{args.root}'. Expected:\n"
            f"  train/imgs/{{Fallen,Normal,Rain,Snow}}\n"
            f"  train/masks/{{Fallen,Normal,Rain,Snow}}"
        )
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_data = UAVSegDataset(train_img_roots, train_mask_roots, image_size=args.image_size, is_train=True)
    val_data   = UAVSegDataset(train_img_roots, train_mask_roots, image_size=args.image_size, is_train=False)

    train_set = Subset(train_data, train_idx)
    val_set   = Subset(val_data,   val_idx)

    # Test 無標註（只輸出 mask）
    test_set = UAVSegDataset(test_img_roots, None, image_size=args.image_size, is_train=False)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        collate_fn=CollateTrain(stride=args.encoder_output_stride)
    )
    val_loader   = DataLoader(
        val_set, batch_size=max(1, args.batch_size*2), shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=CollateTrain(stride=args.encoder_output_stride)
    )
    test_loader  = DataLoader(
        test_set, batch_size=max(1, args.batch_size*2), shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=CollateTest(stride=args.encoder_output_stride)
    )

    # 建模（smp DeepLabV3+，失敗則回退到 torchvision）
    model = build_model(
        num_classes=16,
        pretrained=(not args.no_pretrain),
        encoder_name=args.encoder_name,
        encoder_weights=None if args.no_pretrain else (None if str(args.encoder_weights).lower() in ['none'] else args.encoder_weights),
        encoder_output_stride=args.encoder_output_stride
    )
    model.to(device)
    print("Device in use:", next(model.parameters()).device)

    # 優化器/排程器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 損失（CE + Dice）
    criterion = CombinedLoss(
        num_classes=16,
        ignore_index=255,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight
    )
    scaler = make_scaler()

    # 啟用 EMA（可選）
    ema = None
    if args.ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=args.ema_device)
        print(f"[INFO] EMA enabled: decay={args.ema_decay}, device={args.ema_device}, start@epoch>={args.ema_start}")

    # 訓練歷史（畫圖用）
    history = {'train_loss': [], 'train_acc':  [], 'val_loss':   [], 'val_acc':    []}

    # 依指標挑 best
    best_metric = -float('inf') if args.save_best_metric=='miou' else float('inf')
    best_path = os.path.join(args.out_dir, 'best.ckpt')

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        # 續訓載入（含 EMA 狀態）
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['sched'])
        start_epoch = ckpt.get('epoch', 0)
        if ema is not None and ('ema' in ckpt):
            ema.load_state_dict(ckpt['ema'])
            print("[INFO] EMA state loaded from checkpoint")
        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    if not args.only_infer:
        print(f"[INFO] Train: {len(train_set)}  |  Val: {len(val_set)}  |  Test: {len(test_set)}")
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()

            # 本輪是否使用 EMA
            use_ema_this_epoch = (ema is not None) and ((epoch + 1) >= args.ema_start)

            # 訓練
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, device, criterion,
                ema=ema if use_ema_this_epoch else None
            )

            # 驗證（可切換 EMA 權重）
            if use_ema_this_epoch:
                with use_ema_weights(ema, model):
                    val_loss, val_miou, val_acc, val_prec, val_rec, val_f1 = validate(
                        model, val_loader, device, num_classes=16, criterion=criterion)
            else:
                val_loss, val_miou, val_acc, val_prec, val_rec, val_f1 = validate(
                    model, val_loader, device, num_classes=16, criterion=criterion)

            scheduler.step()
            dt = time.time()-t0

            # 紀錄
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  val_mIoU={val_miou:.3f}  "
                  f"val_P={val_prec:.3f}  val_R={val_rec:.3f}  val_F1={val_f1:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  time={dt/60:.1f}m")

            # 存 last（含 EMA 狀態）
            save_obj = {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict()
            }
            if ema is not None:
                save_obj['ema'] = ema.state_dict()
            torch.save(save_obj, os.path.join(args.out_dir, 'last.ckpt'))

            # 存 best（若用 EMA 驗證，best 亦用 EMA 權重）
            if args.save_best_metric == 'miou':
                is_better = val_miou > best_metric
                candidate_metric = val_miou
            else:
                is_better = val_loss < best_metric
                candidate_metric = val_loss

            if is_better:
                best_metric = candidate_metric
                if use_ema_this_epoch:
                    full_ema_sd = ema.full_state_dict_like(model)
                    torch.save(full_ema_sd, best_path)
                else:
                    torch.save(model.state_dict(), best_path)

        print(f"[INFO] Best {args.save_best_metric}: {best_metric:.4f} saved to {best_path}")

        # ===== 畫圖（x 軸整數刻度）=====
        ensure_dir("./result")
        epochs_axis = np.arange(1, len(history['train_loss']) + 1)

        # loss 曲線
        plt.figure()
        plt.plot(epochs_axis, history['train_loss'], label='train_loss')
        plt.plot(epochs_axis, history['val_loss'],   label='val_loss')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(epochs_axis) > 1:
            ax.set_xlim(1, len(epochs_axis))
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training / Validation Loss')
        plt.grid(True, linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
        plt.savefig('./result/loss.png', dpi=150); plt.close()

        # accuracy 曲線
        plt.figure()
        plt.plot(epochs_axis, history['train_acc'], label='train_acc')
        plt.plot(epochs_axis, history['val_acc'],   label='val_acc')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(epochs_axis) > 1:
            ax.set_xlim(1, len(epochs_axis))
        plt.xlabel('Epoch'); plt.ylabel('Pixel Accuracy'); plt.title('Training / Validation Accuracy')
        plt.grid(True, linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
        plt.savefig('./result/acc.png', dpi=150); plt.close()

    # 推論前載入 best（可能是 EMA 組合好的完整權重）
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best weights from {best_path}")
    else:
        print("[WARN] Best checkpoint not found; using current weights.")

    # === 在驗證集上用 best 權重評估並輸出 CSV（小數三位） ===
    eval_loss, eval_miou, eval_acc, eval_prec, eval_rec, eval_f1 = validate(
        model, val_loader, device, num_classes=16, criterion=criterion
    )
    csv_path = os.path.join(args.out_dir, "Evaluation_Metrics.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model\\metric", "Accuracy", "Precision", "Recall", "F1 Score", "Miou"])
        writer.writerow([
            "DeepLabV3+-SMP",
            f"{eval_acc:.3f}",
            f"{eval_prec:.3f}",
            f"{eval_rec:.3f}",
            f"{eval_f1:.3f}",
            f"{eval_miou:.3f}",
        ])
    print(f"[OK] Saved evaluation metrics to: {csv_path}")

    # 僅輸出 test 的灰階 mask（按原圖大小，裁回原尺寸）
    if ema is not None:
        with use_ema_weights(ema, model):
            predict_and_save_masks(
                model, test_loader, device, args.save_masks_dir,
                num_classes=16, test_img_roots=test_img_roots,
                tta_scales=tta_scales_tuple
            )
    else:
        predict_and_save_masks(
            model, test_loader, device, args.save_masks_dir,
            num_classes=16, test_img_roots=test_img_roots,
            tta_scales=tta_scales_tuple
        )

    print(f"[OK] Saved test masks to: {args.save_masks_dir}")
    return 0

if __name__ == '__main__':
    main()
