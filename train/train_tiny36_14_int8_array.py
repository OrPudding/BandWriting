#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============== 配置 ===============
EPOCHS = 20
BATCH = 512
LR = 1e-3
DATA_DIR = "./data"
OUT_DIR = Path("./out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 36 类：0-9 + a-z (EMNIST byclass: 0-9 + A-Z + a-z)
VALID_LABELS = list(range(0, 10)) + list(range(36, 62))  # digits + lowercase
LABEL_MAP = {v: i for i, v in enumerate(VALID_LABELS)}   # original -> 0..35


# =============== 变换：顺时针 90° + 下采样到 14x14 ===============
def rot90_cw(x: torch.Tensor) -> torch.Tensor:
    # x: [1,28,28]
    return torch.rot90(x, -1, [1, 2])

def downsample_14(x: torch.Tensor) -> torch.Tensor:
    # x: [1,28,28] -> [1,14,14]
    x4 = x.unsqueeze(0)  # [1,1,28,28]
    y4 = F.interpolate(x4, size=(14, 14), mode="bilinear", align_corners=False)
    return y4.squeeze(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(rot90_cw),
    transforms.Lambda(downsample_14),
])


# =============== Dataset：过滤 + remap（可多进程）==============
class RemapDataset(Dataset):
    def __init__(self, subset: Subset, label_map: dict):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return x, self.label_map[int(y)]


def filter_and_remap(ds):
    # 用 targets 过滤（快且不触发 transform）
    targets = ds.targets
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for v in VALID_LABELS:
        mask |= (targets == v)
    idxs = mask.nonzero(as_tuple=False).view(-1).tolist()
    return RemapDataset(Subset(ds, idxs), LABEL_MAP)


# =============== 模型（14x14 输入，两次池化后到 3x3）==============
class TinyCNN14(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 14x14 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 7x7 -> 7x7（池化后）
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)                   # /2 (floor)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)         # 14->7->3
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 14->7
        x = self.pool(self.relu(self.conv2(x)))  # 7->3
        x = x.flatten(1)                         # 32*3*3
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============== 导出：int8 ===============
def quant_int8_weight(w: torch.Tensor):
    w = w.detach().cpu().contiguous().float()
    max_abs = float(w.abs().max().item())
    if max_abs < 1e-12:
        scale = 1.0
        q = torch.zeros_like(w, dtype=torch.int8)
    else:
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, float(scale)

def flatten_list(t: torch.Tensor):
    return t.detach().cpu().contiguous().view(-1).tolist()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 数据集
    train_raw = datasets.EMNIST(root=DATA_DIR, split="byclass", train=True, download=True, transform=transform)
    test_raw  = datasets.EMNIST(root=DATA_DIR, split="byclass", train=False, download=True, transform=transform)

    train_ds = filter_and_remap(train_raw)
    test_ds  = filter_and_remap(test_raw)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    # 模型
    model = TinyCNN14().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_acc = 0.0
    best_path = OUT_DIR / "tiny36_14_best_float.pth"

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        scheduler.step()

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))

        acc = correct / max(1, total)
        print(f"\nEpoch {epoch+1} Loss: {total_loss:.2f}")
        print(f"Test Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print("==> Saved best model")

    print("\nTraining finished. Best Accuracy:", best_acc)
    print("best model:", best_path)

    # ===== 导出 int8 JSON（layers 数组）=====
    state = torch.load(best_path, map_location="cpu")
    model_cpu = TinyCNN14().cpu()
    model_cpu.load_state_dict(state)
    model_cpu.eval()

    q_conv1_w, s_conv1 = quant_int8_weight(model_cpu.conv1.weight)
    b_conv1 = model_cpu.conv1.bias.detach().cpu().float()

    q_conv2_w, s_conv2 = quant_int8_weight(model_cpu.conv2.weight)
    b_conv2 = model_cpu.conv2.bias.detach().cpu().float()

    q_fc1_w, s_fc1 = quant_int8_weight(model_cpu.fc1.weight)
    b_fc1 = model_cpu.fc1.bias.detach().cpu().float()

    q_fc2_w, s_fc2 = quant_int8_weight(model_cpu.fc2.weight)
    b_fc2 = model_cpu.fc2.bias.detach().cpu().float()

    export = {
        "meta": {
            "name": "tiny36_cw90_ds14_int8_array",
            "classes": 36,
            "labels": "0-9,a-z",
            "input": {"h": 14, "w": 14, "layout": "HW_gray_0_1"},
            "preprocess": "rot90_cw_then_downsample14",
            "note": "trained on EMNIST(byclass) digits+lowercase, rotated 90deg clockwise, downsampled to 14x14",
        },
        "layers": [
            {
                "type": "conv2d",
                "name": "conv1",
                "inC": 1, "outC": 16, "kH": 3, "kW": 3, "pad": 1,
                "w_q": flatten_list(q_conv1_w.to(torch.int16)),
                "w_scale": s_conv1,
                "b": flatten_list(b_conv1),
            },
            {"type": "maxpool2d", "name": "pool1", "k": 2, "s": 2},
            {
                "type": "conv2d",
                "name": "conv2",
                "inC": 16, "outC": 32, "kH": 3, "kW": 3, "pad": 1,
                "w_q": flatten_list(q_conv2_w.to(torch.int16)),
                "w_scale": s_conv2,
                "b": flatten_list(b_conv2),
            },
            {"type": "maxpool2d", "name": "pool2", "k": 2, "s": 2},
            {
                "type": "fc",
                "name": "fc1",
                "in": 32 * 3 * 3, "out": 128,
                "w_q": flatten_list(q_fc1_w.to(torch.int16)),
                "w_scale": s_fc1,
                "b": flatten_list(b_fc1),
            },
            {
                "type": "fc",
                "name": "fc2",
                "in": 128, "out": 36,
                "w_q": flatten_list(q_fc2_w.to(torch.int16)),
                "w_scale": s_fc2,
                "b": flatten_list(b_fc2),
            }
        ]
    }

    out_json = OUT_DIR / "tiny36_14_int8_array.json"
    out_json.write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
    print("exported:", out_json, "bytes=", out_json.stat().st_size)


if __name__ == "__main__":
    # forkserver/spawn 下必须用这个保护
    main()
