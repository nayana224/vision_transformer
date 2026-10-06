from pathlib import Path
import csv

from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)


# ============================================
# 1. 기본 설정
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

MANIFEST_PATH = ASSETS_DIR / "oxford_iiit_pet_samples.csv"

TOP_K = 5


# ============================================
# 2. device 설정
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")
print()


# ============================================
# 3. Pretrained ViT-B/16 불러오기
# ============================================

weights = ViT_B_16_Weights.IMAGENET1K_V1

model = vit_b_16(
    weights=weights,
)

model = model.to(device)
model.eval()

print("Loaded pretrained model:")
print("ViT-B/16")
print()


# ============================================
# 4. Pretrained model용 preprocessing
# ============================================

preprocess = weights.transforms()

print("Preprocessing:")
print(preprocess)
print()


# ============================================
# 5. Oxford-IIIT Pet sample 목록 불러오기
# ============================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        "Dataset sample manifest was not found.\n"
        "Run this first:\n"
        "  python study/00_prepare_dataset.py"
    )

with MANIFEST_PATH.open(
    "r",
    newline="",
    encoding="utf-8",
) as f:
    reader = csv.DictReader(f)
    samples = list(reader)

if not samples:
    raise RuntimeError(
        f"No samples were found in {MANIFEST_PATH}"
    )


# ============================================
# 6. ImageNet class 이름 가져오기
# ============================================

categories = weights.meta["categories"]


# ============================================
# 7. 각 sample inference
# ============================================

for sample in samples:
    image_path = ASSETS_DIR / sample["filename"]
    gt_class = sample["class_name"]

    image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(image)

    # [C, H, W] -> [B, C, H, W]
    input_batch = input_tensor.unsqueeze(0)

    input_batch = input_batch.to(device)

    with torch.no_grad():
        logits = model(input_batch)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=TOP_K,
        dim=1,
    )

    print("======================================")
    print(f"Image      : {sample['filename']}")
    print(f"Pet class  : {gt_class}")
    print(f"Input shape: {tuple(input_batch.shape)}")
    print()

    print(f"Top-{TOP_K} ImageNet predictions")

    for rank in range(TOP_K):
        class_index = top_indices[0, rank].item()
        probability = top_probabilities[0, rank].item()

        class_name = categories[class_index]

        print(
            f"{rank + 1}. "
            f"{class_name:<30} "
            f"{probability * 100:6.2f}%"
        )

    print()