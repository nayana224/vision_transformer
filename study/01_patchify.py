from pathlib import Path
import csv

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


# ============================================
# 1. 경로 및 기본 설정
# ============================================

IMAGE_SIZE = 224
PATCH_SIZE = 16

NUM_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_PER_SIDE ** 2

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

MANIFEST_PATH = ASSETS_DIR / "oxford_iiit_pet_samples.csv"


# ============================================
# 2. 00_prepare_dataset.py가 선택한 첫 이미지 찾기
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

sample = samples[0]

IMAGE_PATH = ASSETS_DIR / sample["filename"]
CLASS_NAME = sample["class_name"]
DATASET_INDEX = int(sample["dataset_index"])

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Sample image was not found: {IMAGE_PATH}\n"
        "Run this first:\n"
        "  python study/00_prepare_dataset.py"
    )


# ============================================
# 3. 이미지 불러오기
# ============================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("Selected Oxford-IIIT Pet sample")
print(f"Class         : {CLASS_NAME}")
print(f"Dataset index : {DATASET_INDEX}")
print(f"Image path    : {IMAGE_PATH}")
print(f"Original size : {image.size}")
print()


# ============================================
# 4. 이미지 전처리
# ============================================

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

image_tensor = transform(image)

print("Image tensor shape")
print(image_tensor.shape)
print()


# ============================================
# 5. Patch 개수 확인
# ============================================

print(f"Image size        : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Patch size        : {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Patches per side  : {NUM_PATCHES_PER_SIDE}")
print(f"Number of patches : {NUM_PATCHES}")
print()


# ============================================
# 6. unfold로 16x16 patch 분할
# ============================================

patches = image_tensor.unfold(
    dimension=1,
    size=PATCH_SIZE,
    step=PATCH_SIZE,
)

patches = patches.unfold(
    dimension=2,
    size=PATCH_SIZE,
    step=PATCH_SIZE,
)

print("Shape after unfold")
print(patches.shape)
print()


# ============================================
# 7. dimension 순서를 보기 좋게 정리
# ============================================

patches = patches.permute(1, 2, 0, 3, 4)

print("Shape after permute")
print(patches.shape)
print()


# ============================================
# 8. 2D patch grid를 sequence로 변환
# ============================================

patch_sequence = patches.reshape(
    NUM_PATCHES,
    3,
    PATCH_SIZE,
    PATCH_SIZE,
)

print("Patch sequence shape")
print(patch_sequence.shape)
print()


# ============================================
# 9. 각 patch를 하나의 vector로 flatten
# ============================================

flattened_patches = patch_sequence.reshape(
    NUM_PATCHES,
    -1,
)

print("Flattened patch shape")
print(flattened_patches.shape)
print()

print(
    "Expected flattened dimension:",
    PATCH_SIZE * PATCH_SIZE * 3,
)
print()


# ============================================
# 10. 첫 번째 patch 값 일부 확인
# ============================================

print("First patch - first 20 flattened values")
print(flattened_patches[0, :20])
print()


# ============================================
# 11. Patch grid 시각화
# ============================================

fig, axes = plt.subplots(
    NUM_PATCHES_PER_SIDE,
    NUM_PATCHES_PER_SIDE,
    figsize=(10, 10),
)

for i in range(NUM_PATCHES):
    row = i // NUM_PATCHES_PER_SIDE
    col = i % NUM_PATCHES_PER_SIDE

    patch = patch_sequence[i]

    # PyTorch image: [C, H, W]
    # Matplotlib image: [H, W, C]
    patch_image = patch.permute(1, 2, 0)

    axes[row, col].imshow(patch_image)
    axes[row, col].axis("off")

fig.suptitle(
    f"Oxford-IIIT Pet | {CLASS_NAME} | "
    f"{IMAGE_SIZE}x{IMAGE_SIZE} -> "
    f"{NUM_PATCHES} patches ({PATCH_SIZE}x{PATCH_SIZE})",
    fontsize=12,
)

plt.tight_layout()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = OUTPUT_DIR / (
    f"01_patchify_{Path(sample['filename']).stem}.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print("Patch visualization saved to")
print(OUTPUT_PATH)
