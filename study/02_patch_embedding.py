from pathlib import Path
import csv

from PIL import Image
import torch
from torch import nn
from torchvision import transforms


# ============================================
# 1. 기본 설정
# ============================================

IMAGE_SIZE = 224
PATCH_SIZE = 16
CHANNELS = 3

PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS
# Study setting:
# ViT-Base  : D = 768
# ViT-Large : D = 1024
# ViT-Huge  : D = 1280
#
# 여기서는 tensor flow를 보기 쉽게 D=128로 축소해서 사용한다.
EMBED_DIM = 128

NUM_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_PER_SIDE ** 2

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

MANIFEST_PATH = ASSETS_DIR / "oxford_iiit_pet_samples.csv"


# ============================================
# 2. 00_prepare_dataset.py에서 만든 첫 이미지 선택
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

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Sample image was not found: {IMAGE_PATH}"
    )


# ============================================
# 3. 이미지 불러오기 및 tensor 변환
# ============================================

image = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

image_tensor = transform(image)

print("Selected sample")
print(f"Class      : {CLASS_NAME}")
print(f"Image path : {IMAGE_PATH}")
print()

print("Image tensor shape")
print(image_tensor.shape)
print()


# ============================================
# 4. 16x16 patch로 분할
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

patches = patches.permute(
    1,
    2,
    0,
    3,
    4,
)

patch_sequence = patches.reshape(
    NUM_PATCHES,
    CHANNELS,
    PATCH_SIZE,
    PATCH_SIZE,
)

flattened_patches = patch_sequence.reshape(
    NUM_PATCHES,
    PATCH_DIM,
)

print("Flattened patch shape")
print(flattened_patches.shape)
print()


# ============================================
# 5. Patch Embedding layer 정의
# ============================================

patch_projection = nn.Linear(
    in_features=PATCH_DIM,
    out_features=EMBED_DIM,
)

print("Patch projection layer")
print(patch_projection)
print()

print("Projection weight shape")
print(patch_projection.weight.shape)
print()

print("Projection bias shape")
print(patch_projection.bias.shape)
print()


# ============================================
# 6. Linear Projection 수행
# ============================================

patch_embeddings = patch_projection(
    flattened_patches
)

print("Patch embedding shape")
print(patch_embeddings.shape)
print()


# ============================================
# 7. 첫 번째 patch 비교
# ============================================

print("First raw flattened patch [0, :20]")
print(flattened_patches[0, :20])
print()

print("First patch embedding [0, :20]")
print(patch_embeddings[0, :20])
print()


# ============================================
# 8. shape 요약
# ============================================

print("======================================")
print("Patch Embedding Summary")
print("======================================")

print(
    f"Image              : "
    f"{IMAGE_SIZE} x {IMAGE_SIZE} x {CHANNELS}"
)

print(
    f"Number of patches  : "
    f"{NUM_PATCHES}"
)

print(
    f"Raw patch dimension: "
    f"{PATCH_DIM}"
)

print(
    f"Embedding dimension: "
    f"{EMBED_DIM}"
)

print(
    f"Input shape         : "
    f"{tuple(flattened_patches.shape)}"
)

print(
    f"Output shape        : "
    f"{tuple(patch_embeddings.shape)}"
)