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

# Study setting
# ViT-Base  : D = 768
# ViT-Large : D = 1024
# ViT-Huge  : D = 1280
EMBED_DIM = 128

NUM_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_PER_SIDE ** 2
SEQUENCE_LENGTH = NUM_PATCHES + 1

# Multi-Head Attention
NUM_HEADS = 4

# ViT의 MLP는 보통 hidden dimension을 더 크게 사용
MLP_DIM = 256

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

MANIFEST_PATH = ASSETS_DIR / "oxford_iiit_pet_samples.csv"


# ============================================
# 2. 첫 번째 sample 선택
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


# ============================================
# 3. 이미지 -> patch sequence
# ============================================

image = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

image_tensor = transform(image)

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


# ============================================
# 4. Patch Embedding
# ============================================

patch_projection = nn.Linear(
    PATCH_DIM,
    EMBED_DIM,
)

patch_embeddings = patch_projection(
    flattened_patches
)


# ============================================
# 5. CLS token + Position Embedding
# ============================================

cls_token = nn.Parameter(
    torch.zeros(1, EMBED_DIM)
)

tokens = torch.cat(
    [
        cls_token,
        patch_embeddings,
    ],
    dim=0,
)

position_embedding = nn.Parameter(
    torch.zeros(
        SEQUENCE_LENGTH,
        EMBED_DIM,
    )
)

x = tokens + position_embedding

print("Transformer input shape")
print(x.shape)
print()


# ============================================
# 6. Batch dimension 추가
# ============================================

# MultiheadAttention에서 batch_first=True를 사용하므로
# 입력 shape은 [batch, sequence, embedding] 형태여야 한다.
x = x.unsqueeze(0)

print("After adding batch dimension")
print(x.shape)
print()


# ============================================
# 7. Encoder Block 구성
# ============================================

norm1 = nn.LayerNorm(
    EMBED_DIM
)

attention = nn.MultiheadAttention(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    batch_first=True,
)

norm2 = nn.LayerNorm(
    EMBED_DIM
)

mlp = nn.Sequential(
    nn.Linear(
        EMBED_DIM,
        MLP_DIM,
    ),
    nn.GELU(),
    nn.Linear(
        MLP_DIM,
        EMBED_DIM,
    ),
)


# ============================================
# 8. Pre-LN + Multi-Head Self-Attention
# ============================================

residual = x

x_norm = norm1(x)

attention_output, attention_weights = attention(
    query=x_norm,
    key=x_norm,
    value=x_norm,
    need_weights=True,
)

x = residual + attention_output

print("After Multi-Head Self-Attention")
print(x.shape)
print()

print("Attention weight shape")
print(attention_weights.shape)
print()


# ============================================
# 9. Pre-LN + MLP
# ============================================

residual = x

x_norm = norm2(x)

mlp_output = mlp(x_norm)

x = residual + mlp_output

print("After MLP")
print(x.shape)
print()


# ============================================
# 10. CLS token 확인
# ============================================

final_cls_token = x[:, 0]

print("Final CLS token shape")
print(final_cls_token.shape)
print()

print("Final CLS token - first 10 values")
print(final_cls_token[0, :10])
print()


# ============================================
# 11. 전체 shape 요약
# ============================================

print("======================================")
print("Transformer Encoder Summary")
print("======================================")

print(
    f"Input                : "
    f"(1, {SEQUENCE_LENGTH}, {EMBED_DIM})"
)

print(
    f"After Attention      : "
    f"{tuple(x.shape)}"
)

print(
    f"Final CLS token      : "
    f"{tuple(final_cls_token.shape)}"
)

print()
print("Encoder structure")
print(
    "LN -> Multi-Head Self-Attention "
    "-> Residual"
)
print(
    "LN -> MLP -> Residual"
)