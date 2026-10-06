from pathlib import Path
import csv

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)


# ============================================
# 1. 기본 설정
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

MANIFEST_PATH = ASSETS_DIR / "oxford_iiit_pet_samples.csv"

# 비교 대상
TARGET_SAMPLE_NUMBERS = [1, 3, 4]

PATCH_SIZE = 16
GRID_SIZE = 14


# ============================================
# 2. Pretrained ViT-B/16
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = ViT_B_16_Weights.IMAGENET1K_V1

model = vit_b_16(
    weights=weights,
)

model = model.to(device)
model.eval()

# 실제 모델 inference용 preprocessing
preprocess = weights.transforms()

categories = weights.meta["categories"]

print(f"Device: {device}")
print("Loaded pretrained ViT-B/16")
print()


# ============================================
# 3. Visualization용 geometric preprocessing
# ============================================

# 실제 pretrained model이 보는 공간과 맞추기 위해
# Resize -> CenterCrop까지 동일하게 적용한다.
#
# Normalize는 visualization에는 적용하지 않는다.
display_transform = transforms.Compose(
    [
        transforms.Resize(
            256,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(224),
    ]
)


# ============================================
# 4. Sample 목록
# ============================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        "Dataset sample manifest was not found.\n"
        "Run:\n"
        "  python study/00_prepare_dataset.py"
    )

with MANIFEST_PATH.open(
    "r",
    newline="",
    encoding="utf-8",
) as f:
    reader = csv.DictReader(f)
    samples = list(reader)

selected_samples = [
    samples[i - 1]
    for i in TARGET_SAMPLE_NUMBERS
]


# ============================================
# 5. Attention까지 직접 추출하는 forward
# ============================================

def forward_with_attention(model, x):
    """
    torchvision ViT-B/16을 직접 순전파하면서
    각 Transformer layer의 head별 attention을 저장한다.
    """

    batch_size = x.shape[0]

    # ----------------------------------------
    # Patch Embedding
    # ----------------------------------------

    x = model._process_input(x)

    print(
        "Patch embedding :",
        tuple(x.shape),
    )

    # [B, 196, 768]

    # ----------------------------------------
    # CLS token
    # ----------------------------------------

    cls_token = model.class_token.expand(
        batch_size,
        -1,
        -1,
    )

    x = torch.cat(
        [
            cls_token,
            x,
        ],
        dim=1,
    )

    print(
        "After CLS       :",
        tuple(x.shape),
    )

    # [B, 197, 768]

    # ----------------------------------------
    # Position Embedding
    # ----------------------------------------

    x = x + model.encoder.pos_embedding
    x = model.encoder.dropout(x)

    attention_maps = []

    # ----------------------------------------
    # Transformer Encoder
    # ----------------------------------------

    for layer_index, block in enumerate(
        model.encoder.layers
    ):
        # Pre-LN
        x_norm = block.ln_1(x)

        attention_output, attention_weights = (
            block.self_attention(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=True,
                average_attn_weights=False,
            )
        )

        # [B, 12, 197, 197]
        attention_maps.append(
            attention_weights.detach()
        )

        # Residual
        x = x + block.dropout(
            attention_output
        )

        # Pre-LN -> MLP -> Residual
        y = block.ln_2(x)
        y = block.mlp(y)

        x = x + y

        print(
            f"Layer {layer_index + 1:02d}:",
            tuple(attention_weights.shape),
        )

    # Final LayerNorm
    x = model.encoder.ln(x)

    # CLS token
    cls_output = x[:, 0]

    # Classification head
    logits = model.heads(
        cls_output
    )

    return logits, attention_maps


# ============================================
# 6. Attention Rollout
# ============================================

def attention_rollout(attention_maps):
    """
    Attention Rollout

    1. 여러 head의 attention을 평균
    2. residual connection을 identity로 반영
    3. row normalization
    4. layer별 attention matrix를 순서대로 누적 곱

    반환값:
        [197, 197]

    rollout[i, j]는 여러 layer를 거쳐 누적된
    token i -> token j attention-flow score로 해석한다.
    """

    num_tokens = attention_maps[0].shape[-1]

    rollout = torch.eye(
        num_tokens,
        device=attention_maps[0].device,
    )

    for attention in attention_maps:

        # [1, heads, 197, 197]
        # ->
        # [1, 197, 197]
        attention = attention.mean(
            dim=1
        )

        # batch size = 1
        attention = attention[0]

        # Residual connection
        attention = attention + torch.eye(
            num_tokens,
            device=attention.device,
        )

        # 각 query row의 합을 1로 정규화
        attention = (
            attention
            / attention.sum(
                dim=-1,
                keepdim=True,
            )
        )

        # Layer별 누적
        rollout = attention @ rollout

    return rollout


# ============================================
# 7. CLS rollout -> 14x14 patch grid
# ============================================

def get_cls_rollout_grid(rollout):
    """
    index 0:
        CLS token

    index 1~196:
        image patch tokens
    """

    cls_to_patches = rollout[
        0,
        1:,
    ]

    rollout_grid = cls_to_patches.reshape(
        GRID_SIZE,
        GRID_SIZE,
    )

    # Visualization을 위한 상대값 normalization
    #
    # 0 = 이 이미지에서 상대적으로 낮은 rollout score
    # 1 = 이 이미지에서 상대적으로 높은 rollout score
    rollout_grid = (
        rollout_grid
        - rollout_grid.min()
    )

    rollout_grid = (
        rollout_grid
        / rollout_grid.max().clamp(
            min=1e-8
        )
    )

    return rollout_grid.cpu()


# ============================================
# 8. 14x14 -> 224x224
# ============================================

def upsample_rollout(
    rollout_grid,
):
    rollout_map = rollout_grid.unsqueeze(
        0
    ).unsqueeze(0)

    rollout_map = F.interpolate(
        rollout_map,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    return rollout_map[
        0,
        0,
    ]


# ============================================
# 9. 가장 높은 rollout patch 찾기
# ============================================

def find_max_patch(
    rollout_grid,
):
    flat_index = torch.argmax(
        rollout_grid
    ).item()

    row = flat_index // GRID_SIZE
    col = flat_index % GRID_SIZE

    score = rollout_grid[
        row,
        col,
    ].item()

    return row, col, score


# ============================================
# 10. 각 sample 분석
# ============================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for sample in selected_samples:

    image_path = (
        ASSETS_DIR
        / sample["filename"]
    )

    gt_class = sample["class_name"]

    original_image = Image.open(
        image_path
    ).convert("RGB")

    # 실제 모델 입력
    input_tensor = preprocess(
        original_image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )

    # Visualization용 이미지
    # 모델의 geometric preprocessing과 같은 crop
    display_image = display_transform(
        original_image
    )

    print()
    print(
        "======================================"
    )

    print(
        f"Image: {sample['filename']}"
    )

    print(
        f"GT   : {gt_class}"
    )

    print()

    # ----------------------------------------
    # Forward
    # ----------------------------------------

    with torch.no_grad():
        logits, attention_maps = (
            forward_with_attention(
                model,
                input_tensor,
            )
        )

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    top_probability, top_index = (
        probabilities.max(
            dim=1
        )
    )

    predicted_class = categories[
        top_index.item()
    ]

    probability = (
        top_probability.item()
        * 100
    )

    # ----------------------------------------
    # Rollout
    # ----------------------------------------

    rollout = attention_rollout(
        attention_maps
    )

    rollout_grid = (
        get_cls_rollout_grid(
            rollout
        )
    )

    rollout_map = upsample_rollout(
        rollout_grid
    )

    max_row, max_col, max_score = (
        find_max_patch(
            rollout_grid
        )
    )

    print(
        f"Prediction      : "
        f"{predicted_class}"
    )

    print(
        f"Confidence      : "
        f"{probability:.2f}%"
    )

    print(
        f"Max rollout patch: "
        f"row={max_row}, "
        f"col={max_col}"
    )

    print()


    # ========================================
    # 11. Visualization
    # ========================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17, 4.5),
    )

    # ----------------------------------------
    # (1) 실제 모델 입력 crop
    # ----------------------------------------

    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        "Model input\n"
        f"GT: {gt_class}"
    )

    axes[0].axis("off")


    # ----------------------------------------
    # (2) 14x14 patch rollout
    # ----------------------------------------

    patch_plot = axes[1].imshow(
        rollout_grid,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "CLS → Patch Rollout\n"
        "14 × 14 patch grid"
    )

    axes[1].set_xlabel(
        "Patch column"
    )

    axes[1].set_ylabel(
        "Patch row"
    )

    axes[1].set_xticks(
        range(0, 14, 2)
    )

    axes[1].set_yticks(
        range(0, 14, 2)
    )

    # 최고 score patch 표시
    axes[1].scatter(
        max_col,
        max_row,
        marker="x",
        s=120,
        linewidths=2,
        c="cyan",
    )

    colorbar1 = fig.colorbar(
        patch_plot,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    colorbar1.set_label(
        "Relative rollout score\n"
        "(0 = low, 1 = high)"
    )


    # ----------------------------------------
    # (3) 224x224 rollout map
    # ----------------------------------------

    heatmap_plot = axes[2].imshow(
        rollout_map,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "Upsampled Rollout Map\n"
        "bright = higher score"
    )

    axes[2].axis("off")

    colorbar2 = fig.colorbar(
        heatmap_plot,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    colorbar2.set_label(
        "Relative rollout score"
    )


    # ----------------------------------------
    # (4) Overlay
    # ----------------------------------------

    axes[3].imshow(
        display_image
    )

    axes[3].imshow(
        rollout_map,
        cmap="magma",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Rollout Overlay\n"
        f"Pred: {predicted_class} "
        f"({probability:.2f}%)"
    )

    axes[3].axis("off")


    # ----------------------------------------
    # 전체 설명
    # ----------------------------------------

    fig.suptitle(
        "Attention Rollout Visualization\n"
        "Brighter yellow/white areas = "
        "higher cumulative CLS-to-patch rollout score",
        fontsize=13,
    )

    plt.tight_layout()


    # ========================================
    # 12. 저장
    # ========================================

    output_path = (
        OUTPUT_DIR
        / (
            "06_attention_"
            f"{Path(sample['filename']).stem}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )