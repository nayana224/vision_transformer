from pathlib import Path
import csv
import re

from torchvision.datasets import OxfordIIITPet


# ============================================
# 1. 경로 및 기본 설정
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ASSETS_DIR = SCRIPT_DIR / "assets"
MANIFEST_PATH = ASSETS_DIR / "oxford_iiit_pet_samples.csv"

SPLIT = "test"
NUM_SAMPLES = 6


# ============================================
# 2. Oxford-IIIT Pet 다운로드
# ============================================

print("Downloading / loading Oxford-IIIT Pet dataset...")

dataset = OxfordIIITPet(
    root=DATA_DIR,
    split=SPLIT,
    target_types="category",
    download=True,
)

print(f"Split          : {SPLIT}")
print(f"Dataset size   : {len(dataset)}")
print(f"Number classes : {len(dataset.classes)}")
print()


# ============================================
# 3. 서로 다른 class에서 대표 이미지 선택
# ============================================

selected = []
seen_classes = set()

for index in range(len(dataset)):
    image, class_index = dataset[index]
    class_name = dataset.classes[class_index]

    if class_name in seen_classes:
        continue

    selected.append((index, image, class_index, class_name))
    seen_classes.add(class_name)

    if len(selected) == NUM_SAMPLES:
        break

if len(selected) < NUM_SAMPLES:
    raise RuntimeError(
        f"Could only select {len(selected)} unique classes "
        f"from {len(dataset)} samples."
    )


# ============================================
# 4. 선택한 이미지를 study/assets/에 저장
# ============================================

ASSETS_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for sample_no, (index, image, class_index, class_name) in enumerate(
    selected,
    start=1,
):
    safe_class_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        class_name.strip(),
    )

    filename = f"pet_{sample_no:02d}_{safe_class_name}.jpg"
    output_path = ASSETS_DIR / filename

    image.convert("RGB").save(
        output_path,
        format="JPEG",
        quality=95,
    )

    rows.append(
        {
            "sample_no": sample_no,
            "dataset_index": index,
            "class_index": class_index,
            "class_name": class_name,
            "filename": filename,
        }
    )

    print(
        f"[{sample_no}/{NUM_SAMPLES}] "
        f"{class_name:<24} -> {output_path.name}"
    )


# ============================================
# 5. 선택 정보 manifest 저장
# ============================================

with MANIFEST_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_no",
            "dataset_index",
            "class_index",
            "class_name",
            "filename",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("Done.")
print(f"Dataset directory : {DATA_DIR}")
print(f"Sample images     : {ASSETS_DIR}")
print(f"Manifest          : {MANIFEST_PATH}")
