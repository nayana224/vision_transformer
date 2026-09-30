# Vision Transformer 논문 실습

> **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**  
> Alexey Dosovitskiy et al., ICLR 2021

Google Research의 공식 `vision_transformer` 저장소를 fork하여,
ViT 논문을 읽은 뒤 핵심 구조를 직접 코드로 확인하기 위한 개인 학습 저장소입니다.

이 저장소의 목적은 논문 성능을 장시간 학습으로 재현하는 것이 아니라,
**한 장의 이미지가 ViT 내부에서 어떤 tensor로 변환되고 순전파되는지 직접 추적하는 것**입니다.

- Original repository: https://github.com/google-research/vision_transformer
- Paper: https://arxiv.org/abs/2010.11929
- Paper notes / index: https://github.com/nayana224/dl-paper-research
- Study branch: `study/vit-paper`

## 논문 학습 기준

논문 한 편을 공부할 때 다음 내용을 내 말로 설명하고,
최소 한 번은 직접 실습하는 것을 목표로 합니다.

1. **Problem** — 기존 방법에 어떤 문제가 있었는가
2. **Core idea** — 저자들이 그 문제를 어떤 핵심 아이디어로 풀었는가
3. **Method** — 그 아이디어를 실제 모델 구조로 어떻게 구현했는가
4. **Input / GT / Output / Loss** — 학습 데이터가 모델 안에서 어떻게 흐르는가
5. **Evidence** — 실험 결과가 저자 주장을 실제로 뒷받침하는가
6. **My observation** — 직접 실습했을 때 feature나 실패 사례에서 무엇을 봤는가

현재 ViT 논문은 1~5번까지 읽고 정리했으며,
이 `study/` 실습을 통해 마지막 **My observation**을 채우는 단계입니다.

## 이번 실습에서 확인할 흐름

```text
Image
↓
Patchify
↓
Patch Embedding
↓
[CLS] Token + Position Embedding
↓
Transformer Encoder
↓
Classification Head
↓
Prediction
```

핵심은 각 단계를 직접 실행하면서 다음을 확인하는 것입니다.

- image가 실제로 몇 개의 patch로 나뉘는가
- patch 하나가 어떤 vector로 바뀌는가
- sequence length와 embedding dimension이 어떻게 변하는가
- [CLS] token과 position embedding이 어디에서 추가되는가
- Transformer Encoder 전후의 tensor shape은 어떻게 유지되는가
- pretrained model이 실제 이미지에서 어떤 prediction을 내는가
- attention이 이미지의 어느 영역에 형성되는가
- 틀린 사례에서는 어떤 패턴이 보이는가

## Study 구성

개인 실습 코드는 `study/` 아래에서 진행하며,
Google Research 원본 코드는 비교 및 참고용으로 유지합니다.

```text
study/
├── setup_venv.sh
├── 00_prepare_dataset.py
├── 01_patchify.py
├── data/
├── assets/
└── outputs/
```

실습이 진행되면서 다음 파일을 순서대로 추가할 예정입니다.

```text
00_prepare_dataset.py
01_patchify.py
02_patch_embedding.py
03_cls_position.py
04_encoder_forward.py
05_inference.py
06_attention_visualization.py
```

### 진행 상태

- [x] 00. Oxford-IIIT Pet 데이터 준비
- [ ] 01. Image Patchify
- [ ] 02. Patch Embedding
- [ ] 03. CLS Token + Position Embedding
- [ ] 04. Transformer Encoder Forward
- [ ] 05. Pretrained ViT Inference
- [ ] 06. Attention Visualization / Failure Case

## 00. Dataset Preparation

임의의 인터넷 이미지를 사용하는 대신,
논문에서도 downstream benchmark로 사용한 **Oxford-IIIT Pet** 데이터셋을 실습 이미지로 사용합니다.

`00_prepare_dataset.py`는 test split을 다운로드한 뒤
서로 다른 class에서 대표 이미지 6장을 선택합니다.

```text
Oxford-IIIT Pet
↓
test split
↓
6 representative samples
↓
study/assets/
```

전체 dataset은 `study/data/`에 저장하고 Git에서는 제외합니다.
선택한 대표 이미지는 이후 patchify, inference, attention visualization에서 다시 사용합니다.

실행:

```bash
python study/00_prepare_dataset.py
```

## 01. Patchify

첫 실습에서는 ViT-B/16을 이해하기 위한 가장 기본적인 변환부터 확인합니다.

224×224 RGB 이미지를 16×16 patch로 나누면:

```text
224 × 224 × 3
↓
16 × 16 patch
↓
14 × 14
↓
196 patches
↓
each patch: 3 × 16 × 16
↓
flatten
↓
196 × 768
```

즉 논문의

```text
N = HW / P²
```

를 실제 tensor shape으로 확인합니다.

이 단계에서는 모델 전체를 구현하지 않고,
**image가 Transformer에 들어갈 수 있는 patch sequence로 바뀌는 과정**에만 집중합니다.

## Study Environment

실습 환경은 원본 JAX/Flax 환경과 분리하여
`study/.venv/`에 별도로 생성합니다.

저장소 루트에서:

```bash
bash study/setup_venv.sh
source study/.venv/bin/activate
```

주요 패키지:

- CPU PyTorch
- torchvision
- Pillow
- Matplotlib
- Jupyter

현재 실습은 장시간 training이 아니라 forward tensor flow 확인이 목적이므로
CPU 환경을 기본으로 사용합니다.

## 원본 구현과의 관계

Google Research의 `vit_jax/`, 공식 notebook, model card 등은 가능한 한 수정하지 않습니다.

개인 실습은 PyTorch로 작고 읽기 쉬운 형태로 작성한 뒤,
필요한 단계에서 원본 JAX/Flax 구현과 대응시켜 확인합니다.

```text
study/의 최소 구현
        ↕
Google Research의 vit_jax/
```

즉 이 저장소에서는 **직접 구현해서 이해하는 것**과
**공식 구현이 실제로 어떻게 작성되어 있는지 확인하는 것**을 분리해서 진행합니다.

---

# Google Research 원본 README

아래 내용은 upstream 저장소의 안내를 보존한 것입니다.

# Vision Transformer and MLP-Mixer Architectures

In this repository we release models from the papers

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
- [When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)
- [LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/abs/2111.07991)
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065)

The models were pre-trained on the [ImageNet](http://www.image-net.org/) and
[ImageNet-21k](http://www.image-net.org/) datasets. We provide the code for
fine-tuning the released models in
[JAX](https://jax.readthedocs.io)/[Flax](http://flax.readthedocs.io).

The models from this codebase were originally trained in
https://github.com/google-research/big_vision/
where you can find more advanced code (e.g. multi-host training), as well as
some of the original training scripts (e.g.
[configs/vit_i21k.py](https://github.com/google-research/big_vision/blob/main/big_vision/configs/vit_i21k.py)
for pre-training a ViT, or
[configs/transfer.py](https://github.com/google-research/big_vision/blob/main/big_vision/configs/transfer.py)
for transfering a model).

Table of contents:

- [Vision Transformer and MLP-Mixer Architectures](#vision-transformer-and-mlp-mixer-architectures)
	- [Colab](#colab)
	- [Installation](#installation)
	- [Fine-tuning a model](#fine-tuning-a-model)
	- [Vision Transformer](#vision-transformer)
		- [Available ViT models](#available-vit-models)
		- [Expected ViT results](#expected-vit-results)
	- [MLP-Mixer](#mlp-mixer)
		- [Available Mixer models](#available-mixer-models)
		- [Expected Mixer results](#expected-mixer-results)
	- [LiT models](#lit-models)
	- [Running on cloud](#running-on-cloud)
		- [Create a VM](#create-a-vm)
		- [Setup VM](#setup-vm)
	- [Bibtex](#bibtex)
	- [Disclaimers](#disclaimers)
	- [Changelog](#changelog)


## Colab

Below Colabs run both with GPUs, and TPUs (8 cores, data parallelism).

The first Colab demonstrates the JAX code of Vision Transformers and MLP Mixers.
This Colab allows you to edit the files from the repository directly in the
Colab UI and has annotated Colab cells that walk you through the code step by
step, and lets you interact with the data.

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb

The second Colab allows you to explore the >50k Vision Transformer and hybrid
checkpoints that were used to generate the data of the third paper "How to train
your ViT? ...". The Colab includes code to explore and select checkpoints, and
to do inference both using the JAX code from this repo, and also using the
popular [`timm`] PyTorch library that can directly load these checkpoints as
well. Note that a handful of models are also available directly from TF-Hub:
[sayakpaul/collections/vision_transformer] (external contribution by [Sayak
Paul]).

The second Colab also lets you fine-tune the checkpoints on any tfds dataset
and your own dataset with examples in individual JPEG files (optionally directly
reading from Google Drive).

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb

**Note**: As for now (6/20/21) Google Colab only supports a single GPU (Nvidia
Tesla T4), and TPUs (currently TPUv2-8) are attached indirectly to the Colab VM
and communicate over slow network, which leads to pretty bad training speed. You
would usually want to set up a dedicated machine if you have a non-trivial
amount of data to fine-tune on. For details see the
[Running on cloud](#running-on-cloud) section.


[`timm`]: https://github.com/rwightman/pytorch-image-models
[sayakpaul/collections/vision_transformer]: https://tfhub.dev/sayakpaul/collections/vision_transformer
[Sayak Paul]: https://github.com/sayakpaul



## Installation

Make sure you have `Python>=3.10` installed on your machine.

Install JAX and python dependencies by running:

```
# If using GPU:
pip install -r vit_jax/requirements.txt

# If using TPU:
pip install -r vit_jax/requirements-tpu.txt
```

For newer versions of [JAX](https://github.com/google/jax), follow the instructions
provided in the corresponding repository linked here. Note that installation
instructions for CPU, GPU and TPU differs slightly.

Install [Flaxformer](https://github.com/google/flaxformer), follow the instructions
provided in the corresponding repository linked here.

For more details refer to the section [Running on cloud](#running-on-cloud)
below.


## Fine-tuning a model

You can run fine-tuning of the downloaded model on your dataset of interest. All
models share the same command line interface.

For example for fine-tuning a ViT-B/16 (pre-trained on imagenet21k) on CIFAR10
(note how we specify `b16,cifar10` as arguments to the config, and how we
instruct the code to access the models directly from a GCS bucket instead of
first downloading them into the local directory):

```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
    --config.pretrained_dir='gs://vit_models/imagenet21k'
```

In order to fine-tune a Mixer-B/16 (pre-trained on imagenet21k) on CIFAR10:

```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/mixer_base16_cifar10.py \
    --config.pretrained_dir='gs://mixer_models/imagenet21k'
```

The "How to train your ViT? ..." paper added >50k checkpoints that you can
fine-tune with the [`configs/augreg.py`] config. When you only specify the model
name (the `config.name` value from [`configs/model.py`]), then the best i21k
checkpoint by upstream validation accuracy ("recommended" checkpoint, see
section 4.5 of the paper) is chosen. To make up your mind which model you want
to use, have a look at Figure 3 in the paper. It's also possible to choose a
different checkpoint (see Colab [`vit_jax_augreg.ipynb`]) and then specify the
value from the `filename` or `adapt_filename` column, which correspond to the
filenames without `.npz` from the [`gs://vit_models/augreg`] directory.

```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/augreg.py:R_Ti_16 \
    --config.dataset=oxford_iiit_pet \
    --config.base_lr=0.01
```

Currently, the code will automatically download CIFAR-10 and CIFAR-100 datasets.
Other public or custom datasets can be easily integrated, using [tensorflow
datasets library](https://github.com/tensorflow/datasets/). Note that you will
also need to update `vit_jax/input_pipeline.py` to specify some parameters about
any added dataset.

Note that our code uses all available GPUs/TPUs for fine-tuning.

To see a detailed list of all available flags, run `python3 -m vit_jax.train
--help`.

Notes on memory:

- Different models require different amount of memory. Available memory also
  depends on the accelerator configuration (both type and count). If you
  encounter an out-of-memory error you can increase the value of
  `--config.accum_steps=8` -- alternatively, you could also decrease the
  `--config.batch=512` (and decrease `--config.base_lr` accordingly).
- The host keeps a shuffle buffer in memory. If you encounter a host OOM (as
  opposed to an accelerator OOM), you can decrease the default
  `--config.shuffle_buffer=50000`.


## Vision Transformer

by Alexey Dosovitskiy\*†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk
Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby\*†.

(\*) equal technical contribution, (†) equal advising.

![Figure 1 from paper](vit_figure.png)

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

### Available ViT models

We provide a variety of ViT models in different GCS buckets. The models can be
downloaded with e.g.:

```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

The model filenames (without the `.npz` extension) correspond to the
`config.model_name` in [`vit_jax/configs/models.py`]

- [`gs://vit_models/imagenet21k`] - Models pre-trained on ImageNet-21k.
- [`gs://vit_models/imagenet21k+imagenet2012`] - Models pre-trained on
  ImageNet-21k and fine-tuned on ImageNet.
- [`gs://vit_models/augreg`] - Models pre-trained on ImageNet-21k,
  applying varying amounts of [AugReg]. Improved performance.
- [`gs://vit_models/sam`] - Models pre-trained on ImageNet with [SAM].
- [`gs://vit_models/gsam`] - Models pre-trained on ImageNet with [GSAM].

We recommend using the following checkpoints, trained with [AugReg] that have
the best pre-training metrics:

|  Model   |                                   Pre-trained checkpoint                                   |   Size   |                                                       Fine-tuned checkpoint                                                        | Resolution | Img/sec | Imagenet accuracy |
| :------- | :----------------------------------------------------------------------------------------- | -------: | :--------------------------------------------------------------------------------------------------------------------------------- | ---------: | ------: | ----------------: |
| L/16     | `gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz`     | 1243 MiB | `gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`     |        384 |      50 |            85.59% |
| B/16     | `gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz`     |  391 MiB | `gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`     |        384 |     138 |            85.49% |
| S/16     | `gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz`     |  115 MiB | `gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`     |        384 |     300 |            83.73% |
| R50+L/32 | `gs://vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz` | 1337 MiB | `gs://vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz` |        384 |     327 |            85.99% |
| R26+S/32 | `gs://vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz`  |  170 MiB | `gs://vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`  |        384 |     560 |            83.85% |
| Ti/16    | `gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz`      |   37 MiB | `gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`      |        384 |     610 |            78.22% |
| B/32     | `gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz`      |  398 MiB | `gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`      |        384 |     955 |            83.59% |
| S/32     | `gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz`        |  118 MiB | `gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`        |        384 |    2154 |            79.58% |
| R+Ti/16  | `gs://vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz`    |   40 MiB | `gs://vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz`    |        384 |    2426 |            75.40% |

The results from the original ViT paper (https://arxiv.org/abs/2010.11929) have
been replicated using the models from [`gs://vit_models/imagenet21k`]:

| model        | dataset      | dropout=0.0                                                                                                                                                         | dropout=0.1                                                                                                                                                          |
|:-------------|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R50+ViT-B_16 | cifar10      | 98.72%, 3.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar10/do_0.0&_smoothingWeight=0)      | 98.94%, 10.1h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar10/do_0.1&_smoothingWeight=0)      |
| R50+ViT-B_16 | cifar100     | 90.88%, 4.1h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar100/do_0.0&_smoothingWeight=0)     | 92.30%, 10.1h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/cifar100/do_0.1&_smoothingWeight=0)     |
| R50+ViT-B_16 | imagenet2012 | 83.72%, 9.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/imagenet2012/do_0.0&_smoothingWeight=0) | 85.08%, 24.2h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5ER50.ViT-B_16/imagenet2012/do_0.1&_smoothingWeight=0) |
| ViT-B_16     | cifar10      | 99.02%, 2.2h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar10/do_0.0&_smoothingWeight=0)          | 98.76%, 7.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar10/do_0.1&_smoothingWeight=0)           |
| ViT-B_16     | cifar100     | 92.06%, 2.2h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar100/do_0.0&_smoothingWeight=0)         | 91.92%, 7.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/cifar100/do_0.1&_smoothingWeight=0)          |
| ViT-B_16     | imagenet2012 | 84.53%, 6.5h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/imagenet2012/do_0.0&_smoothingWeight=0)     | 84.12%, 19.3h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_16/imagenet2012/do_0.1&_smoothingWeight=0)     |
| ViT-B_32     | cifar10      | 98.88%, 0.8h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar10/do_0.0&_smoothingWeight=0)          | 98.75%, 1.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar10/do_0.1&_smoothingWeight=0)           |
| ViT-B_32     | cifar100     | 92.31%, 0.8h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar100/do_0.0&_smoothingWeight=0)         | 92.05%, 1.8h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/cifar100/do_0.1&_smoothingWeight=0)          |
| ViT-B_32     | imagenet2012 | 81.66%, 3.3h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/imagenet2012/do_0.0&_smoothingWeight=0)     | 81.31%, 4.9h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-B_32/imagenet2012/do_0.1&_smoothingWeight=0)      |
| ViT-L_16     | cifar10      | 99.13%, 6.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar10/do_0.0&_smoothingWeight=0)          | 99.14%, 24.7h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar10/do_0.1&_smoothingWeight=0)          |
| ViT-L_16     | cifar100     | 92.91%, 7.1h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar100/do_0.0&_smoothingWeight=0)         | 93.22%, 24.4h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/cifar100/do_0.1&_smoothingWeight=0)         |
| ViT-L_16     | imagenet2012 | 84.47%, 16.8h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/imagenet2012/do_0.0&_smoothingWeight=0)    | 85.05%, 59.7h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_16/imagenet2012/do_0.1&_smoothingWeight=0)     |
| ViT-L_32     | cifar10      | 99.06%, 1.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar10/do_0.0&_smoothingWeight=0)          | 99.09%, 6.1h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar10/do_0.1&_smoothingWeight=0)           |
| ViT-L_32     | cifar100     | 93.29%, 1.9h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar100/do_0.0&_smoothingWeight=0)         | 93.34%, 6.2h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/cifar100/do_0.1&_smoothingWeight=0)          |
| ViT-L_32     | imagenet2012 | 81.89%, 7.5h (A100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/imagenet2012/do_0.0&_smoothingWeight=0)     | 81.13%, 15.0h (V100), [tb.dev](https://tensorboard.dev/experiment/nwXQNjudRJW3dtQzhPZwwA/#scalars&regexInput=%5EViT-L_32/imagenet2012/do_0.1&_smoothingWeight=0)     |

We also would like to emphasize that high-quality results can be achieved with
shorter training schedules and encourage users of our code to play with
hyper-parameters to trade-off accuracy and computational budget.
Some examples for CIFAR-10/100 datasets are presented in the table below.

| upstream    | model    | dataset      | total_steps / warmup_steps  | accuracy | wall-clock time |                                                                         link |
| ----------- | -------- | ------------ | --------------------------- | -------- | --------------- | ---------------------------------------------------------------------------- |
| imagenet21k | ViT-B_16 | cifar10      | 500 / 50                    |   98.59% |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/QgkpiW53RPmjkabe1ME31g/) |
| imagenet21k | ViT-B_16 | cifar10      | 1000 / 100                  |   98.86% |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/w8DQkDeJTOqJW5js80gOQg/) |
| imagenet21k | ViT-B_16 | cifar100     | 500 / 50                    |   89.17% |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/5hM4GrnAR0KEZg725Ewnqg/) |
| imagenet21k | ViT-B_16 | cifar100     | 1000 / 100                  |   91.15% |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/QLQTaaIoT9uEcAjtA0eRwg/) |


## MLP-Mixer

by Ilya Tolstikhin\*, Neil Houlsby\*, Alexander Kolesnikov\*, Lucas Beyer\*,
Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers,
Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy.

(\*) equal contribution.

![Figure 1 from paper](mixer_figure.png)

MLP-Mixer (*Mixer* for short) consists of per-patch linear embeddings, Mixer
layers, and a classifier head. Mixer layers contain one token-mixing MLP and one
channel-mixing MLP, each consisting of two fully-connected layers and a GELU
nonlinearity. Other components include: skip-connections, dropout, and linear
classifier head.

For installation follow [the same steps](#installation) as above.

### Available Mixer models

We provide the Mixer-B/16 and Mixer-L/16 models pre-trained on the ImageNet and
ImageNet-21k datasets. Details can be found in Table 3 of the Mixer paper. All
the models can be found at:

https://console.cloud.google.com/storage/mixer_models/

Note that these models are also available directly from TF-Hub:
[sayakpaul/collections/mlp-mixer] (external contribution by [Sayak
Paul]).

[sayakpaul/collections/mlp-mixer]: https://tfhub.dev/sayakpaul/collections/mlp-mixer

### Expected Mixer results

We ran the fine-tuning code on Google Cloud machine with four V100 GPUs with the
default adaption parameters from this repository. Here are the results:

upstream     | model      | dataset | accuracy | wall_clock_time | link
:----------- | :--------- | :------ | -------: | :-------------- | :---
ImageNet     | Mixer-B/16 | cifar10 | 96.72%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/j9zCYt9yQVm93nqnsDZayA/)
ImageNet     | Mixer-L/16 | cifar10 | 96.59%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/Q4feeErzRGGop5XzAvYj2g/)
ImageNet-21k | Mixer-B/16 | cifar10 | 96.82%   | 9.6h            | [tensorboard.dev](https://tensorboard.dev/experiment/mvP4McV2SEGFeIww20ie5Q/)
ImageNet-21k | Mixer-L/16 | cifar10 | 98.34%   | 10.0h           | [tensorboard.dev](https://tensorboard.dev/experiment/dolAJyQYTYmudytjalF6Jg/)


## LiT models

For details, refer to the Google AI blog post
[LiT: adding language understanding to image models](http://ai.googleblog.com/2022/04/locked-image-tuning-adding-language.html),
or read the CVPR paper "LiT: Zero-Shot Transfer with Locked-image text Tuning"
(https://arxiv.org/abs/2111.07991).

We published a Transformer B/16-base model with an ImageNet zeroshot accuracy of
72.1%, and a L/16-large model with an ImageNet zeroshot accuracy of 75.7%. For
more details about these models, please refer to the
[LiT model card](model_cards/lit.md).

We provide a in-browser demo with small text encoders for interactive use (the
smallest models should even run on a modern cell phone):

https://google-research.github.io/vision_transformer/lit/

And finally a Colab to use the JAX models with both image and text encoders:

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

Note that none of above models support multi-lingual inputs yet, but we're
working on publishing such models and will update this repository once they
become available.

This repository only contains evaluation code for LiT models. You can find the
training code in the `big_vision` repository:

https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text

Expected zeroshot results from [`model_cards/lit.md`] (note that the zeroshot
evaluation is slightly different from the simplified evaluation in the Colab):

| Model | B16B_2 | L16L |
| :--- | ---: | ---: |
| ImageNet zero-shot | 73.9% | 75.7% |
| ImageNet v2 zero-shot | 65.1% | 66.6% |
| CIFAR100 zero-shot | 79.0% | 80.5% |
| Pets37 zero-shot | 83.3% | 83.3% |
| Resisc45 zero-shot | 25.3% | 25.6% |
| MS-COCO Captions image-to-text retrieval | 51.6% | 48.5% |
| MS-COCO Captions text-to-image retrieval | 31.8% | 31.1% |

## Running on cloud

While above [colabs](#colab) are pretty useful to get started, you would usually
want to train on a larger machine with more powerful accelerators.

### Create a VM

You can use the following commands to setup a VM with GPUs on Google Cloud:

```bash
# Set variables used by all commands below.
# Note that project must have accounting set up.
# For a list of zones with GPUs refer to
# https://cloud.google.com/compute/docs/gpus/gpu-regions-zones
PROJECT=my-awesome-gcp-project  # Project must have billing enabled.
VM_NAME=vit-jax-vm-gpu
ZONE=europe-west4-b

# Below settings have been tested with this repository. You can choose other
# combinations of images & machines (e.g.), refer to the corresponding gcloud commands:
# gcloud compute images list --project ml-images
# gcloud compute machine-types list
# etc.
gcloud compute instances create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --image=c1-deeplearning-tf-2-5-cu110-v20210527-debian-10 \
    --image-project=ml-images --machine-type=n1-standard-96 \
    --scopes=cloud-platform,storage-full --boot-disk-size=256GB \
    --boot-disk-type=pd-ssd --metadata=install-nvidia-driver=True \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-tesla-v100,count=8

# Connect to VM (after some minutes needed to setup & start the machine).
gcloud compute ssh --project $PROJECT --zone $ZONE $VM_NAME

# Stop the VM after use (only storage is billed for a stopped VM).
gcloud compute instances stop --project $PROJECT --zone $ZONE $VM_NAME

# Delete VM after use (this will also remove all data stored on VM).
gcloud compute instances delete --project $PROJECT --zone $ZONE $VM_NAME
```

Alternatively, you can use the following similar commands to set up a Cloud VM
with TPUs attached to them (below commands copied from the [TPU tutorial]):

[TPU tutorial]: https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm

```bash
PROJECT=my-awesome-gcp-project  # Project must have billing enabled.
VM_NAME=vit-jax-vm-tpu
ZONE=europe-west4-a

# Required to set up service identity initially.
gcloud beta services identity create --service tpu.googleapis.com

# Create a VM with TPUs directly attached to it.
gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --accelerator-type v3-8 \
    --version tpu-vm-base

# Connect to VM (after some minutes needed to setup & start the machine).
gcloud alpha compute tpus tpu-vm ssh --project $PROJECT --zone $ZONE $VM_NAME

# Stop the VM after use (only storage is billed for a stopped VM).
gcloud alpha compute tpus tpu-vm stop --project $PROJECT --zone $ZONE $VM_NAME

# Delete VM after use (this will also remove all data stored on VM).
gcloud alpha compute tpus tpu-vm delete --project $PROJECT --zone $ZONE $VM_NAME
```

### Setup VM

And then fetch the repository and the install dependencies (including `jaxlib`
with TPU support) as usual:

```bash
git clone --depth=1 --branch=master https://github.com/google-research/vision_transformer
cd vision_transformer

# optional: install virtualenv
pip3 install virtualenv
python3 -m virtualenv env
. env/bin/activate
```

If you're connected to a VM with GPUs attached, install JAX and other dependencies with the following
command:

```bash
pip install -r vit_jax/requirements.txt
```

If you're connected to a VM with TPUs attached, install JAX and other dependencies with the following
command:

```bash
pip install -r vit_jax/requirements-tpu.txt
```

Install [Flaxformer](https://github.com/google/flaxformer), follow the instructions
provided in the corresponding repository linked here.

For both GPUs and TPUs, Check that JAX can connect to attached accelerators with the command:
```bash
python -c 'import jax; print(jax.devices())'
```

And finally execute one of the commands mentioned in the section
[fine-tuning a model](#fine-tuning-a-model).


## Bibtex

```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}

@article{tolstikhin2021mixer,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

@article{steiner2021augreg,
  title={How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers},
  author={Steiner, Andreas and Kolesnikov, Alexander and and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.10270},
  year={2021}
}

@article{chen2021outperform,
  title={When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations},
  author={Chen, Xiangning and Hsieh, Cho-Jui and Gong, Boqing},
  journal={arXiv preprint arXiv:2106.01548},
  year={2021},
}

@article{zhuang2022gsam,
  title={Surrogate Gap Minimization Improves Sharpness-Aware Training},
  author={Zhuang, Juntang and Gong, Boqing and Yuan, Liangzhe and Cui, Yin and Adam, Hartwig and Dvornek, Nicha and Tatikonda, Sekhar and Duncan, James and Liu, Ting},
  journal={ICLR},
  year={2022},
}

@article{zhai2022lit,
  title={LiT: Zero-Shot Transfer with Locked-image Text Tuning},
  author={Zhai, Xiaohua and Wang, Xiao and Mustafa, Basil and Steiner, Andreas and Keysers, Daniel and Kolesnikov, Alexander and Beyer, Lucas},
  journal={CVPR},
  year={2022}
}
```


## Changelog

In reverse chronological order:

- 2022-08-18: Added LiT-B16B_2 model that was trained for 60k steps
  (LiT_B16B: 30k) without linear head on the image side (LiT_B16B: 768) and has
  better performance.

- 2022-06-09: Added the ViT and Mixer models trained from scratch using
  [GSAM] on ImageNet without strong data augmentations. The resultant ViTs
  outperform those of similar sizes trained using AdamW optimizer or the
  original [SAM] algorithm, or with strong data augmentations.

- 2022-04-14: Added models and Colab for [LiT models](#lit-models).

- 2021-07-29: Added ViT-B/8 AugReg models (3 upstream checkpoints and adaptations
  with resolution=224).

- 2021-07-02: Added the "When Vision Transformers Outperform
  ResNets..." paper

- 2021-07-02: Added [SAM](https://arxiv.org/abs/2010.01412)
  (Sharpness-Aware Minimization) optimized ViT and MLP-Mixer checkpoints.

- 2021-06-20: Added the "How to train your ViT? ..." paper, and a new
  Colab to explore the >50k pre-trained and fine-tuned checkpoints mentioned in
  the paper.

- 2021-06-18: This repository was rewritten to use Flax Linen API and
  `ml_collections.ConfigDict` for configuration.

- 2021-05-19: With publication of the "How to train your ViT? ..."
  paper, we added more than 50k ViT and hybrid models pre-trained on ImageNet and
  ImageNet-21k with various degrees of data augmentation and model regularization,
  and fine-tuned on ImageNet, Pets37, Kitti-distance, CIFAR-100, and Resisc45.
  Check out [`vit_jax_augreg.ipynb`] to navigate this treasure trove of models!
  For example, you can use that Colab to fetch the filenames of recommended
  pre-trained and fine-tuned checkpoints from the `i21k_300` column of Table 3 in
  the paper.

- 2020-12-01: Added the R50+ViT-B/16 hybrid model (ViT-B/16 on
  top of a Resnet-50 backbone). When pretrained on imagenet21k, this model
  achieves almost the performance of the L/16 model with less than half the
  computational finetuning cost. Note that "R50" is somewhat modified for the
  B/16 variant: The original ResNet-50 has [3,4,6,3] blocks, each reducing the
  resolution of the image by a factor of two. In combination with the ResNet
  stem this would result in a reduction of 32x so even with a patch size of
  (1,1) the ViT-B/16 variant cannot be realized anymore. For this reason we
  instead use [3,4,9] blocks for the R50+B/16 variant.

- 2020-11-09: Added the ViT-L/16 model.

- 2020-10-29: Added ViT-B/16 and ViT-L/16 models pretrained
  on ImageNet-21k and then fine-tuned on ImageNet at 224x224 resolution (instead
  of default 384x384). These models have the suffix "-224" in their name.
  They are expected to achieve 81.2% and 82.7% top-1 accuracies respectively.


## Disclaimers

Open source release prepared by Andreas Steiner.

Note: This repository was forked and modified from
[google-research/big_transfer](https://github.com/google-research/big_transfer).

**This is not an official Google product.**


[GSAM]: https://arxiv.org/abs/2203.08065
[SAM]: https://arxiv.org/abs/2010.01412
[AugReg]: https://arxiv.org/abs/2106.10270

[`vit_jax/configs/models.py`]: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
[`model_cards/lit.md`]: https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md

[`configs/augreg.py`]: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/augreg.py
[`configs/model.py`]: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
[`vit_jax_augreg.ipynb`]: https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb
[`vit_jax.ipynb`]: https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb

[`gs://vit_models/imagenet21k`]: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k/
[`gs://vit_models/imagenet21k+imagenet2012`]: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k+imagenet2012/
[`gs://vit_models/augreg`]: https://console.cloud.google.com/storage/browser/vit_models/augreg/
[`gs://vit_models/sam`]: https://console.cloud.google.com/storage/browser/vit_models/sam/
[`gs://mixer_models/sam`]: https://console.cloud.google.com/storage/mixer_models/sam/
[`gs://vit_models/gsam`]: https://console.cloud.google.com/storage/browser/vit_models/gsam/
[`gs://mixer_models/gsam`]: https://console.cloud.google.com/storage/mixer_models/gsam/
