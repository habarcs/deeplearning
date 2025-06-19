#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLIP + CoOp implementation for Oxford Flowers 102 dataset

This script implements CLIP with Context Optimization (CoOp) for few-shot learning
on Oxford Flowers 102 dataset. The goal is to adapt the model to base categories
while preserving or improving zero-shot performance on novel categories.

Based on:
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- CoOp: Learning to Prompt for Vision-Language Models
"""

# %% [markdown]
# # CLIP + CoOp Implementation for Oxford Flowers 102
#
# This notebook implements few-shot adaptation using CLIP with Context Optimization (CoOp).
# We'll evaluate our method on base and novel categories of the Oxford Flowers 102 dataset.

# %% [markdown]
# ## Setup and Imports
#
# Let's start by installing the necessary packages and importing libraries.

# %%
# Install dependencies
# Run this in your environment:
# !pip install torch==2.6.0 torchvision==0.21.0 wandb==0.19.10 scipy==1.15.3 segmentation-models-pytorch==0.5.0 scikit-learn scikit-image ftfy
# !pip install git+https://github.com/openai/CLIP.git

# %%
import json
import logging
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import PIL.Image
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import transforms
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as TF
from tqdm import tqdm

import clip

import wandb

# Configuration following deep_learning.py pattern
CFG = {
    "COOP": {
        "base_model": {
            "name": "ViT-B/16",  # Using ViT-B/16 as specified in requirements
            "weights": "openai",  # OpenAI pretrained weights (not used with original CLIP)
        },
        "prompt_learner": {
            "n_ctx": 16,  # Number of context tokens
            "ctx_init": "",  # Context initialization (empty for random)
            "class_token_position": "end",  # Position of class token
            "csc": False,  # Class-specific context
        },
    },
    "wandb": True,
    "training": {
        "resume_id": None,
        "resume_from": "best",
        "batch_size": 32,
        "test_batch_size": 64,
        "epochs": 100,
        "patience": 2,  # Early stopping patience
        "shots_per_class": 10,  # k=10 for Flowers 102
        "checkpoint_dir": "./checkpoints",
        "optimizer": {
            "lr": 0.002,
            "weight_decay": 0.0001,
            "momentum": 0.9,
        },
        "scheduler": {
            "type": "cosine",
            "eta_min": 1e-5,
        },
        "augmentation_mode": None,  # choose from: None, "rotate_illumination", "rotate_contrast", "rotate_contrast_illumination"
        "seed": 47,
    },
    "input": {"size": [224, 224]},
    "data": {
        "data_dir": "./data",
        "num_workers": 4,
        "pin_memory": True,
    },
    "validation": {
        "evaluate_zero_shot": False,  # Evaluate CLIP zero-shot as baseline
        "ema": {
            "enabled": False,  # this doesn't need to be disabled
            "decay": 0.9,  # EMA decay factor
            "start_epoch": 5,  # ema starting epoch, before this weights are not updated
        },
    },
    "helip": {
        "enabled": False,
        "k": 3,  # Number of hard pairs to mine per sample
        "p": 20,  # Number of random pairs to sample
        "alpha": 0.25,  # Margin parameter for HNML
        "lambda_hnml": 0.5,  # Weight for HNML in the combined loss
        "mining_freq": 5,  # Mine hard pairs every N epochs
        "cache_embeddings": True,  # Whether to cache embeddings between epochs
    },
    "promptsrc": {
        "enabled": False,
        "mutual_agreement": {
            "lambda_ma": 0.5,  # Weight for mutual agreement loss
            "temperature": 0.07,  # Temperature scaling parameter
        },
        "prompt_ensemble": {
            "enabled": True,
            "window_size": 5,  # Number of recent prompts to ensemble
            "sigma": 2.0,  # Sigma for Gaussian weighting
        },
        "textual_diversity": {
            "lambda_td": 0.3,  # Weight for textual diversity loss
            "temperature": 0.07,  # Temperature for text similarity
        },
    },
    "bg_masking": {
        "enabled": False,
        "backbone": "unet",  # Segmentation model backbone
        "encoder": {
            "name": "mobilenet_v2",  # Encoder for segmentation model
            "weights": "imagenet",  # Pretrained weights
        },
        "training": {
            "epochs": 3,  # Epochs for segmentation model training
            "batch_size": 64,
            "image_size": 224,
        },
    },
}

# Global variables following deep_learning.py pattern
RUN_ID = (
    CFG["training"]["resume_id"]
    if CFG["training"]["resume_id"]
    else datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
)
RUN_DIR = f"out/{RUN_ID}/"
os.makedirs(RUN_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(RUN_DIR + "training.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("coop_training")

# Initialize wandb
WANDB = (
    wandb.init(
        project="Deep learning project",
        config=CFG,
        name="CoOp_" + RUN_ID,
        tags=["coop", CFG["COOP"]["base_model"]["name"]],
    )
    if CFG["wandb"]
    else wandb.init(mode="disabled")
)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using {DEVICE} device")


def set_seed(seed=47):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG["training"]["seed"])


class Flowers102WithSegmentation(Flowers102):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        download: bool = False,
        segmentation_training: bool = False,
    ) -> None:
        self.segmentation_training = segmentation_training
        self._file_dict["seg"] = (
            "102segmentations.tgz",
            "51375996b6c4f367a4b9f4492d7ecef5",
        )
        self.mask_transform = mask_transform
        super().__init__(root, split, transform, target_transform, download)
        self._seg_folder = self._base_folder / "segmim"
        self._image_segs = []
        for image_file in self._image_files:
            self._image_segs.append(
                self._seg_folder / re.sub(r"image_", "segmim_", image_file.name)
            )

    def get_bg_mask(self, idx: int) -> Any:
        """Get background mask for image at index idx"""
        seg_file = self._image_segs[idx]
        seg_image = PIL.Image.open(seg_file).convert("RGB")
        seg = pil_to_tensor(seg_image)
        # Inverting the bg mask, to get 0 for background
        # 0 0 254 blue represents the background in the segmentation
        mask = ~((seg[0] == 0) & (seg[1] == 0) & (seg[2] == 254))
        mask = mask.int()
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return mask

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image, label = super().__getitem__(idx)
        if self.segmentation_training:
            mask = self.get_bg_mask(idx)
            return image, mask
        return image, label

    def download(self):
        super().download()
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['seg'][0]}",
            str(self._base_folder),
            md5=self._file_dict["seg"][1],
        )


# Class names for Oxford Flowers 102
CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

# %% [markdown]
# ## Dataset Loading and Processing
#
# We'll load the Oxford Flowers 102 dataset and implement functions for splitting
# into base/novel categories and creating few-shot datasets.

# %%
# Helper functions for dataset loading and processing


# Load Oxford Flowers 102 dataset with transforms and optional segmentation
def get_data(
    data_dir="./data",
    train_transform=None,
    test_transform=None,
    mask_transform=None,
    segmentation_training=False,
):
    LOGGER.info(
        f"Loading Flowers102 dataset from {data_dir} {'for segmentation training' if segmentation_training else ''}"
    )

    try:
        train_set = Flowers102WithSegmentation(
            root=data_dir,
            split="train",
            download=True,
            transform=train_transform,
            mask_transform=mask_transform,
            segmentation_training=segmentation_training,
        )
        val_set = Flowers102WithSegmentation(
            root=data_dir,
            split="val",
            download=True,
            transform=test_transform,
            mask_transform=mask_transform,
            segmentation_training=segmentation_training,
        )
        test_set = Flowers102WithSegmentation(
            root=data_dir,
            split="test",
            download=True,
            transform=test_transform,
            mask_transform=mask_transform,
            segmentation_training=segmentation_training,
        )

        LOGGER.info(
            f"Dataset loaded: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test"
        )
        return train_set, val_set, test_set

    except Exception as e:
        LOGGER.error(f"Error loading dataset: {e}")
        raise


# Split dataset classes into base and novel categories (50/50 split)
def base_novel_categories(dataset):
    all_classes = (
        set(dataset._labels) if hasattr(dataset, "_labels") else set(range(102))
    )
    num_classes = len(all_classes)

    # Split classes in half: first 51 as base, last 51 as novel
    base_classes = list(range(num_classes // 2))
    novel_classes = list(range(num_classes // 2, num_classes))

    LOGGER.info(
        f"Split {num_classes} classes: {len(base_classes)} base, {len(novel_classes)} novel"
    )
    return base_classes, novel_classes


# Split dataset into base and novel
def split_data(dataset, base_classes):
    base_indices = []
    novel_indices = []
    base_set = set(base_classes)

    # Iterate through dataset to find indices for each category
    for idx in range(len(dataset)):
        if hasattr(dataset, "_labels"):
            label = dataset._labels[idx]
        else:
            # For Subset objects, we need to get the original label
            if hasattr(dataset, "dataset"):
                original_idx = dataset.indices[idx]
                label = dataset.dataset._labels[original_idx]
            else:
                _, label = dataset[idx]

        if label in base_set:
            base_indices.append(idx)
        else:
            novel_indices.append(idx)

    base_dataset = Subset(dataset, base_indices)
    novel_dataset = Subset(dataset, novel_indices)

    LOGGER.info(
        f"Split dataset: {len(base_dataset)} base samples, {len(novel_dataset)} novel samples"
    )
    return base_dataset, novel_dataset


# Create few-shot dataset with limited samples per class
def create_few_shot_dataset(dataset, shots_per_class=10):
    if shots_per_class <= 0:
        return dataset

    class_indices = {}

    for idx in range(len(dataset)):
        if hasattr(dataset, "dataset"):
            # Handle Subset objects
            original_idx = dataset.indices[idx]
            label = dataset.dataset._labels[original_idx]
        else:
            # Handle direct dataset objects
            if hasattr(dataset, "_labels"):
                label = dataset._labels[idx]
            else:
                _, label = dataset[idx]

        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Sample few-shot indices
    few_shot_indices = []
    for class_label, indices in class_indices.items():
        # Randomly sample up to shots_per_class indices
        sampled_indices = random.sample(indices, min(shots_per_class, len(indices)))
        few_shot_indices.extend(sampled_indices)

    few_shot_dataset = Subset(dataset, few_shot_indices)

    LOGGER.info(
        f"Created {shots_per_class}-shot dataset: {len(few_shot_dataset)} samples"
    )
    return few_shot_dataset


# Create data loaders for training and evaluation
def create_data_loaders(
    train_set,
    val_set,
    test_base,
    test_novel,
    batch_size=32,
    test_batch_size=64,
    num_workers=4,
    pin_memory=True,
):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_base_loader = DataLoader(
        test_base,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_novel_loader = DataLoader(
        test_novel,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    LOGGER.info(
        f"Created data loaders with batch_size={batch_size}, test_batch_size={test_batch_size}"
    )
    return train_loader, val_loader, test_base_loader, test_novel_loader


# %% [markdown]
# ## CLIP Model and Zero-shot Evaluation
#
# First, we'll load the CLIP model and implement zero-shot evaluation.


# %%
# Load CLIP model and preprocessing transforms using original OpenAI CLIP
def load_clip_model(model_name="ViT-B/16", pretrained=None):
    LOGGER.info(f"Loading CLIP model: {model_name}")

    try:
        model, preprocess = clip.load(model_name, device=DEVICE)
        model = model.float()
        model.eval()

        LOGGER.info(f"Model loaded successfully: {model_name}")
        return model, preprocess

    except Exception as e:
        LOGGER.error(f"Error loading CLIP model: {e}")
        raise


# Evaluate CLIP'S zero-shot performance
def zero_shot_evaluation(
    model, data_loader, class_names, class_indices=None, text_templates=None
):
    if text_templates is None:
        text_templates = [
            "a photo of a {}.",
            "a picture of a {}.",
            "this is a photo of a {}.",
            "this is a picture of a {}.",
            "a close-up photo of a {}.",
        ]

    LOGGER.info(f"Zero-shot evaluation on {len(data_loader.dataset)} samples")

    # Create text features for all classes
    all_texts = []
    for class_name in class_names:
        class_texts = [template.format(class_name) for template in text_templates]
        all_texts.extend(class_texts)

    text_tokens = clip.tokenize(all_texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Average features across templates for each class
        text_features = text_features.view(len(class_names), len(text_templates), -1)
        text_features = text_features.mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Create mapping from original indices to evaluation indices if needed
    if class_indices is not None:
        index_mapping = {
            orig_idx: eval_idx for eval_idx, orig_idx in enumerate(class_indices)
        }
    else:
        index_mapping = None

    # Evaluate
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Zero-shot evaluation"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Map targets to evaluation indices if needed
            if index_mapping is not None:
                mapped_targets = torch.tensor(
                    [index_mapping[t.item()] for t in targets], device=DEVICE
                )
            else:
                mapped_targets = targets

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarities
            logits = (image_features @ text_features.T) * model.logit_scale.exp()
            predictions = logits.argmax(dim=-1)

            correct += (predictions == mapped_targets).sum().item()
            total += mapped_targets.size(0)

    accuracy = (correct / total) * 100
    LOGGER.info(f"Zero-shot accuracy: {accuracy:.2f}% ({correct}/{total})")

    return accuracy


# %% [markdown]
# ## CoOp Implementation
#
# Now let's implement the Context Optimization (CoOp) method for few-shot learning.


# %%
# CoOp T.E. using original OpenAI CLIP
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


# Prompt learner for CoOp using original OpenAI CLIP
class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        classnames,
        n_ctx=4,
        ctx_init="",
        class_token_position="end",
        csc=False,
    ):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224

        # Initialize context vectors
        if ctx_init:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(DEVICE)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(clip_model.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            if csc:
                LOGGER.info("Initializing class-specific contexts")
                ctx_vectors = torch.empty(
                    n_cls, n_ctx, ctx_dim, dtype=clip_model.dtype, device=DEVICE
                )
            else:
                LOGGER.info("Initializing a generic context")
                ctx_vectors = torch.empty(
                    n_ctx, ctx_dim, dtype=clip_model.dtype, device=DEVICE
                )

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        LOGGER.info(f'Initial context: "{prompt_prefix}"')
        LOGGER.info(f"Number of context words (tokens): {n_ctx}")

        # learnable context parameters
        self.ctx = nn.Parameter(ctx_vectors)

        # Prepare class names and prompts
        classnames = [name.replace("_", " ") for name in classnames]
        simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        name_lens = [len(simple_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(DEVICE)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                clip_model.dtype
            )

        # Register buffers for prefix and suffix tokens
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    # Generate prompts with learnable context
    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(
                f"Unknown class token position: {self.class_token_position}"
            )

        return prompts


# CoOp model: CLIP with learnable prompts, optional background masking, and PromptSRC support
class CoOp(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx=4,
        ctx_init="",
        class_token_position="end",
        csc=False,
        segmentation_model=None,
        segmentation_transform=None,
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx, ctx_init, class_token_position, csc
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.classnames = classnames

        # Storage for features (needed for PromptSRC)
        self.current_image_features = None
        self.current_text_features = None

        # Background masking components
        self.segmentation_model = segmentation_model
        self.segmentation_transform = segmentation_transform

        # Add prompt ensemble
        if (
            CFG["promptsrc"]["enabled"]
            and CFG["promptsrc"]["prompt_ensemble"]["enabled"]
        ):
            window_size = CFG["promptsrc"]["prompt_ensemble"]["window_size"]
            sigma = CFG["promptsrc"]["prompt_ensemble"]["sigma"]
            self.prompt_ensemble = PromptEnsemble(window_size=window_size, sigma=sigma)
            LOGGER.info(
                f"PromptSRC ensemble enabled: window_size={window_size}, sigma={sigma}"
            )
        else:
            self.prompt_ensemble = None

        # Freeze CLIP weights
        for param in self.image_encoder.parameters():
            param.requires_grad_(False)
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)

    def apply_background_masking_if_needed(self, images):
        if self.segmentation_model:
            if self.segmentation_transform:
                seg_input = self.segmentation_transform(images)
            else:
                seg_input = images
            mask = self.segmentation_model(seg_input)
            binary_mask = (mask > 0.5).int()
            return binary_mask * images
        return images

    def forward(self, image):
        image = self.apply_background_masking_if_needed(image)

        # Extract image features
        image_features = self.image_encoder(image)
        self.current_image_features = image_features

        # prompts and extract text features
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        self.current_text_features = text_features

        if self.prompt_ensemble is not None and self.training:
            # Update the ensemble with current prompt
            self.prompt_ensemble.update(self.prompt_learner.ctx.data)

        # Feature normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# %% [markdown]
# ## Training and Evaluation Functions
#
# Functions for training the CoOp model and evaluating its performance.


# %%
def train_coop(
    model,
    train_loader,
    optimizer,
    criterion,
    epoch,
    avg_model=None,
    hard_pair_miner=None,
    use_helip=False,
    base_model=None,
    use_promptsrc=False,
):
    """Train CoOp model for one epoch with EMA, HELIP, and PromptSRC support"""
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_hnml_loss = 0.0
    total_ma_loss = 0.0
    total_td_loss = 0.0
    correct = 0
    total = 0

    # Get PromptSRC parameters
    if use_promptsrc:
        lambda_ma, lambda_td = get_promptsrc_weights()
        temperature = CFG["promptsrc"]["mutual_agreement"]["temperature"]
        LOGGER.info(
            f"PromptSRC enabled: lambda_ma={lambda_ma:.4f}, lambda_td={lambda_td:.4f}"
        )
    else:
        lambda_ma, lambda_td = 0.0, 0.0
        temperature = 0.07

    # Initialize hard pairs for HELIP
    hard_pairs = {}
    if use_helip and hard_pair_miner is not None:
        dataset_indices = (
            train_loader.dataset.indices
            if hasattr(train_loader.dataset, "indices")
            else list(range(len(train_loader.dataset)))
        )
        # Mine hard pairs for this epoch
        hard_pairs = hard_pair_miner.mine_hard_pairs(dataset_indices, epoch)
        LOGGER.info(f"HELIP enabled: mined {len(hard_pairs)} hard pair sets")

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        try:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            with torch.amp.autocast("cuda"):
                logits = model(images)
                contrastive_loss = criterion(logits, targets)

                # Additional losses
                hnml_loss = torch.tensor(0.0, device=DEVICE)
                ma_loss = torch.tensor(0.0, device=DEVICE)
                td_loss = torch.tensor(0.0, device=DEVICE)

                # Apply PromptSRC
                if (
                    use_promptsrc
                    and base_model is not None
                    and model.current_image_features is not None
                    and model.current_text_features is not None
                ):
                    batch_size = images.size(0)

                    # Get original model features
                    with torch.no_grad():
                        # Frozen image features
                        original_image_features = base_model.encode_image(images)
                        original_image_features = (
                            original_image_features
                            / original_image_features.norm(dim=-1, keepdim=True)
                        )

                        # Frozen text features - create text templates for each class in the batch
                        text_inputs = [
                            f"a photo of a {CLASS_NAMES[t.item()]}, a type of flower."
                            for t in targets
                        ]
                        original_text_features = base_model.encode_text(
                            clip.tokenize(text_inputs).to(DEVICE)
                        )
                        original_text_features = (
                            original_text_features
                            / original_text_features.norm(dim=-1, keepdim=True)
                        )

                    # Mutual agreement loss
                    if lambda_ma > 0:
                        prompted_image_features = model.current_image_features[
                            :batch_size
                        ]
                        prompted_text_features = model.current_text_features[
                            :batch_size
                        ]

                        ma_loss = mutual_agreement_loss(
                            prompted_image_features,
                            prompted_text_features,
                            original_image_features,
                            original_text_features,
                            temperature=temperature,
                        )

                    # Textual diversity loss
                    if lambda_td > 0:
                        text_features_for_td = model.current_text_features[:batch_size]
                        td_loss = textual_diversity_loss(
                            text_features_for_td, temperature=temperature
                        )

                # HELIP
                if use_helip and hard_pair_miner is not None and len(hard_pairs) > 0:
                    # Get current batch indices for hard pair mapping
                    batch_start = batch_idx * train_loader.batch_size
                    batch_end = min(
                        batch_start + train_loader.batch_size, len(dataset_indices)
                    )
                    batch_indices = list(range(batch_start, batch_end))

                    # Map batch indices to hard pairs
                    batch_hard_pairs = {}
                    for i, orig_idx in enumerate(batch_indices):
                        if i >= images.size(0):  # Safety check
                            break

                        if not hasattr(train_loader.dataset, "indices"):
                            dataset_idx = orig_idx
                        else:
                            if orig_idx < len(dataset_indices):
                                dataset_idx = dataset_indices[orig_idx]
                            else:
                                continue

                        dataset_idx_int = (
                            dataset_idx.item()
                            if hasattr(dataset_idx, "item")
                            else dataset_idx
                        )

                        # Get hard pairs for this sample
                        batch_hard_pairs[i] = hard_pairs.get(dataset_idx_int, [])

                    # Compute HNML if we have hard pairs
                    if batch_hard_pairs:
                        # Get normalized features
                        image_features = model.image_encoder(images)
                        image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )

                        prompts = model.prompt_learner()
                        text_features = model.text_encoder(
                            prompts, model.tokenized_prompts
                        )
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )

                        # Compute HNML
                        hnml_loss = hard_negative_margin_loss(
                            image_features,
                            text_features,
                            batch_hard_pairs,
                            alpha=hard_pair_miner.alpha,
                        )

                # Combined loss
                lambda_hnml = CFG["helip"]["lambda_hnml"] if use_helip else 0.0
                loss = (
                    contrastive_loss
                    + lambda_hnml * hnml_loss
                    + lambda_ma * ma_loss
                    + lambda_td * td_loss
                )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update EMA model if available
            if avg_model and epoch + 1 >= CFG["validation"]["ema"]["start_epoch"]:
                avg_model.update_parameters(model)

            # Statistics
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            if use_helip:
                total_hnml_loss += (
                    hnml_loss.item() if isinstance(hnml_loss, torch.Tensor) else 0.0
                )
            if use_promptsrc:
                total_ma_loss += (
                    ma_loss.item() if isinstance(ma_loss, torch.Tensor) else 0.0
                )
                total_td_loss += (
                    td_loss.item() if isinstance(td_loss, torch.Tensor) else 0.0
                )

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            current_acc = 100.0 * correct / total
            current_loss = total_loss / (batch_idx + 1)

            if use_helip and use_promptsrc:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{current_loss:.4f}",
                        "Cont": f"{contrastive_loss.item():.4f}",
                        "HNML": f"{hnml_loss.item() if isinstance(hnml_loss, torch.Tensor) else 0.0:.4f}",
                        "MA": f"{ma_loss.item() if isinstance(ma_loss, torch.Tensor) else 0.0:.4f}",
                        "TD": f"{td_loss.item() if isinstance(td_loss, torch.Tensor) else 0.0:.4f}",
                        "Acc": f"{current_acc:.2f}%",
                    }
                )
            elif use_helip:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{current_loss:.4f}",
                        "Cont": f"{contrastive_loss.item():.4f}",
                        "HNML": f"{hnml_loss.item() if isinstance(hnml_loss, torch.Tensor) else 0.0:.4f}",
                        "Acc": f"{current_acc:.2f}%",
                    }
                )
            elif use_promptsrc:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{current_loss:.4f}",
                        "Cont": f"{contrastive_loss.item():.4f}",
                        "MA": f"{ma_loss.item() if isinstance(ma_loss, torch.Tensor) else 0.0:.4f}",
                        "TD": f"{td_loss.item() if isinstance(td_loss, torch.Tensor) else 0.0:.4f}",
                        "Acc": f"{current_acc:.2f}%",
                    }
                )
            else:
                progress_bar.set_postfix(
                    {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
                )

            # Log to wandb
            log_data = {
                "train_batch_loss": loss.item(),
                "train_batch_contrastive_loss": contrastive_loss.item(),
                "train_batch_acc": 100.0
                * predicted.eq(targets).sum().item()
                / targets.size(0),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
                "batch": batch_idx + epoch * len(train_loader),
            }

            if use_helip:
                log_data["train_batch_hnml_loss"] = (
                    hnml_loss.item() if isinstance(hnml_loss, torch.Tensor) else 0.0
                )
                log_data["lambda_hnml"] = lambda_hnml

            if use_promptsrc:
                log_data["train_batch_ma_loss"] = (
                    ma_loss.item() if isinstance(ma_loss, torch.Tensor) else 0.0
                )
                log_data["train_batch_td_loss"] = (
                    td_loss.item() if isinstance(td_loss, torch.Tensor) else 0.0
                )
                log_data["lambda_ma"] = lambda_ma
                log_data["lambda_td"] = lambda_td

            WANDB.log(log_data)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                LOGGER.error(f"CUDA OOM in training batch {batch_idx}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    # Calculate epoch averages
    avg_loss = total_loss / len(train_loader)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader)
    avg_hnml_loss = total_hnml_loss / len(train_loader) if use_helip else 0.0
    avg_ma_loss = total_ma_loss / len(train_loader) if use_promptsrc else 0.0
    avg_td_loss = total_td_loss / len(train_loader) if use_promptsrc else 0.0
    avg_acc = 100.0 * correct / total

    if use_helip and use_promptsrc:
        LOGGER.info(
            f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"HNML Loss: {avg_hnml_loss:.4f}, "
            f"MA Loss: {avg_ma_loss:.4f}, "
            f"TD Loss: {avg_td_loss:.4f}, "
            f"Accuracy: {avg_acc:.2f}%"
        )
    elif use_helip:
        LOGGER.info(
            f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"HNML Loss: {avg_hnml_loss:.4f}, "
            f"Accuracy: {avg_acc:.2f}%"
        )
    elif use_promptsrc:
        LOGGER.info(
            f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"MA Loss: {avg_ma_loss:.4f}, "
            f"TD Loss: {avg_td_loss:.4f}, "
            f"Accuracy: {avg_acc:.2f}%"
        )
    else:
        LOGGER.info(
            f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%"
        )

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, data_loader, criterion):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader, desc="Evaluating")

    for images, targets in progress_bar:
        try:
            # Move data to device
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({"Acc": f"{current_acc:.2f}%"})

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                LOGGER.error(f"CUDA OOM in evaluation: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    # Calculate averages
    avg_loss = total_loss / len(data_loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def evaluate_and_report_validation(model, val_loader, criterion, ema=False):
    """Evaluate model on validation set, return metrics"""
    # Validation accuracy
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    # Log results
    LOGGER.info(f"{'EMA ' if ema else ''}Validation Accuracy: {val_acc:.2f}%")

    return {
        f"{'ema_' if ema else ''}val_acc": val_acc,
        f"{'ema_' if ema else ''}val_loss": val_loss,
    }


def evaluate_and_report_test(
    model, test_base_loader, test_novel_loader, criterion, ema=False
):
    """Evaluate model on test sets, return metrics"""
    # Test base accuracy
    test_base_loss, test_base_acc = evaluate(model, test_base_loader, criterion)

    # Test novel accuracy
    test_novel_loss, test_novel_acc = evaluate(model, test_novel_loader, criterion)

    # Calculate harmonic mean
    if test_base_acc > 0 and test_novel_acc > 0:
        harmonic_mean = (
            2 * (test_base_acc * test_novel_acc) / (test_base_acc + test_novel_acc)
        )
    else:
        harmonic_mean = 0.0

    # Log results
    LOGGER.info(f"Test Base Accuracy: {test_base_acc:.2f}%")
    LOGGER.info(f"Test Novel Accuracy: {test_novel_acc:.2f}%")
    LOGGER.info(f"Harmonic Mean: {harmonic_mean:.2f}%")

    return {
        f"{'ema_' if ema else ''}test_base_acc": test_base_acc,
        f"{'ema_' if ema else ''}test_base_loss": test_base_loss,
        f"{'ema_' if ema else ''}test_novel_acc": test_novel_acc,
        f"{'ema_' if ema else ''}test_novel_loss": test_novel_loss,
        f"{'ema_' if ema else ''}harmonic_mean": harmonic_mean,
    }


def log_metrics(metrics, epoch):
    """Log metrics to wandb and tensorboard"""
    log_data = {}
    for key, value in metrics.items():
        log_data[key] = value
    log_data["epoch"] = epoch
    WANDB.log(log_data)


# %% [markdown]
# ## Background Masking Functions
#
# Functions for creating and training segmentation models for background suppression.


# %%
def create_segmentation_model():
    """Create segmentation model for background masking"""
    model = smp.create_model(
        arch=CFG["bg_masking"]["backbone"],
        encoder_name=CFG["bg_masking"]["encoder"]["name"],
        classes=1,
        activation="sigmoid",
        encoder_weights=CFG["bg_masking"]["encoder"]["weights"],
    ).to(DEVICE)

    params = smp.encoders.get_preprocessing_params(
        encoder_name=CFG["bg_masking"]["encoder"]["name"],
        pretrained=CFG["bg_masking"]["encoder"]["weights"],
    )
    normalizer_transform = torchvision.transforms.Normalize(
        params["mean"], params["std"]
    )
    size_transform = torchvision.transforms.CenterCrop(
        CFG["bg_masking"]["training"]["image_size"]
    )

    return model, normalizer_transform, size_transform


def fine_tune_segmenter(data_loader, val_data_loader, segmentation_model):
    """Fine-tune segmentation model for background masking"""
    LOGGER.info("Fine tuning segmentation model for background masking!")
    epochs = CFG["bg_masking"]["training"]["epochs"]
    loss_fn = smp.losses.DiceLoss(mode=smp.losses.constants.BINARY_MODE)
    optimizer = torch.optim.Adam(segmentation_model.parameters())

    segmentation_model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(data_loader, desc=f"Segmentation train Epoch {epoch}")
        total_loss = 0

        for batch, (img, mask) in enumerate(progress_bar):
            optimizer.zero_grad()
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            output = segmentation_model(img)
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "seg_loss": f"{loss.item():.4f}",
                    "seg_avg_loss": f"{total_loss / (batch + 1):.4f}",
                }
            )

            WANDB.log(
                {
                    "seg_train_loss": loss.item(),
                    "seg_train_avg_loss": total_loss / (batch + 1),
                }
            )

        avg_loss = total_loss / len(data_loader)
        LOGGER.info(f"Segmentation epoch {epoch + 1} - Training Loss: {avg_loss:.4f}")

    LOGGER.info("Finished fine-tuning segmentation model, freezing all parameters")
    for p in segmentation_model.parameters():
        p.requires_grad = False
    segmentation_model.eval()

    LOGGER.info("Testing segmentation model")
    test_loss = 0
    for img, mask in val_data_loader:
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)
        pred = segmentation_model(img)
        test_loss += loss_fn(pred, mask).item()

    test_loss /= len(val_data_loader)
    LOGGER.info(f"Segmentation model average test loss: {test_loss:.4f}")


def setup_segmentation_module():
    """Setup segmentation module for background masking"""
    if CFG["bg_masking"]["enabled"]:
        LOGGER.info("Setting up background masking with segmentation model")

        # Create segmentation model and transforms
        segmentation_model, normalizer_transform, size_transform = (
            create_segmentation_model()
        )
        segmentation_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), size_transform, normalizer_transform]
        )

        # Get datasets for segmentation training
        seg_train_set, seg_val_set, _ = get_data(
            train_transform=segmentation_transform,
            test_transform=segmentation_transform,
            mask_transform=size_transform,
            segmentation_training=True,
        )

        # Split into base classes for segmentation training
        base_classes, _ = base_novel_categories(seg_train_set)
        seg_train_base, _ = split_data(seg_train_set, base_classes)
        seg_val_base, _ = split_data(seg_val_set, base_classes)

        # Create data loaders for segmentation training
        seg_train_loader = DataLoader(
            seg_train_base,
            batch_size=CFG["bg_masking"]["training"]["batch_size"],
            shuffle=True,
        )
        seg_val_loader = DataLoader(
            seg_val_base,
            batch_size=CFG["bg_masking"]["training"]["batch_size"],
            shuffle=False,
        )

        # Fine-tune segmentation model
        fine_tune_segmenter(seg_train_loader, seg_val_loader, segmentation_model)

        return segmentation_model, normalizer_transform
    else:
        LOGGER.info("Background masking disabled")
        return None, None


# %% [markdown]
# ## Data Augmentation Functions
#
# Functions for applying different types of data augmentation to improve training.


# %%
def image_augmentation(mode, base_preprocess):
    """
    Apply data augmentation based on the specified mode.

    Args:
        mode: Augmentation mode - None, "rotate_illumination", "rotate_contrast", "rotate_contrast_illumination"
        base_preprocess: Base preprocessing transform from CLIP

    Returns:
        Composed transform with augmentation
    """
    if mode == "rotate_illumination":
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=30),
                transforms.Lambda(
                    lambda img: TF.adjust_brightness(
                        img, brightness_factor=np.random.uniform(1.1, 1.5)
                    )
                ),
                transforms.RandomResizedCrop(
                    size=CFG["input"]["size"], scale=(0.7, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                base_preprocess,
            ]
        )
    elif mode == "rotate_contrast":
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=30),
                transforms.Lambda(
                    lambda img: TF.adjust_contrast(
                        img, contrast_factor=np.random.uniform(0.8, 2.0)
                    )
                ),
                transforms.RandomResizedCrop(
                    size=CFG["input"]["size"], scale=(0.7, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                base_preprocess,
            ]
        )
    elif mode == "rotate_contrast_illumination":
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=30),
                transforms.Lambda(
                    lambda img: TF.adjust_contrast(
                        img, contrast_factor=np.random.uniform(0.8, 2.0)
                    )
                ),
                transforms.Lambda(
                    lambda img: TF.adjust_brightness(
                        img, brightness_factor=np.random.uniform(1.1, 1.5)
                    )
                ),
                transforms.RandomResizedCrop(
                    size=CFG["input"]["size"], scale=(0.7, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                base_preprocess,
            ]
        )
    else:
        # No augmentation, just return base preprocessing
        return base_preprocess


# %% [markdown]
# ## HELIP Implementation
#
# Hard Negative Mining Loss implementation for improved contrastive learning.


# %%
class HardPairMining:
    """
    HELIP: Hard Negative Mining for improved contrastive learning.
    Mines hard negative pairs to improve model training.
    """

    def __init__(self, custom_clip_model, dataset):
        self.model = custom_clip_model
        self.dataset = dataset
        self.k = CFG["helip"]["k"]
        self.p = CFG["helip"]["p"]
        self.alpha = CFG["helip"]["alpha"]

        # Storage for cached embeddings and hard pairs
        self.cached_embeddings = None
        self.cached_hard_pairs = {}
        self.cache_embeddings = CFG["helip"]["cache_embeddings"]
        self.last_mining_epoch = 0

    def _compute_pair_embeddings(self, sample_indices=None):
        """Compute embeddings for all samples in the dataset"""
        if sample_indices is None:
            dataloader = DataLoader(
                self.dataset,
                batch_size=4,
                shuffle=False,
                num_workers=CFG["data"]["num_workers"],
            )
        else:
            # Create a subset with specific indices
            subset = torch.utils.data.Subset(self.dataset, sample_indices)
            dataloader = DataLoader(
                subset,
                batch_size=4,
                shuffle=False,
                num_workers=CFG["data"]["num_workers"],
            )

        all_image_features = []
        all_text_features = []
        all_labels = []

        with torch.no_grad():
            self.model.eval()
            for images, labels in tqdm(
                dataloader, desc="Computing embeddings for HELIP"
            ):
                images = images.to(DEVICE)

                # Forward pass to get features
                logits = self.model(images)

                # Get image features
                image_features = self.model.image_encoder(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                # Get text features
                prompts = self.model.prompt_learner()
                tokenized_prompts = self.model.tokenized_prompts
                text_features = self.model.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Store the embeddings
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                all_labels.append(labels)

        if not all_image_features or not all_text_features:
            LOGGER.error(
                "No embeddings were collected during mining. Check model implementation."
            )
            return None

        embeddings = {
            "image_features": torch.cat(all_image_features),
            "text_features": torch.cat(all_text_features),
            "labels": torch.cat(all_labels),
        }

        LOGGER.info(
            f"Computed embeddings - Images: {embeddings['image_features'].shape}, "
            f"Text: {embeddings['text_features'].shape}"
        )

        return embeddings

    def mine_hard_pairs(self, dataset_indices, epoch):
        """Mine hard negative pairs for the current epoch"""
        # Check if we need to mine new pairs based on epoch frequency
        if (
            epoch // CFG["helip"]["mining_freq"]
            == self.last_mining_epoch // CFG["helip"]["mining_freq"]
            and self.cached_hard_pairs
        ):
            LOGGER.info(f"Using cached hard pairs from epoch {self.last_mining_epoch}")
            return self.cached_hard_pairs

        # Compute or load embeddings
        if self.cached_embeddings is None or not self.cache_embeddings:
            LOGGER.info("Computing embeddings for hard pair mining")
            self.cached_embeddings = self._compute_pair_embeddings()

        if self.cached_embeddings is None:
            LOGGER.warning("Failed to compute embeddings, returning empty hard pairs")
            return {}

        hard_pairs = {}

        # Get all embeddings
        all_image_features = self.cached_embeddings["image_features"]
        all_text_features = self.cached_embeddings["text_features"]

        for target_idx in tqdm(
            dataset_indices, desc=f"Mining hard pairs for epoch {epoch}"
        ):
            # Convert target_idx to int if it's a tensor
            target_idx_int = (
                target_idx.item() if hasattr(target_idx, "item") else target_idx
            )

            # Skip if index is out of bounds
            if target_idx_int >= len(all_image_features):
                continue

            # Get target embeddings
            target_image_emb = all_image_features[target_idx_int].to(DEVICE)
            target_text_emb = all_text_features[target_idx_int].to(DEVICE)

            # Randomly sample p candidate pairs
            total_samples = len(all_image_features)
            random_indices = torch.randperm(total_samples)[: self.p]

            # Calculate agreement scores (how well other pairs match with target pair)
            agreement_scores = []

            for idx in random_indices:
                if idx == target_idx_int:
                    continue

                candidate_image_emb = all_image_features[idx].to(DEVICE)
                candidate_text_emb = all_text_features[idx].to(DEVICE)

                # Calculate cross-modal similarity scores
                i2t_score = torch.matmul(
                    target_image_emb.flatten(), candidate_text_emb.flatten()
                )
                t2i_score = torch.matmul(
                    candidate_image_emb.flatten(), target_text_emb.flatten()
                )
                score = i2t_score + t2i_score

                # Convert idx to int if it's a tensor
                idx_int = idx.item() if hasattr(idx, "item") else idx
                agreement_scores.append((idx_int, score.item()))

            # Select k hardest pairs (lowest agreement scores)
            hard_pairs_indices = sorted(agreement_scores, key=lambda x: x[1])[: self.k]
            hard_pairs[target_idx_int] = [idx for idx, _ in hard_pairs_indices]

        # Cache results
        self.cached_hard_pairs = hard_pairs
        self.last_mining_epoch = epoch

        LOGGER.info(f"Mined {len(hard_pairs)} hard pair sets for epoch {epoch}")
        return hard_pairs


def hard_negative_margin_loss(
    image_features,  # Normalized image features [batch_size, feature_dim]
    text_features,  # Normalized text features [batch_size, feature_dim]
    hard_pairs_indices,  # Dictionary mapping indices to lists of hard pair indices
    alpha=0.25,  # Margin parameter
):
    """
    Compute Hard Negative Margin Loss (HNML) for HELIP.

    Args:
        image_features: Normalized image features
        text_features: Normalized text features
        hard_pairs_indices: Dictionary mapping sample indices to hard negative indices
        alpha: Margin parameter for hinge loss

    Returns:
        HNML loss tensor
    """
    batch_size = image_features.shape[0]

    # Calculate similarity matrix between all images and texts
    sim_matrix = image_features @ text_features.T

    # Define positive pair similarities (diagonal elements)
    pos_idx = torch.arange(batch_size, device=DEVICE)
    pos_sim = sim_matrix[pos_idx, pos_idx]

    hnml_loss = torch.tensor(0.0, device=DEVICE)
    count = 0

    for i in range(batch_size):
        sample_idx = i

        if sample_idx not in hard_pairs_indices or not hard_pairs_indices[sample_idx]:
            continue

        hard_indices_list = hard_pairs_indices[sample_idx]
        hard_indices = torch.tensor(hard_indices_list, device=DEVICE)

        # Find indices that are in the current batch
        batch_mask = hard_indices < batch_size
        if not batch_mask.any():
            continue

        hard_indices = hard_indices[batch_mask]

        # Get similarities for hard negatives
        hard_neg_sim_i2t = sim_matrix[i, hard_indices]
        hard_neg_sim_t2i = sim_matrix[hard_indices, i]

        # Calculate hinge loss with margin for each hard negative
        i2t_loss = torch.sum(torch.clamp(hard_neg_sim_i2t - pos_sim[i] + alpha, min=0))
        t2i_loss = torch.sum(torch.clamp(hard_neg_sim_t2i - pos_sim[i] + alpha, min=0))

        hnml_loss += i2t_loss + t2i_loss
        count += len(hard_indices) * 2

    # Normalize by actual number of pairs processed, or return zero if no pairs
    if count > 0:
        return hnml_loss / count
    return hnml_loss


# %% [markdown]
# ## PromptSRC Implementation
#
# PromptSRC (Prompt Source Regularization and Consistency) implementation for improved few-shot learning.


# %%
class PromptEnsemble:
    """
    PromptSRC: Gaussian weighted prompt aggregation for temporal consistency.
    Maintains a history of context vectors and creates ensembles using Gaussian weighting.
    """

    def __init__(self, window_size=None, sigma=None):
        self.window_size = (
            window_size
            if window_size is not None
            else CFG["promptsrc"]["prompt_ensemble"]["window_size"]
        )
        self.sigma = (
            sigma if sigma is not None else CFG["promptsrc"]["prompt_ensemble"]["sigma"]
        )
        self.prompt_history = []

    def update(self, current_prompt):
        """Update prompt history with current prompt"""
        cpu_prompt = current_prompt.detach().clone().cpu()
        self.prompt_history.append(cpu_prompt)

        # Keep only window_size most recent prompts
        if len(self.prompt_history) > self.window_size:
            self.prompt_history.pop(0)

    def get_ensemble_prompt(self):
        """Create Gaussian-weighted ensemble of prompts"""
        if not self.prompt_history:
            return None

        device_prompts = [p.to(DEVICE) for p in self.prompt_history]

        # Compute Gaussian weights favoring recent prompts
        weights = [
            math.exp(-((len(device_prompts) - 1 - i) ** 2) / (2 * self.sigma**2))
            for i in range(len(device_prompts))
        ]
        weights = torch.tensor(weights, device=DEVICE)
        weights = weights / weights.sum()

        # Weighted ensemble
        ensemble_prompt = sum(w * p for w, p in zip(weights, device_prompts))
        return ensemble_prompt


def mutual_agreement_loss(
    image_features,
    text_features,
    original_image_features,
    original_text_features,
    temperature=0.07,
):
    """
    PromptSRC: Mutual agreement loss to encourage consistency with frozen CLIP features.

    Args:
        image_features: Prompted model image features
        text_features: Prompted model text features
        original_image_features: Frozen CLIP image features
        original_text_features: Frozen CLIP text features
        temperature: Temperature scaling parameter

    Returns:
        Mutual agreement loss (symmetric KL divergence)
    """
    # Normalize features for cosine similarity
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    original_image_features = F.normalize(original_image_features, dim=-1)
    original_text_features = F.normalize(original_text_features, dim=-1)

    # Compute similarity matrices for prompted and frozen features
    prompted_logits = (image_features @ text_features.T) / temperature
    frozen_logits = (original_image_features @ original_text_features.T) / temperature

    # Convert to probability distributions
    prompted_probs = F.softmax(prompted_logits, dim=-1)
    frozen_probs = F.softmax(frozen_logits, dim=-1)

    # Symmetric KL divergence (Jensen-Shannon without the averaging)
    loss_p_to_f = F.kl_div(prompted_probs.log(), frozen_probs, reduction="batchmean")
    loss_f_to_p = F.kl_div(frozen_probs.log(), prompted_probs, reduction="batchmean")

    return loss_p_to_f + loss_f_to_p


def textual_diversity_loss(text_features, temperature=0.07):
    """
    PromptSRC: Textual diversity loss to encourage diversity in text features.
    Uses Jensen-Shannon divergence to promote feature diversity.

    Args:
        text_features: Text features from prompted model
        temperature: Temperature scaling parameter

    Returns:
        Textual diversity loss (Jensen-Shannon divergence)
    """
    text_features = F.normalize(text_features, dim=-1)

    sim_matrix = (text_features @ text_features.T) / temperature

    # Compute probability distribution for each text feature
    probs = F.softmax(sim_matrix, dim=-1)

    # Compute average distribution (mixture distribution M in JS divergence)
    avg_probs = probs.mean(dim=0, keepdim=True)

    # Compute KL divergence between each distribution and average
    kl_div = F.kl_div(probs.log(), avg_probs.expand_as(probs), reduction="none")
    reverse_kl = F.kl_div(avg_probs.expand_as(probs).log(), probs, reduction="none")

    # JS divergence
    js_div = 0.5 * (kl_div + reverse_kl).sum(dim=-1).mean()

    return js_div


def get_promptsrc_weights(epoch=None, max_epochs=None):
    """Get PromptSRC loss weights from configuration"""
    lambda_ma = CFG["promptsrc"]["mutual_agreement"]["lambda_ma"]
    lambda_td = CFG["promptsrc"]["textual_diversity"]["lambda_td"]
    return lambda_ma, lambda_td


# %% [markdown]
# ## Main Function for Running Experiments
#
# This function handles the complete training and evaluation pipeline.


# %%
def main():
    """Main function for running the CoOp experiment"""
    LOGGER.info("Starting CLIP + CoOp experiment on Oxford Flowers 102")

    try:
        # Create output directory
        os.makedirs(RUN_DIR, exist_ok=True)

        # Save configuration
        config_path = os.path.join(RUN_DIR, "config.json")
        with open(config_path, "w") as f:
            json.dump(CFG, f, indent=4)
        LOGGER.info(f"Configuration saved to {config_path}")

        # Setup background masking if enabled
        segmentation_model, segmentation_transform = setup_segmentation_module()

        # Load CLIP model using configuration
        model_name = CFG["COOP"]["base_model"]["name"]

        clip_model, preprocess = load_clip_model(model_name)

        # Apply data augmentation to training data if enabled
        augmentation_mode = CFG["training"]["augmentation_mode"]
        LOGGER.info(
            f"Applying data augmentation: {augmentation_mode if augmentation_mode else 'default'}"
        )
        train_transform = image_augmentation(augmentation_mode, preprocess)

        # Load dataset with appropriate transforms
        train_set, val_set, test_set = get_data(
            data_dir=CFG["data"]["data_dir"],
            train_transform=train_transform,
            test_transform=preprocess,
        )

        # Split dataset into base and novel categories
        base_classes, novel_classes = base_novel_categories(train_set)
        LOGGER.info(
            f"Base classes: {len(base_classes)}, Novel classes: {len(novel_classes)}"
        )

        # Split datasets
        train_base, _ = split_data(train_set, base_classes)
        val_base, _ = split_data(val_set, base_classes)
        test_base, test_novel = split_data(test_set, base_classes)

        # Create few-shot dataset for training
        shots_per_class = CFG["training"]["shots_per_class"]
        if shots_per_class > 0:
            LOGGER.info(f"Creating {shots_per_class}-shot dataset for training")
            train_base = create_few_shot_dataset(train_base, shots_per_class)

        # Create data loaders
        train_loader, val_loader, test_base_loader, test_novel_loader = (
            create_data_loaders(
                train_base,
                val_base,
                test_base,
                test_novel,
                batch_size=CFG["training"]["batch_size"],
                test_batch_size=CFG["training"]["test_batch_size"],
                num_workers=CFG["data"]["num_workers"],
                pin_memory=CFG["data"]["pin_memory"],
            )
        )

        # Report dataset info
        LOGGER.info(f"Training set size: {len(train_base)}")
        LOGGER.info(f"Validation set size: {len(val_base)}")
        LOGGER.info(f"Test base set size: {len(test_base)}")
        LOGGER.info(f"Test novel set size: {len(test_novel)}")

        # Zero-shot evaluation with CLIP if requested
        if CFG["validation"]["evaluate_zero_shot"]:
            LOGGER.info("Evaluating CLIP with zero-shot learning")

            # Evaluate on base and novel categories
            LOGGER.info("Evaluating CLIP on base categories")
            base_acc = zero_shot_evaluation(
                clip_model,
                test_base_loader,
                [CLASS_NAMES[i] for i in base_classes],
                base_classes,
            )

            LOGGER.info("Evaluating CLIP on novel categories")
            novel_acc = zero_shot_evaluation(
                clip_model,
                test_novel_loader,
                [CLASS_NAMES[i] for i in novel_classes],
                novel_classes,
            )

            # Calculate harmonic mean
            if base_acc > 0 and novel_acc > 0:
                harmonic_mean = 2 * (base_acc * novel_acc) / (base_acc + novel_acc)
            else:
                harmonic_mean = 0.0

            LOGGER.info(f"CLIP Zero-Shot - Base Accuracy: {base_acc:.2f}%")
            LOGGER.info(f"CLIP Zero-Shot - Novel Accuracy: {novel_acc:.2f}%")
            LOGGER.info(f"CLIP Zero-Shot - Harmonic Mean: {harmonic_mean:.2f}%")

            # Log to wandb
            WANDB.log(
                {
                    "zero_shot_base_acc": base_acc,
                    "zero_shot_novel_acc": novel_acc,
                    "zero_shot_harmonic_mean": harmonic_mean,
                }
            )

        # Create CoOp model
        coop_config = CFG["COOP"]["prompt_learner"]
        model = CoOp(
            classnames=[CLASS_NAMES[i] for i in base_classes + novel_classes],
            clip_model=clip_model,
            n_ctx=coop_config["n_ctx"],
            ctx_init=coop_config["ctx_init"],
            class_token_position=coop_config["class_token_position"],
            csc=coop_config["csc"],
            segmentation_model=segmentation_model,
            segmentation_transform=segmentation_transform,
        ).to(DEVICE)

        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.info(f"Total parameters: {total_params:,}")
        LOGGER.info(f"Trainable parameters: {trainable_params:,}")

        # Create EMA model if enabled
        avg_model = None
        if CFG["validation"]["ema"]["enabled"]:
            avg_model = AveragedModel(
                model,
                multi_avg_fn=get_ema_multi_avg_fn(CFG["validation"]["ema"]["decay"]),
                use_buffers=True,
            )
            LOGGER.info(
                f"Created EMA model with decay {CFG['validation']['ema']['decay']}"
            )

        # Initialize HELIP if enabled
        hard_pair_miner = None
        use_helip = CFG["helip"]["enabled"]
        if use_helip:
            LOGGER.info("HELIP is enabled for training")
            # Initialize hard pair mining for base classes only
            hard_pair_miner = HardPairMining(
                custom_clip_model=model, dataset=train_base
            )
            LOGGER.info(
                f"HELIP parameters: k={CFG['helip']['k']}, p={CFG['helip']['p']}, alpha={CFG['helip']['alpha']}, lambda_hnml={CFG['helip']['lambda_hnml']}"
            )
        else:
            LOGGER.info("HELIP is disabled, using standard training")

        # Initialize PromptSRC if enabled
        use_promptsrc = CFG["promptsrc"]["enabled"]
        if use_promptsrc:
            LOGGER.info("PromptSRC is enabled for training")
            LOGGER.info(
                f"PromptSRC parameters: lambda_ma={CFG['promptsrc']['mutual_agreement']['lambda_ma']}, lambda_td={CFG['promptsrc']['textual_diversity']['lambda_td']}"
            )
            if CFG["promptsrc"]["prompt_ensemble"]["enabled"]:
                LOGGER.info(
                    f"Prompt ensemble enabled: window_size={CFG['promptsrc']['prompt_ensemble']['window_size']}, sigma={CFG['promptsrc']['prompt_ensemble']['sigma']}"
                )
        else:
            LOGGER.info("PromptSRC is disabled, using standard training")

        # Set up optimizer
        optimizer_config = CFG["training"]["optimizer"]
        optimizer = optim.SGD(
            model.prompt_learner.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
        )

        # Set up learning rate scheduler
        scheduler_config = CFG["training"]["scheduler"]
        if scheduler_config["type"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG["training"]["epochs"],
                eta_min=scheduler_config["eta_min"],
            )
        else:
            scheduler = None

        # Set up loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_validation_acc = 0.0
        best_epoch = -1
        patience_counter = 0

        # Evaluate before training
        LOGGER.info("Evaluating before training")
        initial_metrics = evaluate_and_report_test(
            model, test_base_loader, test_novel_loader, criterion
        )
        log_metrics(initial_metrics, 0)

        for epoch in range(CFG["training"]["epochs"]):
            # Train for one epoch
            train_loss, train_acc = train_coop(
                model,
                train_loader,
                optimizer,
                criterion,
                epoch,
                avg_model,
                hard_pair_miner,
                use_helip,
                clip_model,
                use_promptsrc,
            )

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Evaluate the model
            LOGGER.info(f"Epoch {epoch + 1}/{CFG['training']['epochs']} evaluation")
            val_metrics = evaluate_and_report_validation(model, val_loader, criterion)

            # Evaluate EMA model if available
            if avg_model and epoch + 1 >= CFG["validation"]["ema"]["start_epoch"]:
                LOGGER.info(
                    f"Epoch {epoch + 1}/{CFG['training']['epochs']} EMA evaluation"
                )
                ema_metrics = evaluate_and_report_validation(
                    avg_model, val_loader, criterion, ema=True
                )
                log_metrics(ema_metrics, epoch + 1)
            else:
                ema_metrics = {}

            # Log metrics
            epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            }
            log_metrics(epoch_metrics, epoch + 1)

            # Choose best model
            current_val_acc = val_metrics["val_acc"]
            if current_val_acc > best_validation_acc:
                best_validation_acc = current_val_acc
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                checkpoint_dir = CFG["training"]["checkpoint_dir"]
                os.makedirs(checkpoint_dir, exist_ok=True)
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "ema_model_state_dict": avg_model.state_dict()
                        if avg_model
                        else None,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                        if scheduler
                        else None,
                        "metrics": val_metrics,
                        "ema_metrics": ema_metrics,
                        "config": CFG,
                    },
                    best_model_path,
                )

                LOGGER.info(
                    f"New best model saved with validation accuracy: {best_validation_acc:.2f}%"
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= CFG["training"]["patience"]:
                LOGGER.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

        # Load best model and do final evaluation
        if best_epoch >= 0:
            LOGGER.info(f"Loading best model from epoch {best_epoch + 1}")
            checkpoint_path = os.path.join(
                CFG["training"]["checkpoint_dir"], "best_model.pth"
            )
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

                # Load into appropriate model
                if avg_model and checkpoint["ema_model_state_dict"]:
                    avg_model.load_state_dict(checkpoint["ema_model_state_dict"])
                    LOGGER.info("Loaded best EMA model for final evaluation")
                else:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    LOGGER.info("Loaded best regular model for final evaluation")

                # Final evaluation
                LOGGER.info("Final evaluation with best model")
                final_metrics = evaluate_and_report_test(
                    model, test_base_loader, test_novel_loader, criterion
                )
                LOGGER.info("=== FINAL RESULTS ===")
                LOGGER.info(f"Best epoch: {best_epoch + 1}")
                LOGGER.info(
                    f"Final test base accuracy: {final_metrics['test_base_acc']:.2f}%"
                )
                LOGGER.info(
                    f"Final test novel accuracy: {final_metrics['test_novel_acc']:.2f}%"
                )
                LOGGER.info(
                    f"Final harmonic mean: {final_metrics['harmonic_mean']:.2f}%"
                )
                final_log = {"final_" + k: v for k, v in final_metrics.items()}
                final_log["best_epoch"] = best_epoch + 1

                if avg_model:
                    LOGGER.info("Final evaluation with best ema model")
                    final_ema_metrics = evaluate_and_report_test(
                        avg_model,
                        test_base_loader,
                        test_novel_loader,
                        criterion,
                        ema=True,
                    )
                    LOGGER.info("=== FINAL EMA RESULTS ===")
                    LOGGER.info(
                        f"Final test base accuracy: {final_ema_metrics['ema_test_base_acc']:.2f}%"
                    )
                    LOGGER.info(
                        f"Final test novel accuracy: {final_ema_metrics['ema_test_novel_acc']:.2f}%"
                    )
                    LOGGER.info(
                        f"Final harmonic mean: {final_ema_metrics['ema_harmonic_mean']:.2f}%"
                    )
                    for k, v in final_ema_metrics.items():
                        final_log["final" + k] = v

                # Log final results to wandb
                WANDB.log(final_log)

                return final_metrics
            else:
                LOGGER.warning("Best model checkpoint not found")
                return initial_metrics
        else:
            LOGGER.warning("No best model found")
            return initial_metrics

    except Exception as e:
        LOGGER.error(f"Error in main execution: {e}")
        raise e
    finally:
        # Cleanup
        WANDB.finish()


# %%
if __name__ == "__main__":
    main()
