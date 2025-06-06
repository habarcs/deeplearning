# TODO - HELIP:
# 1. Once HELIP is validated by the group, take out all the comments that containt the word "HELIP" (except the ones in the config)
# 2. To make sure HELIP is actually doing something, the HNML in the logs should be non-zero.
# 3. The loss should be lower than without HELIP.

# TODO - PromptSRC:
# 1. Once PromptSRC is validated, delete all the comments that containt the word "PromptSRC"


# installing packages, use specific versions
# TODO: Make sure HELIP requirements are well included here
# !pip3 install torch==2.6.0 torchvision==0.21.0 open_clip_torch==2.32.0 wandb==0.19.10 scipy==1.15.3 segmentation-models-pytorch==0.5.0

import logging
import math
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import open_clip
import PIL.Image
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

import wandb

CFG = {
    "COCOOP": {
        "base_model": {
            "name": "ViT-B-32",
            "weights": "laion2b_s34b_b79k",
        },
        "prompt_learner": {
            "n_ctx": 8,  # Reduced from 16 to save memory
            "ctx_init": "",
        },
        "text_encoder": {},
    },
    "bg-masking": {
        "enabled": True,
        # available options: https://smp.readthedocs.io/en/latest/encoders.html
        "backbone": "unet",
        "encoder": {
            "name": "mobilenet_v2",
            "weights": "imagenet",
        },
        "training": {"epochs": 1, "batch_size": 8, "image_size": 224},
    },
    "wandb": False,
    "training": {
        "resume_id": None,
        "resume_from": "best",
        "batch_size": 4,  # 32 is the default batch size for the ViT-B-32 model but it was crashing my local machine
        "epochs": 3,  # TODO: More
        "warmup_epochs": 1,  # Possibly we'll need more
        "patience": 5,
        "checkpoint_dir": "./checkpoints",
        "optimizer": {
            "lr": 0.002,  # TODO: test
            "weight_decay": 0.05,  # TODO: test
        },
        "scheduler": {
            "warmup_epochs": 1,
        },
    },
    "input": {
        "size": [224, 224]  # Input image size
    },
    "data": {
        "data_dir": "./data",
        "num_workers": 0,  # Enable parallel data loading when running in cloud, use 2 or 4
        "pin_memory": False,  # Idem
    },
    "validation": {
        "ema": {
            "enabled": True,
            "decay": 0.999,
        }
    },
    "helip": {
        "enabled": True,
        "k": 5,  # Number of hard pairs to mine per sample, diminishing returns, in the paper they also suggested 3
        "p": 20,  # Number of random pairs to sample
        "alpha": 0.05,  # Margin parameter for HNML (Hard Negative Mining Loss) # original: 0.25 # TUNE
        "lambda_hnml": 0.5,  # Weight for HNML in the combined loss -                                 # TUNE
        # TODO - 1: find a better value for key hyperparam
        # TODO - 2: balance this lambda with other regularizers lambdas, if all high, we'll underfit
        # (cross-modal consistency, augmentation, etc.)
        "mining_freq": 5,  # Mine hard pairs every N epochs (to avoid mining each epoch)
        "cache_embeddings": True,  # Whether to cache embeddings between epochs, to avoid recomputing them
    },
    "promptsrc": {
        "enabled": True,
        "mutual_agreement": {
            "lambda_ma": 0.5,  # Weight for mutual agreement loss                                 # TUNE
            "temperature": 0.07,  # Temperature scaling parameter
            "schedule": True,  # Enable weight scheduling
        },
        "prompt_ensemble": {
            "enabled": True,
            "window_size": 5,  # Number of recent prompts to ensemble
            "sigma": 2.0,  # Sigma for Gaussian weighting
        },
        "textual_diversity": {
            "lambda_td": 0.3,  # Weight for textual diversity loss                                # TUNE
            "temperature": 0.07,  # Temperature for text similarity
            "schedule": True,  # Enable weight scheduling
        },
    },
}

RUN_ID = (
    CFG["training"]["resume_id"]
    if CFG["training"]["resume_id"]
    else datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
)
RUN_DIR = f"out/{RUN_ID}/"
os.makedirs(RUN_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(RUN_DIR + "training.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("cocoop_training")

WANDB = (
    wandb.init(
        entity="mhevizi-unitn",
        project="Deep learning project",
        config=CFG,
        name="CoCoOp_" + RUN_ID,
        tags=["cocoop", CFG["COCOOP"]["base_model"]["name"]],
    )
    if CFG["wandb"]
    else wandb.init(mode="disabled")
)

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
LOGGER.info(f"Using {DEVICE} device")


class Flowers102WithSeqmentation(Flowers102):
    """
    Flowers102 dataset with seqmentation
    if segmentation_training is enabled, __getitem__ will return the segmentation mask instead of the label
    can be disabled after initialization
    """

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
        self._file_dict["seg"] = ("102segmentations.tgz", "51375996b6c4f367a4b9f4492d7ecef5")
        self.mask_transform = mask_transform
        super().__init__(root, split, transform, target_transform, download)
        self._seg_folder = self._base_folder / "segmim"
        self._image_segs = []
        for image_file in self._image_files:
            self._image_segs.append(self._seg_folder / re.sub(r"image_", "segmim_", image_file.name))

    def get_bg_mask(self, idx: int) -> Any:
        seg_file = self._image_segs[idx]
        seg_image = PIL.Image.open(seg_file).convert("RGB")
        seg = pil_to_tensor(seg_image)
        # inverting the bg mask, to get 0 for background
        mask = ~((seg[0] == 0) & (seg[1] == 0) & (seg[2] == 254))  # 0 0 254 blue represents the bg in the segmentation
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
    "pink-yellow dahlia?",
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


def get_data(
    data_dir="./data", train_transform=None, test_transform=None, mask_transform=None, segmentation_training=False
):
    LOGGER.info(
        f"Loading Flowers102 dataset from {data_dir} {'for segmentation training' if segmentation_training else ''}"
    )

    train = Flowers102WithSeqmentation(
        root=data_dir,
        split="train",
        download=True,
        transform=train_transform,
        mask_transform=mask_transform,
        segmentation_training=segmentation_training,
    )
    val = Flowers102WithSeqmentation(
        root=data_dir,
        split="val",
        download=True,
        transform=test_transform,
        mask_transform=mask_transform,
        segmentation_training=segmentation_training,
    )
    test = Flowers102WithSeqmentation(
        root=data_dir,
        split="test",
        download=True,
        transform=test_transform,
        mask_transform=mask_transform,
        segmentation_training=segmentation_training,
    )

    LOGGER.info(f"Dataset loaded: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def base_novel_categories(dataset):
    # set returns the unique set of all dataset classes
    all_classes = set(dataset._labels)
    # and let's count them
    num_classes = len(all_classes)

    # here list(range(num_classes)) returns a list from 0 to num_classes - 1
    # then we slice the list in half and generate base and novel category lists
    base_classes = list(range(num_classes))[: num_classes // 2]
    novel_classes = list(range(num_classes))[num_classes // 2 :]
    return base_classes, novel_classes


def split_data(dataset, base_classes):
    # these two lists will store the sample indexes
    base_categories_samples = []
    novel_categories_samples = []

    # we create a set of base classes to compute the test below in O(1)
    # this is optional and can be removed
    base_set = set(base_classes)

    # here we iterate over sample labels and also get the correspondent sample index
    for sample_id, label in enumerate(dataset._labels):
        if label in base_set:
            base_categories_samples.append(sample_id)
        else:
            novel_categories_samples.append(sample_id)

    # here we create the dataset subsets
    # the torch Subset is just a wrapper around the dataset
    # it simply stores the subset indexes and the original dataset (your_subset.dataset)
    # when asking for sample i in the subset, torch will look for its original position in the dataset and retrieve it
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
    base_dataset = torch.utils.data.Subset(dataset, base_categories_samples)
    novel_dataset = torch.utils.data.Subset(dataset, novel_categories_samples)
    return base_dataset, novel_dataset


# HELIP


class HardPairMining:
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
        if sample_indices is None:
            dataloader = DataLoader(
                self.dataset,
                batch_size=4,  # TODO: find proper batch size (for now, a very small one to avoid memory issues
                shuffle=False,
                num_workers=CFG["data"]["num_workers"],
            )
        else:
            # Create a subset with specific indices
            subset = torch.utils.data.Subset(self.dataset, sample_indices)
            dataloader = DataLoader(
                subset,
                batch_size=4,  # TODO: Idem above
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

                # Run the full CoCoOp forward pass to get text features with the meta-network
                # Important to use CoCoOp's prompt mechanism properly
                _ = self.model(
                    images
                )  # This will populate self.model.text_features

                image_features = self.model.image_encoder(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                if self.model.text_features is not None:
                    # For CoCoOp, text_features can have shape [batch_size * num_classes, feature_dim]
                    # Ensure we only take features for this batch
                    batch_size = image_features.size(0)

                    # Handle different text_features shapes based on CoCoOp implementation
                    if self.model.text_features.size(0) > batch_size:
                        # If text_features contains features for all classes, get just batch_size entries
                        text_features = self.model.text_features[:batch_size]
                    else:
                        text_features = self.model.text_features

                    # Store the embeddings
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    all_labels.append(labels)
                else:
                    LOGGER.warning(
                        "No text features found after forward pass. Skipping batch."
                    )

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
            f"Computed embeddings - Images: {embeddings['image_features'].shape}, Text: {embeddings['text_features'].shape}"
        )

        # Verify sizes match
        if embeddings["image_features"].size(0) != embeddings["text_features"].size(0):
            LOGGER.warning(
                f"Mismatch in number of samples: image={embeddings['image_features'].size(0)}, text={embeddings['text_features'].size(0)}"
            )
            LOGGER.warning("Attempting to fix by truncating to shorter dimension")
            min_size = min(
                embeddings["image_features"].size(0),
                embeddings["text_features"].size(0),
            )
            embeddings["image_features"] = embeddings["image_features"][:min_size]
            embeddings["text_features"] = embeddings["text_features"][:min_size]
            embeddings["labels"] = embeddings["labels"][:min_size]

        return embeddings

    def mine_hard_pairs(self, dataset_indices, epoch):
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

            # Get target embeddings
            target_image_emb = all_image_features[target_idx].to(DEVICE)
            target_text_emb = all_text_features[target_idx].to(DEVICE)

            # Randomly sample p candidate pairs
            total_samples = len(all_image_features)
            random_indices = torch.randperm(total_samples)[: self.p]

            # Calculate agreement scores (how well other pairs match with target pair)
            agreement_scores = []

            for idx in random_indices:
                if idx == target_idx:
                    continue

                candidate_image_emb = all_image_features[idx].to(DEVICE)
                candidate_text_emb = all_text_features[idx].to(DEVICE)

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

            hard_pairs_indices = sorted(agreement_scores, key=lambda x: x[1])[: self.k]
            hard_pairs[target_idx_int] = [idx for idx, _ in hard_pairs_indices]

        # Cache results
        self.cached_hard_pairs = hard_pairs
        self.last_mining_epoch = epoch

        return hard_pairs


def hard_negative_margin_loss(
    image_features,  # Normalized image features [batch_size, feature_dim]
    text_features,  # Normalized text features [batch_size, feature_dim]
    hard_pairs_indices,  # Dictionary mapping indices to lists of hard pair indices
    alpha=0.05,  # Margin parameter (originally set at 0.25, flexibilized, should be tested)
):
    batch_size = image_features.shape[0]

    # Calculate similarity matrix between all images and texts
    sim_matrix = image_features @ text_features.T

    # Define positive pair similarities (diagonal elements)
    pos_idx = torch.arange(batch_size, device=DEVICE)
    pos_sim = sim_matrix[pos_idx, pos_idx]

    hnml_loss = torch.tensor(0.0, device=DEVICE)
    count = 0

    for i in range(batch_size):
        # Convert sample_idx to int if it's a tensor
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

        # Smilarities
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


# Implement Gaussian weighted prompt aggregation from PromptSRC
# What we're doing here is:
# 1. Create a history of context vectors
# 2. Update the history with the current context vectors
# 3. Create an ensemble of the prompts
# 4. Use the ensemble as the prompt for the next forward pass
class PromptEnsemble:
    def __init__(self, window_size=5, sigma=2.0):
        self.window_size = window_size
        self.sigma = sigma
        self.prompt_history = []

    def update(self, current_prompt):
        cpu_prompt = current_prompt.detach().clone().cpu()
        self.prompt_history.append(cpu_prompt)

        # Keep only window_size most recent prompts
        if len(self.prompt_history) > self.window_size:
            self.prompt_history.pop(0)

    # Create Gaussian-weighted ensemble of prompts
    def get_ensemble_prompt(self):
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


# Implement mutual agreement loss from PromptSRC
# Goal: Encourage the model to produce text features that are similar to CLIP's original frozen model features
def mutual_agreement_loss(
    image_features,
    text_features,
    original_image_features,
    original_text_features,
    temperature=0.07,
):
    # Normalize features for cosine similarity
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    original_image_features = F.normalize(original_image_features, dim=-1)
    original_text_features = F.normalize(original_text_features, dim=-1)

    # Compute similarity matrices for prompted and frozen features
    prompted_logits = (image_features @ text_features.T) / temperature
    frozen_logits = (original_image_features @ original_text_features.T) / temperature

    # Convert to probabilty distributions
    prompted_probs = F.softmax(prompted_logits, dim=-1)
    frozen_probs = F.softmax(frozen_logits, dim=-1)

    # Symmetric KL divergence (Jensen-Shannon without the averaging)
    loss_p_to_f = F.kl_div(prompted_probs.log(), frozen_probs, reduction="batchmean")
    loss_f_to_p = F.kl_div(frozen_probs.log(), prompted_probs, reduction="batchmean")

    return loss_p_to_f + loss_f_to_p


# Implement Jensen-Shannon divergence from PromptSRC. Goal: encourage diversity in text features
def textual_diversity_loss(text_features, temperature=0.07):
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


# Implements scheduling for PromptSRC loss weights. Increase gradually the weights with a warmup phase for more stable training.
def get_promptsrc_weights(epoch, max_epochs, base_ma=0.5, base_td=0.3):
    # Linear warmup for first 20% of training
    warmup_epochs = max_epochs * 0.2

    if epoch < warmup_epochs:
        # Gradually increase weights during warmup
        factor = epoch / warmup_epochs
        lambda_ma = base_ma * factor
        lambda_td = base_td * factor
    else:
        # Use full weights after warmup
        lambda_ma = base_ma
        lambda_td = base_td

    return lambda_ma, lambda_td


def train_loop(
    dataloader,
    model,
    avg_model,
    optimizer,
    epoch,
    scheduler=None,
    # HELIP
    use_helip=False,
    hard_pair_miner=None,  # Hard pair mining module
    lambda_hnml=0.5,  # Weight for hard negative margin loss
    # Start PromptSRC
    use_promptsrc=False,
    base_model=None,  # Original CLIP model for mutual agreement
    tokenizer=None,  # Tokenizer for encoding class names
    lambda_ma=0.5,  # Weight for mutual agreement loss
    lambda_td=0.3,  # Weight for textual diversity loss
    temperature=0.07,  # Temperature for loss calculation
    max_epochs=None,  # Max epochs for weight scheduling
):
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    total_hnml_loss = 0

    # Initialize tracking for PromptSRC losses
    total_ma_loss = 0
    total_td_loss = 0

    # Get scheduled weights if enabled
    if (
        use_promptsrc
        and max_epochs is not None
        and CFG["promptsrc"]["mutual_agreement"].get("schedule", False)
    ):
        lambda_ma, lambda_td = get_promptsrc_weights(
            epoch,
            max_epochs,
            base_ma=CFG["promptsrc"]["mutual_agreement"]["lambda_ma"],
            base_td=CFG["promptsrc"]["textual_diversity"]["lambda_td"],
        )
        LOGGER.info(
            f"PromptSRC scheduled weights: lambda_ma={lambda_ma:.4f}, lambda_td={lambda_td:.4f}"
        )
    # End PromptSRC

    # Progress bar for training
    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

    # Initialize hard_pairs as empty dict by default
    hard_pairs = {}

    # If HELIP is enabled, mine hard pairs for this epoch
    if use_helip and hard_pair_miner is not None:
        # Get all dataset indices for current batch
        dataset_indices = (
            dataloader.dataset.indices
            if hasattr(dataloader.dataset, "indices")
            else list(range(len(dataloader.dataset)))
        )

        # Mine hard pairs for this epoch
        hard_pairs = hard_pair_miner.mine_hard_pairs(dataset_indices, epoch)
    # End of HELIP

    # Training loop
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to device
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Standard CoCoOp forward pass which returns cross-entropy loss
        contrastive_loss = model(images, targets)

        # Convert non-scalar loss to scalar for backprop
        if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.dim() > 0:
            LOGGER.info(
                f"Converting contrastive loss of shape {contrastive_loss.shape} to scalar for backprop"
            )
            contrastive_loss = contrastive_loss.mean()

        try:
            # Check if contrastive_loss is a scalar or can be converted to one
            if (
                isinstance(contrastive_loss, torch.Tensor)
                and contrastive_loss.numel() == 1
            ):
                cont_loss_val = contrastive_loss.item()
            else:
                # If it's not a scalar, log a warning and use a dummy value
                LOGGER.warning(
                    f"Contrastive loss is not a scalar (shape: {contrastive_loss.shape if hasattr(contrastive_loss, 'shape') else 'unknown'}). Using mean for logging."
                )
                cont_loss_val = (
                    contrastive_loss.mean().item()
                    if isinstance(contrastive_loss, torch.Tensor)
                    else 0.0
                )
        except Exception as e:
            LOGGER.warning(
                f"Error converting contrastive loss to scalar: {e}. Using 0.0 for logging."
            )
            cont_loss_val = 0.0

        total_contrastive_loss += cont_loss_val

        # Initialize additional losses
        hnml = torch.tensor(0.0, device=DEVICE)
        hnml_val = 0.0

        # Initialize PromptSRC losses
        ma_loss = torch.tensor(0.0, device=DEVICE)
        td_loss = torch.tensor(0.0, device=DEVICE)
        ma_loss_val = 0.0
        td_loss_val = 0.0

        # If HELIP is enabled, compute additional hard negative margin loss
        if use_helip and hard_pair_miner is not None and lambda_hnml > 0:
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min(batch_start + dataloader.batch_size, len(dataset_indices))
            batch_indices = list(range(batch_start, batch_end))

            # Get hard pairs for current batch samples
            batch_hard_pairs = {}
            for i, orig_idx in enumerate(batch_indices):
                if not hasattr(dataloader.dataset, "indices"):
                    dataset_idx = orig_idx
                else:
                    dataset_idx = dataset_indices[orig_idx]

                dataset_idx_int = (
                    dataset_idx.item() if hasattr(dataset_idx, "item") else dataset_idx
                )

                # Get hard pairs for this index
                batch_hard_pairs[i] = hard_pairs.get(dataset_idx_int, [])

            # Compute image features with gradients for HNML
            # Forward pass through image encoder - KEEP GRADIENTS for backward pass
            image_features = model.image_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # The forward pass already computed and stored text features in model.text_features
            # We can directly use them for HNML
            if model.text_features is not None:
                # The text features shape might be [n_classes, feature_dim] instead of [batch_size, feature_dim]
                # Ensure we only use the first batch_size features if needed
                batch_size = image_features.size(0)
                if model.text_features.size(0) > batch_size:
                    text_features = model.text_features[:batch_size]
                else:
                    text_features = model.text_features

                # Verify shapes match for matrix operations
                if image_features.size(0) != text_features.size(0):
                    LOGGER.warning(
                        f"Shape mismatch in HNML: image_features={image_features.shape}, text_features={text_features.shape}"
                    )
                    # Fall back to just contrastive loss
                    loss = contrastive_loss
                    hnml = torch.tensor(0.0, device=DEVICE)
                    hnml_val = 0.0
                else:
                    # Compute Hard Negative Margin Loss
                    hnml = hard_negative_margin_loss(
                        image_features,
                        text_features,
                        batch_hard_pairs,
                        alpha=hard_pair_miner.alpha,
                    )

                    if isinstance(hnml, torch.Tensor) and hnml.dim() > 0:
                        hnml = hnml.mean()

                    # Ensure we have scalar tensors with gradients for loss calculation
                    if (
                        isinstance(contrastive_loss, torch.Tensor)
                        and not contrastive_loss.requires_grad
                    ):
                        LOGGER.warning(
                            "Contrastive loss does not require gradients - fixing"
                        )
                        contrastive_loss = (
                            contrastive_loss.detach().clone().requires_grad_(True)
                        )

                    # Combined loss
                    loss = contrastive_loss + lambda_hnml * hnml

                    with torch.no_grad():
                        try:
                            hnml_val = (
                                hnml.item()
                                if isinstance(hnml, torch.Tensor) and hnml.numel() == 1
                                else float(hnml)
                            )
                            total_hnml_loss += hnml_val
                        except Exception as e:
                            LOGGER.warning(f"Error converting HNML to scalar: {e}")
                            hnml_val = 0.0
            else:
                # If we couldn't get text features, just use contrastive loss
                loss = contrastive_loss
                hnml = torch.tensor(0.0, device=DEVICE)
                hnml_val = 0.0
        else:
            loss = contrastive_loss

        # compute mutual agreement and textual diversity losses
        if (
            use_promptsrc
            and base_model is not None
            and model.current_image_features is not None
            and model.current_text_features is not None
        ):
            batch_size = images.size(0)

            prompted_image_features = model.current_image_features
            prompted_text_features = model.current_text_features

            # Get original model features
            with torch.no_grad():
                # rozen image features
                original_image_features = base_model.encode_image(
                    images
                )
                original_image_features = (
                    original_image_features
                    / original_image_features.norm(dim=-1, keepdim=True)
                )

                # frozen text features
                original_text_features = base_model.encode_text(
                    tokenizer(
                        [
                            f"a photo of a {CLASS_NAMES[t.item()]}, a type of flower."
                            for t in targets
                        ]
                    ).to(DEVICE)
                )
                original_text_features = (
                    original_text_features
                    / original_text_features.norm(dim=-1, keepdim=True)
                )

            if lambda_ma > 0:
                ma_loss = mutual_agreement_loss(
                    prompted_image_features[:batch_size],
                    prompted_text_features[:batch_size],
                    original_image_features,
                    original_text_features,
                    temperature=temperature,
                )

                try:
                    ma_loss_val = (
                        ma_loss.item()
                        if isinstance(ma_loss, torch.Tensor) and ma_loss.numel() == 1
                        else float(ma_loss)
                    )
                    total_ma_loss += ma_loss_val
                except Exception as e:
                    LOGGER.warning(
                        f"Error converting mutual agreement loss to scalar: {e}"
                    )
                    ma_loss_val = 0.0

            if lambda_td > 0:
                td_loss = textual_diversity_loss(
                    prompted_text_features[:batch_size], temperature=temperature
                )

                # scalar value for logging
                try:
                    td_loss_val = (
                        td_loss.item()
                        if isinstance(td_loss, torch.Tensor) and td_loss.numel() == 1
                        else float(td_loss)
                    )
                    total_td_loss += td_loss_val
                except Exception as e:
                    LOGGER.warning(
                        f"Error converting textual diversity loss to scalar: {e}"
                    )
                    td_loss_val = 0.0

            # Add PromptSRC losses to total loss
            loss = loss + lambda_ma * ma_loss + lambda_td * td_loss
        # Closing PromptSRC

        # Final check to ensure loss is a scalar with gradients before backward
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            LOGGER.warning(f"Final loss has shape {loss.shape} - converting to mean")
            loss = loss.mean()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update EMA model
        if avg_model:
            avg_model.update_parameters(model)

        # loss value for logging
        try:
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        except Exception as e:
            LOGGER.warning(f"Error converting loss to scalar: {e}. Using sum of parts.")
            loss_val = cont_loss_val + hnml_val

        total_loss += loss_val

        if use_promptsrc and use_helip:
            progress_bar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "cont_loss": f"{cont_loss_val:.4f}",
                    "ma_loss": f"{ma_loss_val:.4f}",
                    "td_loss": f"{td_loss_val:.4f}",
                    "hnml": f"{hnml_val:.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

            WANDB.log(
                {
                    "train_loss": loss_val,
                    "contrastive_loss": cont_loss_val,
                    "mutual_agreement_loss": ma_loss_val,
                    "textual_diversity_loss": td_loss_val,
                    "hnml": hnml_val,
                    "train_avg_loss": total_loss / (batch_idx + 1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "lambda_ma": lambda_ma,
                    "lambda_td": lambda_td,
                }
            )
        elif use_promptsrc:
            progress_bar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "cont_loss": f"{cont_loss_val:.4f}",
                    "ma_loss": f"{ma_loss_val:.4f}",
                    "td_loss": f"{td_loss_val:.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

            WANDB.log(
                {
                    "train_loss": loss_val,
                    "contrastive_loss": cont_loss_val,
                    "mutual_agreement_loss": ma_loss_val,
                    "textual_diversity_loss": td_loss_val,
                    "train_avg_loss": total_loss / (batch_idx + 1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "lambda_ma": lambda_ma,
                    "lambda_td": lambda_td,
                }
            )
        elif use_helip:
            progress_bar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "cont_loss": f"{cont_loss_val:.4f}",
                    "hnml": f"{hnml_val:.4f}" if lambda_hnml > 0 else "0.0000",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

            WANDB.log(
                {
                    "train_loss": loss_val,
                    "contrastive_loss": cont_loss_val,
                    "hnml": hnml_val if lambda_hnml > 0 else 0.0,
                    "train_avg_loss": total_loss / (batch_idx + 1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        else:
            progress_bar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

            WANDB.log(
                {
                    "train_loss": loss_val,
                    "train_avg_loss": total_loss / (batch_idx + 1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

    if scheduler is not None:
        scheduler.step()

    # Return average losses
    avg_loss = total_loss / len(dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(dataloader)

    # Log all loss components
    if use_promptsrc and use_helip:
        avg_hnml_loss = total_hnml_loss / len(dataloader) if lambda_hnml > 0 else 0.0
        avg_ma_loss = total_ma_loss / len(dataloader) if lambda_ma > 0 else 0.0
        avg_td_loss = total_td_loss / len(dataloader) if lambda_td > 0 else 0.0

        LOGGER.info(
            f"Epoch {epoch} - Training Loss: {avg_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"MA Loss: {avg_ma_loss:.4f}, "
            f"TD Loss: {avg_td_loss:.4f}, "
            f"HNML: {avg_hnml_loss:.4f}"
        )
    elif use_promptsrc:
        avg_ma_loss = total_ma_loss / len(dataloader) if lambda_ma > 0 else 0.0
        avg_td_loss = total_td_loss / len(dataloader) if lambda_td > 0 else 0.0

        LOGGER.info(
            f"Epoch {epoch} - Training Loss: {avg_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"MA Loss: {avg_ma_loss:.4f}, "
            f"TD Loss: {avg_td_loss:.4f}"
        )
    # End PromptSRC
    elif use_helip:
        avg_hnml_loss = total_hnml_loss / len(dataloader) if lambda_hnml > 0 else 0.0

        LOGGER.info(
            f"Epoch {epoch} - Training Loss: {avg_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"HNML: {avg_hnml_loss:.4f}"
        )
    else:
        LOGGER.info(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
    # End of HELIP

    return avg_loss


@torch.no_grad()
def eval(model, dataset, categories, batch_size, tokenizer, label=""):
    model.eval()

    # Remap labels into a contiguous set starting from zero
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    # here we apply the standard CLIP template used for oxford flowers to all categories
    # and immediately tokenize each sentence (convert natural language into numbers - feel free to print the text input to inspect them)
    text_inputs = tokenizer(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(DEVICE)

    # we can encode the text features once as they are shared for all images
    # therefore we do it outside the evaluation loop
    text_features = model.encode_text(text_inputs)
    # and here we normalize them (standard pratice with CLIP)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # simple dataloader creation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # here we store the number of correct predictions we will make
    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=label):
        # base categories range from 0 to 50, whil novel ones from 51 to 101
        # therefore we must map categories to the [0, 50], otherwise we will have wrong predictions
        # Map targets in contiguous set starting from zero
        # Labels needs to be .long() in pytorch
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long()

        image = image.to(DEVICE)
        target = target.to(DEVICE)

        # forward image through CLIP image encoder
        image_features = model.encode_image(image)
        # and normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # here cosine similarity between image and text features and keep the argmax for every row (every image)
        predicted_class = (image_features @ text_features.T).argmax(dim=-1)
        # now we check which are correct, and sum them (False == 0, True == 1)
        correct_predictions += (predicted_class == target).sum().item()

    # and now we compute the accuracy
    accuracy = correct_predictions / len(dataset)
    return accuracy


def harmonic_mean(base_accuracy, novel_accuracy):
    numerator = 2
    denominator = 1 / base_accuracy + 1 / novel_accuracy
    hm = numerator / denominator
    return hm


@torch.no_grad()
def evaluate(model, dataloader, epoch, split="val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"{split.capitalize()} Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to device
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)

        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.2f}%"}
        )

    # final metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    if split == "val":
        WANDB.log(
            {f"{split}_loss": avg_loss, f"{split}_accuracy": accuracy, "epoch": epoch}
        )

    LOGGER.info(
        f"Epoch {epoch} - {split.capitalize()} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
    return accuracy, avg_loss


# Evaluate model on both base and novel categories and compute harmonic mean
def eval_with_both_categories(
    custom_model,
    base_model,
    test_base_loader,
    test_novel_loader,
    base_classes,
    novel_classes,
    tokenizer,
    epoch,
):
    # Create dataloader for the metrics
    test_base_dataset = test_base_loader.dataset
    test_novel_dataset = test_novel_loader.dataset

    # Evaluate on base and novel categories
    base_accuracy = eval(
        model=custom_model if epoch > 0 else base_model,
        dataset=test_base_dataset,
        categories=base_classes,
        batch_size=128,
        tokenizer=tokenizer,
        label=f"Epoch {epoch} - Base Classes",
    )

    novel_accuracy = eval(
        model=custom_model if epoch > 0 else base_model,
        dataset=test_novel_dataset,
        categories=novel_classes,
        batch_size=128,
        tokenizer=tokenizer,
        label=f"Epoch {epoch} - Novel Classes",
    )

    hm = harmonic_mean(base_accuracy, novel_accuracy)

    LOGGER.info(
        f"Epoch {epoch} - Base Accuracy: {base_accuracy:.4f}, "
        f"Novel Accuracy: {novel_accuracy:.4f}, "
        f"Harmonic Mean: {hm:.4f}"
    )

    WANDB.log(
        {
            "base_accuracy": base_accuracy,
            "novel_accuracy": novel_accuracy,
            "harmonic_mean": hm,
            "epoch": epoch,
        }
    )

    return base_accuracy, novel_accuracy, hm


"""## Define model"""


class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class PromptLearner(torch.nn.Module):
    def __init__(self, classnames, clip_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = CFG["COCOOP"]["prompt_learner"]["n_ctx"]
        ctx_init = CFG["COCOOP"]["prompt_learner"]["ctx_init"]
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = getattr(clip_model.visual, "input_resolution", 224)
        cfg_imsize = CFG["input"]["size"][0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        # Start PromptSRC
        # Initialize prompt ensemble if enabled
        self.prompt_ensemble_enabled = (
            CFG.get("promptsrc", {}).get("prompt_ensemble", {}).get("enabled", False)
        )
        if self.prompt_ensemble_enabled:
            window_size = CFG["promptsrc"]["prompt_ensemble"]["window_size"]
            sigma = CFG["promptsrc"]["prompt_ensemble"]["sigma"]
            self.prompt_ensemble = PromptEnsemble(window_size=window_size, sigma=sigma)
            LOGGER.info(
                f"Initialized PromptSRC ensemble with window_size={window_size}, sigma={sigma}"
            )
        # End PromptSRC

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenizer(ctx_init).to(DEVICE)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=DEVICE)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = torch.nn.Parameter(ctx_vectors)

        self.meta_net = torch.nn.Sequential(
            OrderedDict(
                [
                    ("linear1", torch.nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", torch.nn.ReLU(inplace=True)),
                    ("linear2", torch.nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        ).to(DEVICE)

        classnames = [name.replace("_", " ") for name in classnames]
        # Use tokenizer to encode classnames safely
        name_lens = [len(tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # Move tokenized prompts to the correct device
        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]).to(
            DEVICE
        )  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Start PromptSRC
        # Update prompt ensemble with current context
        if self.prompt_ensemble_enabled and self.training:
            self.prompt_ensemble.update(ctx_shifted)

            # Get ensemble prompt if available
            ensemble_ctx = self.prompt_ensemble.get_ensemble_prompt()
            if ensemble_ctx is not None:
                # Use ensemble context instead of current one during training
                ctx_shifted = ensemble_ctx
        # End PromptSRC

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CustomCLIP(torch.nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, segmentation_model, segmentation_transform):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.segmentation_model = segmentation_model
        self.segmentation_tranforms = segmentation_transform
        # HELIP: Store the text features for later use
        self.text_features = None

        # Start PromptSRC
        # Additional storage for text features at different stages of processing
        self.current_text_features = None
        self.current_image_features = None
        self.classnames = classnames
        # End PromptSRC

    def forward(self, image, label=None):
        if self.segmentation_model:
            seg_input = self.segmentation_tranforms(image) if self.segmentation_tranforms else image
            mask = self.segmentation_model(seg_input)
            binary_mask = (mask > 0.5).int()
            image = binary_mask * image

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Start PromptSRC
        # Store image features for loss computation
        self.current_image_features = image_features
        # End PromptSRC

        prompts = self.prompt_learner(image_features)

        logits = []

        # HELIP: Store all text features
        all_text_features = []

        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_text_features.append(text_features)

            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)

        self.text_features = torch.cat(all_text_features)

        # Save current batch of text features for PromptSRC's loss computation
        self.current_text_features = self.text_features
        # PromptSRC

        if self.prompt_learner.training:
            if label is not None:
                # Check if logits shape is compatible with cross_entropy
                if logits.dim() > 2:
                    # Reshape logits to [batch_size, num_classes]
                    batch_size = image.size(0)
                    logits = logits.view(batch_size, -1)

                # Ensure label shape matches what cross_entropy expects
                if label.dim() > 1:
                    label = label.view(-1)

                # Try to compute cross entropy, with fallback
                try:
                    return F.cross_entropy(logits, label)
                except Exception as e:
                    # Fallback: compute mean of each logit row as a scalar loss
                    LOGGER.warning(f"Cross entropy failed: {e}. Using fallback loss.")
                    return logits.mean()
            else:
                # If no label, return mean of logits as scalar loss during training
                return logits.mean()

        return logits

    # Compute the similarity between image features and text features to make HELIP work
    def compute_similarity(self, image_features):
        # Ensure text features exist
        if self.text_features is None:
            raise ValueError(
                "Text features not computed yet. Call forward method first."
            )

        # Get logit scale
        logit_scale = self.logit_scale.exp()

        # Compute similarity scores
        logits_per_image = logit_scale * (image_features @ self.text_features.T)
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    # Start PromptSRC
    # Use the frozen text encoder to generate text features. Necessary for PromptSRC's mutual agreement loss.
    def get_text_features_for_classes(self, class_indices, tokenizer):
        # Format prompts with template
        prompts = [
            f"a photo of a {self.classnames[idx]}, a type of flower."
            for idx in class_indices
        ]

        text_inputs = tokenizer(prompts).to(DEVICE)

        # Forward through text encoder
        with torch.no_grad():
            text_features = self.text_encoder(text_inputs, text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    # End PromptSRC


def create_base_model():
    result = open_clip.create_model_from_pretrained(
        model_name=CFG["COCOOP"]["base_model"]["name"],
        pretrained=CFG["COCOOP"]["base_model"]["weights"],
        return_transform=True,
    )
    assert isinstance(result, tuple)
    model, preprocess = result
    tokenizer = open_clip.get_tokenizer(CFG["COCOOP"]["base_model"]["name"])
    return model, preprocess, tokenizer

def create_segmentation_model():
    model = smp.create_model(
        arch=CFG["bg-masking"]["backbone"],
        encoder_name=CFG["bg-masking"]["encoder"]["name"],
        classes=1,
        activation="sigmoid",
        encoder_weights=CFG["bg-masking"]["encoder"]["weights"],
    ).to(DEVICE)
    params = smp.encoders.get_preprocessing_params(
        encoder_name=CFG["bg-masking"]["encoder"]["name"], pretrained=CFG["bg-masking"]["encoder"]["weights"]
    )
    normalizer_transform = torchvision.transforms.Normalize(params["mean"], params["std"])
    size_transform = torchvision.transforms.CenterCrop(CFG["bg-masking"]["training"]["image_size"])
    return model, normalizer_transform, size_transform

def create_custom_model(segmentation_model=None, segmentation_transform=None):
    base_model, preprocess, tokenizer = create_base_model()
    model = CustomCLIP(CLASS_NAMES, base_model, tokenizer, segmentation_model, segmentation_transform)
    return (
        model,
        preprocess,
        tokenizer,
    )  # adding tokenizer in line with "self.prompt_learner" definiton


# Define optimizer and LR scheduler.
def get_optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["training"]["optimizer"]["lr"],
        weight_decay=CFG["training"]["optimizer"]["weight_decay"],
    )

    schedulers = []
    milestones = []

    # First, use warmup
    if CFG["training"]["warmup_epochs"] > 0:
        warmup_scheduler = (
            torch.optim.lr_scheduler.LinearLR(  # TODO: Explore better warmups
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=CFG["training"]["warmup_epochs"],
            )
        )
        schedulers.append(warmup_scheduler)
        milestones.append(CFG["training"]["warmup_epochs"])

    # Then use cosine annealing for the main training
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["training"]["epochs"], eta_min=1e-6
    )
    schedulers.append(cosine_scheduler)

    # SequentialLR combines schedulers
    if len(schedulers) > 1:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones
        )
    else:
        scheduler = cosine_scheduler

    return optimizer, scheduler


def fine_tune_segmenter(data_loader, val_data_loader, segmentation_model):
    LOGGER.info("Fine tuning segmenter!")
    epochs = CFG["bg-masking"]["training"]["epochs"]
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
        LOGGER.info(f"Segmentation epoch {epoch} - Segmentation training Loss: {avg_loss:.4f}")

    LOGGER.info("Finished finetuning segmentation model, freezing all parameters")
    for p in segmentation_model.parameters():
        p.requires_grad = False

    LOGGER.info("Testing segmentation model")
    segmentation_model.eval()
    test_loss = 0
    for img, mask in val_data_loader:
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)
        pred = segmentation_model(img)
        test_loss += loss_fn(pred, mask).item()
    test_loss /= len(val_data_loader)
    LOGGER.info(f"Segmentation model avarage test loss: {test_loss:.4f}")


def setup_segmentation_module():
    if CFG["bg-masking"]["enabled"]:
        segmentation_model, normalizer_transform, size_transform = create_segmentation_model()
        segmentation_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), size_transform, normalizer_transform])
        seg_train_set, seg_val_set, _ = get_data(
            train_transform=segmentation_transform,
            test_transform=segmentation_transform,
            mask_transform=size_transform,
            segmentation_training=True,
        )
        base_classes, _ = base_novel_categories(seg_train_set)
        seg_train_base, _ = split_data(seg_train_set, base_classes)
        seg_val_base, _ = split_data(seg_val_set, base_classes)
        seg_train_loader = DataLoader(seg_train_base, batch_size=CFG["bg-masking"]["training"]["batch_size"], shuffle=True)
        seg_val_loader = DataLoader(seg_val_base, batch_size=CFG["bg-masking"]["training"]["batch_size"], shuffle=False)
        fine_tune_segmenter(seg_train_loader, seg_val_loader, segmentation_model)
    else:
        segmentation_model = None
        normalizer_transform = None
    return segmentation_model, normalizer_transform

# DataLoaders for training and evaluation + test_base, and test_novel
def create_data_loaders(
    train_dataset,
    val_dataset,
    test_base_dataset,
    test_novel_dataset,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_base_loader = DataLoader(
        test_base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_novel_loader = DataLoader(
        test_novel_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    LOGGER.info(f"Created dataloaders with batch size {batch_size}")
    return train_loader, val_loader, test_base_loader, test_novel_loader


def save_checkpoint(
    model,
    avg_model,
    optimizer,
    scheduler,
    epoch,
    accuracy,
    best_acc,
    checkpoint_dir,
    is_best=False,
):
    checkpoint = {
        "model": model.state_dict(),
        "avg_model": avg_model.state_dict() if avg_model else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_acc,
    }

    # checkpoint_path = RUN_DIR + "checkpoint/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    LOGGER.info(f"Saved checkpoint at {checkpoint_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(checkpoint, best_path)
        LOGGER.info(f"Saved best model with accuracy {accuracy:.4f} at {best_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    try:
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler and checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])

        epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_accuracy", 0.0)

        LOGGER.info(
            f"Loaded checkpoint from epoch {epoch} with accuracy {checkpoint.get('accuracy', 0.0):.4f}"
        )
        return epoch, best_accuracy
    except Exception as e:
        LOGGER.error(f"Error loading checkpoint: {e}")
        return 0, 0.0


def train_cocoop():
    segmentation_model, segmentation_transform = setup_segmentation_module()

    # Check if HELIP is enabled
    helip_enabled = CFG.get("helip", {}).get("enabled", False)

    # Start PromptSRC
    # Check if PromptSRC is enabled
    promptsrc_enabled = CFG.get("promptsrc", {}).get("enabled", False)

    if promptsrc_enabled and helip_enabled:
        LOGGER.info("==== Starting CoCoOp Training with PromptSRC and HELIP ====")
    elif promptsrc_enabled:
        LOGGER.info("==== Starting CoCoOp Training with PromptSRC ====")
    # End PromptSRC
    elif helip_enabled:
        LOGGER.info("==== Starting CoCoOp Training with HELIP ====")
    else:
        LOGGER.info("==== Starting CoCoOp Training ====")

    LOGGER.info(f"Config: {CFG}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    LOGGER.info("Creating models...")
    base_model, base_preprocess, tokenizer = create_base_model()
    custom_model, _, _ = create_custom_model(segmentation_model, segmentation_transform)
    custom_model = custom_model.to(DEVICE)

    avg_model = None
    if CFG["validation"]["ema"]["enabled"]:
        avg_model = AveragedModel(
            custom_model,
            multi_avg_fn=get_ema_multi_avg_fn(CFG["validation"]["ema"]["decay"]),
        )
        LOGGER.info(f"Created EMA model with decay {CFG['validation']['ema']['decay']}")

    # For PromptSRC, keep base model on device for mutual agreement loss
    if promptsrc_enabled:
        base_model = base_model.to(DEVICE)
        base_model.eval()  # Keep base model in eval mode
        LOGGER.info("Base model loaded for PromptSRC mutual agreement")
    # End PromptSRC

    custom_model, _, _ = create_custom_model(segmentation_model, segmentation_transform)
    custom_model = custom_model.to(DEVICE)

    # Create data augmentation for training
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4
            ),
            base_preprocess,
        ]
    )

    # Get datasets with transforms
    LOGGER.info("Loading datasets...")
    train_set, val_set, test_set = get_data(
        data_dir=CFG["data"]["data_dir"],
        train_transform=train_transform,
        test_transform=base_preprocess,
    )

    # Split base and novel classes
    base_classes, novel_classes = base_novel_categories(train_set)

    # Split datasets
    train_base, _ = split_data(train_set, base_classes)
    val_base, _ = split_data(val_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)

    # Create dataloaders
    LOGGER.info("Creating dataloaders...")
    train_loader, val_loader, test_base_loader, test_novel_loader = create_data_loaders(
        train_base,
        val_base,
        test_base,
        test_novel,
        batch_size=CFG["training"]["batch_size"],
        num_workers=CFG["data"]["num_workers"],
        pin_memory=CFG["data"]["pin_memory"],
    )

    # Create optimizer and scheduler
    LOGGER.info("Setting up optimizer and scheduler...")
    optimizer, scheduler = get_optimizer_and_scheduler(custom_model)

    # Initialize HELIP if enabled
    hard_pair_miner = None
    if helip_enabled:
        LOGGER.info("HELIP is enabled for training")
        # Initialize hard pair mining for base classes only
        hard_pair_miner = HardPairMining(
            custom_clip_model=custom_model, dataset=train_base
        )
    else:
        LOGGER.info("HELIP is disabled, using standard training")

    # Extract PromptSRC parameters if enabled
    if promptsrc_enabled:
        LOGGER.info("PromptSRC is enabled for training")
        mutual_agreement_cfg = CFG["promptsrc"]["mutual_agreement"]
        textual_diversity_cfg = CFG["promptsrc"]["textual_diversity"]

        lambda_ma = mutual_agreement_cfg["lambda_ma"]
        lambda_td = textual_diversity_cfg["lambda_td"]
        temperature = mutual_agreement_cfg["temperature"]

        LOGGER.info(
            f"PromptSRC parameters: lambda_ma={lambda_ma}, lambda_td={lambda_td}, temperature={temperature}"
        )
    # End PromptSRC

    # Training setup
    start_epoch = 0
    best_accuracy = 0.0
    best_base_acc = 0.0
    best_novel_acc = 0.0
    best_hm = 0.0
    patience_counter = 0

    # Check for resume training
    checkpoint_path = os.path.join(CFG["training"]["checkpoint_dir"], "model_best.pth")
    if os.path.exists(checkpoint_path):
        LOGGER.info(f"Found checkpoint at {checkpoint_path}, resuming training...")
        start_epoch, best_accuracy = load_checkpoint(
            custom_model, optimizer, scheduler, checkpoint_path
        )
        start_epoch += 1  # Start from the next epoch

    # Evaluate zero-shot performance of the base model (as a baseline, before any training)
    LOGGER.info("Evaluating zero-shot performance of base model...")
    base_zero_shot_acc, novel_zero_shot_acc, zero_shot_hm = eval_with_both_categories(
        custom_model=custom_model,
        base_model=base_model,
        test_base_loader=test_base_loader,
        test_novel_loader=test_novel_loader,
        base_classes=base_classes,
        novel_classes=novel_classes,
        tokenizer=tokenizer,
        epoch=0,
    )

    LOGGER.info(
        f"Zero-shot: Base Acc={base_zero_shot_acc:.4f}, "
        f"Novel Acc={novel_zero_shot_acc:.4f}, "
        f"HM={zero_shot_hm:.4f}"
    )

    # Training loop
    LOGGER.info("Starting training...")
    for epoch in range(start_epoch, CFG["training"]["epochs"]):
        try:
            _ = train_loop(
                dataloader=train_loader,
                model=custom_model,
                avg_model=avg_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                scheduler=scheduler,
                use_helip=helip_enabled,
                hard_pair_miner=hard_pair_miner,
                lambda_hnml=CFG["helip"].get("lambda_hnml", 0.5)
                if helip_enabled
                else 0.0,
                # PromptSRC
                use_promptsrc=promptsrc_enabled,
                base_model=base_model if promptsrc_enabled else None,
                tokenizer=tokenizer if promptsrc_enabled else None,
                lambda_ma=mutual_agreement_cfg["lambda_ma"]
                if promptsrc_enabled
                else 0.0,
                lambda_td=textual_diversity_cfg["lambda_td"]
                if promptsrc_enabled
                else 0.0,
                temperature=mutual_agreement_cfg["temperature"]
                if promptsrc_enabled
                else 0.07,
                max_epochs=CFG["training"]["epochs"],
                # End PromptSRC
            )

            # Evaluate on validation set
            val_acc, val_loss = evaluate(
                model=avg_model if avg_model else custom_model,
                dataloader=val_loader,
                epoch=epoch + 1,
                split="val",
            )

            # Evaluate on test sets (base and novel)
            base_acc, novel_acc, hm = eval_with_both_categories(
                custom_model=custom_model,
                base_model=base_model,
                test_base_loader=test_base_loader,
                test_novel_loader=test_novel_loader,
                base_classes=base_classes,
                novel_classes=novel_classes,
                tokenizer=tokenizer,
                epoch=epoch + 1,
            )

            # Check if this is the best model (by harmonic mean)
            is_best = hm > best_hm
            if is_best:
                best_hm = hm
                best_base_acc = base_acc
                best_novel_acc = novel_acc
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoints
            save_checkpoint(
                model=custom_model,
                avg_model=avg_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                accuracy=hm,
                best_acc=best_hm,
                checkpoint_dir=CFG["training"]["checkpoint_dir"],
                is_best=is_best,
            )

            # Early stopping
            if patience_counter >= CFG["training"]["patience"]:
                LOGGER.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

        except Exception as e:
            LOGGER.error(f"Error during training: {e}")
            # Save emergency checkpoint
            save_checkpoint(
                model=custom_model,
                avg_model=avg_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                accuracy=best_hm,
                best_acc=best_hm,
                checkpoint_dir=CFG["training"]["checkpoint_dir"],
                is_best=False,
            )
            raise e

    # Load best model for final evaluation
    best_model_path = os.path.join(CFG["training"]["checkpoint_dir"], "model_best.pth")
    if os.path.exists(best_model_path):
        _, _ = load_checkpoint(custom_model, optimizer, None, best_model_path)

    # Final evaluation
    LOGGER.info("==== Final Evaluation ====")
    final_base_acc, final_novel_acc, final_hm = eval_with_both_categories(
        custom_model=custom_model,
        base_model=base_model,
        test_base_loader=test_base_loader,
        test_novel_loader=test_novel_loader,
        base_classes=base_classes,
        novel_classes=novel_classes,
        tokenizer=tokenizer,
        epoch=CFG["training"]["epochs"] + 1,
    )

    LOGGER.info(
        f"Final: Base Acc={final_base_acc:.4f}, "
        f"Novel Acc={final_novel_acc:.4f}, "
        f"HM={final_hm:.4f}"
    )

    # Log improvement over zero-shot
    LOGGER.info(
        f"Improvement over zero-shot: "
        f"Base: {(final_base_acc - base_zero_shot_acc) * 100:.2f}%, "
        f"Novel: {(final_novel_acc - novel_zero_shot_acc) * 100:.2f}%, "
        f"HM: {(final_hm - zero_shot_hm) * 100:.2f}%"
    )

    # Return best results
    return best_base_acc, best_novel_acc, best_hm


def main():
    try:
        # Initial zero-shot evaluation with the base model
        base_model, base_preprocess, base_tokenizer = create_base_model()

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
                base_preprocess,
            ]
        )

        train_set, val_set, test_set = get_data(
            data_dir=CFG["data"]["data_dir"],
            train_transform=train_transform,
            test_transform=base_preprocess,
        )

        base_classes, novel_classes = base_novel_categories(train_set)

        train_base, _ = split_data(train_set, base_classes)
        val_base, _ = split_data(val_set, base_classes)
        test_base, test_novel = split_data(test_set, base_classes)

        # Initial zero-shot evaluation
        LOGGER.info("Running initial zero-shot evaluation...")
        base_accuracy = eval(
            model=base_model,
            dataset=test_base,
            categories=base_classes,
            batch_size=128,
            tokenizer=base_tokenizer,
            label="🧠 Zero-shot evaluation on Base Classes",
        )
        novel_accuracy = eval(
            model=base_model,
            dataset=test_novel,
            categories=novel_classes,
            batch_size=128,
            tokenizer=base_tokenizer,
            label="🧠 Zero-shot evaluation on Novel Classes",
        )

        hm = harmonic_mean(base_accuracy, novel_accuracy)

        LOGGER.info("==== Zero-shot Evaluation Results ====")
        LOGGER.info(f"🔍 Base classes accuracy: {base_accuracy * 100:.2f}%")
        LOGGER.info(f"🔍 Novel classes accuracy: {novel_accuracy * 100:.2f}%")
        LOGGER.info(f"🔍 Harmonic Mean: {hm * 100:.2f}%")

        # Run training with HELIP configured via config
        LOGGER.info("Starting training...")
        best_base_acc, best_novel_acc, best_hm = train_cocoop()

        # Final comparison
        LOGGER.info("==== Improvement Summary ====")
        LOGGER.info(
            f"Base classes: {base_accuracy * 100:.2f}% -> {best_base_acc * 100:.2f}% "
            f"(+{(best_base_acc - base_accuracy) * 100:.2f}%)"
        )
        LOGGER.info(
            f"Novel classes: {novel_accuracy * 100:.2f}% -> {best_novel_acc * 100:.2f}% "
            f"(+{(best_novel_acc - novel_accuracy) * 100:.2f}%)"
        )
        LOGGER.info(
            f"Harmonic Mean: {hm * 100:.2f}% -> {best_hm * 100:.2f}% "
            f"(+{(best_hm - hm) * 100:.2f}%)"
        )

    except Exception as e:
        LOGGER.error(f"Error in main execution: {e}")
        raise e
    finally:
        LOGGER.info("Finishing wandb run...")
        WANDB.finish()


if __name__ == "__main__":
    main()
