# !pip3 install torch==2.6.0 torchvision==0.21.0 open_clip_torch==2.32.0 wandb==0.19.10 scipy==1.15.3 segmentation-models-pytorch==0.5.0
import os
from collections import OrderedDict
from pathlib import Path
import re
from typing import Any, Callable, Optional, Tuple, Union
import torch
import torchvision
import open_clip
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import logging
from datetime import datetime
import PIL.Image
import segmentation_models_pytorch as smp

CFG = {
    "COCOOP": {
        "base_model": {
            "name": "ViT-B-32",
            "weights": "laion2b_s34b_b79k",
        },
        "prompt_learner": {
            "n_ctx": 8,
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
        "patience": 5,
        "epochs": 10,
        "batch_size": 32,
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 0.05,
        },
        "scheduler": {
            "warmup_epochs": 1,
        },
    },
    "validation": {
        "ema": {
            "enabled": True,
            "decay": 0.999,
        }
    },
}

RUN_ID = CFG["training"]["resume_id"] if CFG["training"]["resume_id"] else datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
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

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_base_dataset,
    test_novel_dataset,
    batch_size,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_base_loader = DataLoader(
        test_base_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_novel_loader = DataLoader(
        test_novel_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    LOGGER.info(f"Created dataloaders with batch size {batch_size}")
    return train_loader, val_loader, test_base_loader, test_novel_loader


def harmonic_mean(base_accuracy, novel_accuracy):
    try:
        numerator = 2
        denominator = 1 / base_accuracy + 1 / novel_accuracy
        hm = numerator / denominator
    except ZeroDivisionError:
        return 0
    return hm


@torch.no_grad()
def evaluate(model, dataloader, epoch=0, split="val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"{split.capitalize()} Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.2f}%"})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    if split == "val":
        WANDB.log({f"{split}_loss": avg_loss, f"{split}_accuracy": accuracy, "epoch": epoch})

        LOGGER.info(f"Epoch {epoch} - {split.capitalize()} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy, avg_loss


def eval_with_both_categories(custom_model, test_base_loader, test_novel_loader, base_classes, novel_classes, epoch):
    base_accuracy, _ = evaluate(model=custom_model, dataloader=test_base_loader, split="test")

    novel_accuracy, _ = evaluate(model=custom_model, dataloader=test_novel_loader, split="test")

    hm = harmonic_mean(base_accuracy, novel_accuracy)

    LOGGER.info(
        f"Epoch {epoch} - Base Accuracy: {base_accuracy:.4f}, "
        f"Novel Accuracy: {novel_accuracy:.4f}, "
        f"Harmonic Mean: {hm:.4f}"
    )

    WANDB.log({"base_accuracy": base_accuracy, "novel_accuracy": novel_accuracy, "harmonic_mean": hm, "epoch": epoch})

    return base_accuracy, novel_accuracy, hm


def save_checkpoint(model, avg_model, optimizer, scheduler, epoch, accuracy, best_acc, is_best=False):
    checkpoint = {
        "model": model.state_dict(),
        "avg_model": avg_model.state_dict() if avg_model else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_acc,
    }

    checkpoint_dir = RUN_DIR + "checkpoint/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_dir + "checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    LOGGER.info(f"Saved checkpoint at {checkpoint_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(checkpoint, best_path)
        LOGGER.info(f"Saved best model with accuracy {accuracy:.4f} at {best_path}")


def load_checkpoint(model, avg_model, optimizer, scheduler, resume_from="best"):
    checkpoint_dir = RUN_DIR + "checkpoint/"
    if resume_from == "best":
        checkpoint_path = checkpoint_dir + "model_best.pth"
    else:
        checkpoint_path = checkpoint_dir + f"checkpoint_epoch_{resume_from}.pth"
    LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if avg_model and checkpoint["avg_model"]:
        avg_model.load_state_dict(checkpoint["avg_model"])

    epoch = checkpoint["epoch"]
    best_accuracy = checkpoint.get("best_accuracy", 0.0)

    LOGGER.info(f"Loaded checkpoint from epoch {epoch} with accuracy {checkpoint.get('accuracy', 0.0):.4f}")
    return epoch, best_accuracy


class TextEncoder(torch.nn.Module):
    def __init__(self, cfg, clip_model):
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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(torch.nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg["n_ctx"]
        ctx_init = cfg["ctx_init"]
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

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
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=DEVICE)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

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
        name_lens = [len(clip_model.encode_text(tokenizer(name).to(DEVICE))) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]).to(DEVICE)  # (n_cls, n_tkn)
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

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CustomCLIP(torch.nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer, segmentation_model, segmentation_transform):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg["prompt_learner"], classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg["text_encoder"], clip_model)
        self.logit_scale = clip_model.logit_scale
        self.segmentation_model = segmentation_model
        self.segmentation_tranforms = segmentation_transform

    def forward(self, image, label=None):
        if self.segmentation_model:
            seg_input = self.segmentation_tranforms(image) if self.segmentation_tranforms else image
            mask = self.segmentation_model(seg_input)
            binary_mask = (mask > 0.5).int()
            import pdb; pdb.set_trace()
            image = binary_mask * image

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            assert label is not None
            return torch.nn.functional.cross_entropy(logits, label)

        return logits


def create_base_model(cfg):
    result = open_clip.create_model_from_pretrained(
        model_name=cfg["name"],
        pretrained=cfg["weights"],
        device=DEVICE,
        return_transform=True,
    )
    assert isinstance(result, tuple)
    model, preprocess = result
    tokenizer = open_clip.get_tokenizer(cfg["name"])
    return model, preprocess, tokenizer


def create_segmentation_model(cfg):
    model = smp.create_model(
        arch=cfg["backbone"],
        encoder_name=cfg["encoder"]["name"],
        classes=1,
        activation="sigmoid",
        encoder_weights=cfg["encoder"]["weights"],
    ).to(DEVICE)
    params = smp.encoders.get_preprocessing_params(
        encoder_name=cfg["encoder"]["name"], pretrained=cfg["encoder"]["weights"]
    )
    normalizer_transform = torchvision.transforms.Normalize(params["mean"], params["std"])
    size_transform = torchvision.transforms.CenterCrop(cfg["training"]["image_size"])
    return model, normalizer_transform, size_transform


def create_custom_model(cfg, segmentation_model=None, segmentation_transform=None):
    base_model, preprocess, tokenizer = create_base_model(cfg["base_model"])
    model = CustomCLIP(cfg, CLASS_NAMES, base_model, tokenizer, segmentation_model, segmentation_transform)
    return model, preprocess, tokenizer


def get_optimizer(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return optimizer


def get_scheduler(optimizer, cfg):
    if cfg["warmup_epochs"] > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg["warmup_epochs"]
        )

    return scheduler


def fine_tune_segmenter(cfg, data_loader, val_data_loader, segmentation_model):
    LOGGER.info("Fine tuning segmenter!")
    epochs = cfg["training"]["epochs"]
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


def setup_segmentation_module(cfg):
    if cfg["enabled"]:
        segmentation_model, normalizer_transform, size_transform = create_segmentation_model(cfg)
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
        seg_train_loader = DataLoader(seg_train_base, batch_size=cfg["training"]["batch_size"], shuffle=True)
        seg_val_loader = DataLoader(seg_val_base, batch_size=cfg["training"]["batch_size"], shuffle=False)
        fine_tune_segmenter(cfg, seg_train_loader, seg_val_loader, segmentation_model)
    else:
        segmentation_model = None
        normalizer_transform = None
    return segmentation_model, normalizer_transform


def train_loop(dataloader, model, avg_model, optimizer, epoch, scheduler=None):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        loss = model(images, targets)
        loss.backward()
        optimizer.step()
        if avg_model:
            avg_model.update_parameters(model)
        total_loss += loss.item()

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
            }
        )

        WANDB.log(
            {
                "train_loss": loss.item(),
                "train_avg_loss": total_loss / (batch_idx + 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    # Step the scheduler if provided
    if scheduler is not None:
        scheduler.step()

    # Return average loss
    avg_loss = total_loss / len(dataloader)
    LOGGER.info(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
    return avg_loss


############################################### END OF DEFININTIONS ###################################################


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    segmentation_model, segmentation_transform = setup_segmentation_module(CFG["bg-masking"])

    custom_model, custom_preprocess, custom_tokenizer = create_custom_model(
        CFG["COCOOP"], segmentation_model, segmentation_transform
    )
    avg_model = (
        AveragedModel(custom_model, multi_avg_fn=get_ema_multi_avg_fn(CFG["validation"]["ema"]["decay"]))
        if CFG["validation"]["ema"]["enabled"]
        else None
    )

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            custom_preprocess,
        ]
    )

    LOGGER.info("Loading datasets...")
    train_set, val_set, test_set = get_data(train_transform=train_transform, test_transform=custom_preprocess)

    base_classes, novel_classes = base_novel_categories(train_set)

    train_base, _ = split_data(train_set, base_classes)
    val_base, _ = split_data(val_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)

    LOGGER.info("Creating dataloaders...")
    train_loader, val_loader, test_base_loader, test_novel_loader = create_data_loaders(
        train_base,
        val_base,
        test_base,
        test_novel,
        batch_size=CFG["training"]["batch_size"],
    )

    optimizer = get_optimizer(custom_model, CFG["training"]["optimizer"])
    scheduler = get_scheduler(optimizer, CFG["training"]["scheduler"])

    LOGGER.info("Starting CoCoOp training...")
    if CFG["training"]["resume_id"]:
        start_epoch, best_acc = load_checkpoint(custom_model, avg_model, optimizer, scheduler)
        start_epoch += 1  # Start from the next epoch
    else:
        start_epoch = 0
        best_acc = 0.0

    # Training setup
    patience_counter = 0

    LOGGER.info("Starting training...")
    for epoch in range(start_epoch, CFG["training"]["epochs"]):
        train_loop(
            dataloader=train_loader,
            model=custom_model,
            avg_model=avg_model,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
        )

        val_acc, val_loss = evaluate(
            model=avg_model if avg_model else custom_model, dataloader=val_loader, epoch=epoch, split="val"
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            model=custom_model,
            avg_model=avg_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            accuracy=val_acc,
            best_acc=best_acc,
            is_best=is_best,
        )

        # Early stopping
        if patience_counter >= CFG["training"]["patience"]:
            LOGGER.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

    # Load best model for final evaluation
    load_checkpoint(custom_model, avg_model, optimizer, None, resume_from="best")

    # Final evaluation
    LOGGER.info("==== Final Evaluation ====")
    final_base_acc, final_novel_acc, final_hm = eval_with_both_categories(
        custom_model=avg_model if avg_model else custom_model,
        test_base_loader=test_base_loader,
        test_novel_loader=test_novel_loader,
        base_classes=base_classes,
        novel_classes=novel_classes,
        epoch="END",
    )

    LOGGER.info(f"Final: Base Acc={final_base_acc:.4f}, Novel Acc={final_novel_acc:.4f}, HM={final_hm:.4f}")
    WANDB.finish()


if __name__ == "__main__":
    main()
