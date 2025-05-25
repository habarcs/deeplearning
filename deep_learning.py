# !pip3 install torch==2.6.0 torchvision==0.21.0 open_clip_torch==2.32.0 wandb==0.19.10 scipy==1.15.3
import os
from collections import OrderedDict
import torch
import torchvision
import open_clip
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import logging
from datetime import datetime

CFG = {
    "COCOOP": {
        "base_model": {
            "name": "ViT-B-32",
            "weights": "laion2b_s34b_b79k",
        },
        "prompt_learner": {
            "n_ctx": 8,
            "ctx_init": "", 
            "class_token_position": "end",  # "beg" for beginning, "mid" for middle, "end" for end
            "tokenizer": "tokenizer"
    
        },
        "text_encoder": {},
    },
    "wandb": True,
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


def get_data(data_dir="./data", train_transform=None, test_transform=None):
    LOGGER.info(f"Loading Flowers102 dataset from {data_dir}")

    train = torchvision.datasets.Flowers102(root=data_dir, split="train", download=True, transform=train_transform)
    val = torchvision.datasets.Flowers102(root=data_dir, split="val", download=True, transform=test_transform)
    test = torchvision.datasets.Flowers102(root=data_dir, split="test", download=True, transform=test_transform)

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
    def __init__(self, clip_model, classnames, n_ctx, ctx_init, class_token_position, tokenizer, csc=False):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Use given words to initialize context vectors
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenizer(ctx_init).to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)

            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Initial context: '{prompt_prefix}'")
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = torch.nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(tokenizer(name)[0]) - 2 for name in classnames]  # exclude SOS and EOS
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]).to(clip_model.token_embedding.weight.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = class_token_position

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

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
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
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
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Invalid class_token_position: {self.class_token_position}")

        return prompts



class CustomCLIP(torch.nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = PromptLearner(
                                clip_model=clip_model,
                                classnames=classnames,
                                n_ctx=cfg["prompt_learner"]["n_ctx"],
                                ctx_init=cfg["prompt_learner"]["ctx_init"],
                               class_token_position=cfg["prompt_learner"]["class_token_position"],
                                tokenizer=tokenizer
                            ) 

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg["text_encoder"], clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, label=None):
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


def create_custom_model(cfg):
    base_model, preprocess, tokenizer = create_base_model(cfg["base_model"])
    model = CustomCLIP(cfg, CLASS_NAMES, base_model, tokenizer)
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
    custom_model, custom_preprocess, custom_tokenizer = create_custom_model(CFG["COCOOP"])
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

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
