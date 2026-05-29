import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "root": "data/cifar10",
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
        "hub_model": "cifar10_resnet20",
    },
    "cifar100": {
        "num_classes": 100,
        "root": "data/cifar100",
        "mean": [0.5070, 0.4865, 0.4409],
        "std": [0.2673, 0.2564, 0.2761],
        "hub_model": "cifar100_resnet20",
    },
}


def build_dataloader(dataset_name: str, batch_size: int, num_workers: int):
    cfg = CIFAR_CONFIGS[dataset_name]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg["root"],
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=cfg["root"],
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return loader


def load_model(dataset_name: str, model_name: str | None, device: torch.device):
    cfg = CIFAR_CONFIGS[dataset_name]

    if model_name is None:
        model_name = cfg["hub_model"]

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
    )

    model = model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)

        _, pred_top1 = logits.max(dim=1)
        correct_top1 += pred_top1.eq(targets).sum().item()

        k = min(5, logits.size(1))
        _, pred_top5 = logits.topk(k, dim=1)
        correct_top5 += pred_top5.eq(targets.view(-1, 1)).sum().item()

        total += targets.size(0)

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total

    return top1, top5


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Torch Hub model name. If not provided, uses cifar10_resnet20 or cifar100_resnet20.",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this on a GPU machine.")

    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = True

    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")

    loader = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_model(
        dataset_name=args.dataset,
        model_name=args.model,
        device=device,
    )

    top1, top5 = evaluate(model, loader, device)

    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")


if __name__ == "__main__":
    main()