"""Run CIFAR-10 inference on a single image using a saved CNN checkpoint."""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from train import CLASS_NAMES, MEAN, STD, SmallCifarCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--checkpoint", default="artifacts/cifar10-cnn/best_model.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SmallCifarCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    with Image.open(args.image) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    score, index = probabilities.max(dim=0)
    print(f"{CLASS_NAMES[int(index)]}: {float(score):.4f}")
    print("Top 3:")
    for class_index in probabilities.topk(3).indices.tolist():
        print(f"- {CLASS_NAMES[class_index]}: {float(probabilities[class_index]):.4f}")


if __name__ == "__main__":
    main()
