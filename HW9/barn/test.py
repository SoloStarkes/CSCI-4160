import sys
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from dataset import BuildingDataset
from model import CNN

# Image dimensions from the assignment
IMAGE_HEIGHT = 189
IMAGE_WIDTH = 252

# IMPORTANT: this must match whatever you used in train_buildings.py
RESIZE_FACTOR = 1  # if you changed this in training, change it here too

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_loader(img_dir, labels_path, batch_size=100):
    """
    Build a DataLoader for the test set.
    """
    new_h = int(IMAGE_HEIGHT / RESIZE_FACTOR)
    new_w = int(IMAGE_WIDTH / RESIZE_FACTOR)

    # Same preprocessing as in train_buildings.py
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.225, 0.225, 0.225],
    )
    resize = torchvision.transforms.Resize(size=(new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)

    test_transforms = torchvision.transforms.Compose(
        [resize, convert, normalize]
    )

    test_dataset = BuildingDataset(
        annotations_file=labels_path,
        img_dir=img_dir,
        transform=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader, (3, new_h, new_w)


def test(net, loader, device):
    """
    Evaluate the network on the given loader and print accuracy.
    """
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            logits = net(data)
            # if you care about loss, you can compute log_probs + NLLLoss here
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test accuracy: {correct}/{total} ({acc:.2f}%)")
    return acc


def main():
    if len(sys.argv) != 3:
        print("Usage: python test.py [test_set_directory] [labels_csv_path]")
        sys.exit(1)

    test_img_dir = sys.argv[1]
    labels_path = sys.argv[2]

    if not os.path.isdir(test_img_dir):
        print(f"Error: directory does not exist: {test_img_dir}")
        sys.exit(1)
    if not os.path.isfile(labels_path):
        print(f"Error: labels file does not exist: {labels_path}")
        sys.exit(1)

    # Build loader and infer input dimension for CNN
    test_loader, input_dim = build_test_loader(test_img_dir, labels_path)

    # out_dim = 11 for the 11 campus buildings
    out_dim = 11
    model = CNN(in_dim=input_dim, out_dim=out_dim).to(device)

    # Adjust this path if you saved the model somewhere else
    model_path = os.path.join("models", "cnn_buildings.pth")
    if not os.path.isfile(model_path):
        print(f"Error: model file not found at {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Run evaluation
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
