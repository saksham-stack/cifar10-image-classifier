"""
CIFAR-10 Image Classifier (CNN from scratch, PyTorch)

Install:   pip install torch torchvision matplotlib scikit-learn pillow
Train:     python cifar10_classifier.py train --epochs 15
Evaluate:  python cifar10_classifier.py eval
Predict:   python cifar10_classifier.py predict --image my_photo.jpg

On CPU expect roughly 10-20 minutes. Expected test accuracy: ~85-88%.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
MODEL_PATH = "cifar10_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
def get_loaders(batch_size=128):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_set = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


# --------------------------------------------------------------- model
def conv_block(in_ch, out_ch, drop):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(drop),
    )


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64, 0.1),
            conv_block(64, 128, 0.2),
            conv_block(128, 256, 0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------------------------------------------ training
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def train(epochs):
    train_loader, test_loader = get_loaders()
    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_loader))

    history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    print(f"Training on {DEVICE} for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        tr_loss, tr_acc = loss_sum / total, correct / total
        te_loss, te_acc = evaluate(model, test_loader)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"Epoch {epoch:2d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| test loss {te_loss:.3f} acc {te_acc:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    plot_history(history)


def plot_history(h):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["train_acc"], label="train")
    ax[0].plot(h["test_acc"], label="test")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["train_loss"], label="train")
    ax[1].plot(h["test_loss"], label="test")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")


# ---------------------------------------------------------- evaluation
def load_model():
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def full_eval():
    _, test_loader = get_loaders()
    model = load_model()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    print(classification_report(labels, preds, target_names=CLASSES, digits=3))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10), CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


# ---------------------------------------------------------- prediction
def predict(image_path):
    model = load_model()
    tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, STD)])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()

    top3 = np.argsort(probs)[::-1][:3]
    print("Top predictions:")
    for i in top3:
        print(f"  {CLASSES[i]:<12} {probs[i] * 100:5.1f}%")

    plt.figure(figsize=(8, 3.5))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.axis("off"); plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.barh(CLASSES, probs); plt.gca().invert_yaxis(); plt.title("Class probabilities")
    plt.tight_layout()
    plt.savefig("prediction.png", dpi=150)
    print("Saved prediction.png")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "predict"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--image", type=str, help="path to an image (predict mode)")
    args = p.parse_args()

    if args.mode == "train":
        train(args.epochs)
    elif args.mode == "eval":
        full_eval()
    else:
        if not args.image:
            raise SystemExit("Please provide --image for predict mode")
        predict(args.image)
