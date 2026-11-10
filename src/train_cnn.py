"""Train a CNN (ResNet) via transfer learning for binary real-vs-fake classification."""
import argparse
from pathlib import Path
import sys
import torch
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.models as models
import time
import numpy as np


def build_transforms(img_size=224, train=True):
    if train:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='root processed dataset with train/val/test subfolders')
    parser.add_argument('--arch', choices=['resnet18','resnet50'], default='resnet18')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', default='outputs/cnn')
    args = parser.parse_args()

    data_root = Path(args.data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = ImageFolder(data_root / 'train', transform=build_transforms(args.img_size, train=True))
    val_ds = ImageFolder(data_root / 'val', transform=build_transforms(args.img_size, train=False))
    test_ds = ImageFolder(data_root / 'test', transform=build_transforms(args.img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = model.fc.in_features
        model.fc = nn.Linear(feat_dim, len(train_ds.classes))
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = model.fc.in_features
        model.fc = nn.Linear(feat_dim, len(train_ds.classes))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        t1 = time.time()
        print(f'Epoch {epoch} Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} | {t1-t0:.1f}s')
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'classes': train_ds.classes}, out_dir / 'best.pth')

    # final test
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print('Test acc', test_acc, 'loss', test_loss)
    torch.save({'model_state': model.state_dict(), 'classes': train_ds.classes}, out_dir / 'final.pth')


if __name__ == '__main__':
    main()
