"""Extract feature embeddings using a pretrained model (DINO-capable via timm).

Usage:
    python src/extract_embeddings.py --input data/raw --output data/embeddings.npz --model vit_small_patch16_224
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True  # skip truncated instead of crashing
Image.MAX_IMAGE_PIXELS = None  # suppress decompression bomb error; we cap manually
import timm
from tqdm import tqdm


class ImageListDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p)
        # cap very large images before converting to avoid huge memory spikes
        if max(img.size) > 2048:
            img.thumbnail((2048, 2048), Image.LANCZOS)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        # infer label from parent folder name if present
        label = p.parent.name
        return img, str(p), label


def build_transform(img_size=224):
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def extract(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_path = Path(args.input)
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for p in patterns:
        files.extend(list(in_path.rglob(p)))
    files = sorted(set(files))
    print(f'Found {len(files)} images under {in_path}')
    if len(files) == 0:
        raise SystemExit(f'No image files found under {in_path}. Check the path and file extensions.')
    if args.limit:
        # sample evenly across classes (parent folder = class label)
        from collections import defaultdict
        import random
        by_class = defaultdict(list)
        for f in files:
            by_class[f.parent.name].append(f)
        per_class = args.limit // len(by_class)
        sampled = []
        for cls_files in by_class.values():
            random.shuffle(cls_files)
            sampled.extend(cls_files[:per_class])
        files = sampled

    transform = build_transform(args.img_size)
    ds = ImageListDataset(files, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f'Creating model {args.model_name} (pretrained={args.pretrained})')
    model = timm.create_model(args.model_name, pretrained=args.pretrained)
    model.eval()
    model.to(device)

    embeddings = []
    fnames = []
    labels = []

    with torch.no_grad():
        for imgs, paths, labs in tqdm(dl, desc='Extracting'):
            imgs = imgs.to(device)
            # try forward_features first (works for ViT/DINO models in timm)
            if hasattr(model, 'forward_features'):
                feats = model.forward_features(imgs)
            else:
                # remove classifier/head if possible
                if hasattr(model, 'fc'):
                    model.fc = torch.nn.Identity()
                if hasattr(model, 'head'):
                    try:
                        model.head = torch.nn.Identity()
                    except Exception:
                        pass
                feats = model(imgs)

            # pool to (B, embed_dim):
            #   ViT full sequence: (B, N, C) -> use CLS token at index 0
            #   CNN feature maps:  (B, C, H, W) -> global average pool
            #   already pooled:    (B, C) -> no-op
            if feats.ndim == 3:
                feats = feats[:, 0, :]          # CLS token -> (B, C)
            elif feats.ndim == 4:
                feats = feats.mean(dim=(-2, -1))  # GAP -> (B, C)
            feats = feats.flatten(start_dim=1)
            feats = feats.cpu().numpy()
            embeddings.append(feats)
            fnames.extend(paths)
            labels.extend(labs)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    np.savez_compressed(args.output, embeddings=embeddings, filenames=np.array(fnames), labels=labels)
    print(f'Saved embeddings to {args.output} with shape {embeddings.shape}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input folder with images')
    parser.add_argument('--output', required=True, help='output .npz file')
    parser.add_argument('--model-name', dest='model_name', default='vit_small_patch16_224', help='timm model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='use pretrained weights')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
