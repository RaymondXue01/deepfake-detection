"""Basic preprocessing pipeline: compression, gaussian blur, gaussian noise."""
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import io
import argparse


def compress_image(img: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_gaussian_noise(img: Image.Image, sigma: float = 5.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def process_folder(src: Path, dst: Path, compress_q=85, blur_r=0.0, noise_s=0.0, limit=None):
    dst.mkdir(parents=True, exist_ok=True)
    files = list(src.rglob('*.jpg')) + list(src.rglob('*.png'))
    if limit:
        files = files[:limit]
    for i, f in enumerate(files, 1):
        try:
            img = Image.open(f).convert('RGB')
        except Exception:
            continue
        if compress_q and compress_q < 100:
            img = compress_image(img, quality=compress_q)
        if blur_r and blur_r > 0.0:
            img = gaussian_blur(img, radius=blur_r)
        if noise_s and noise_s > 0.0:
            img = add_gaussian_noise(img, sigma=noise_s)
        out_path = dst / f.name
        img.save(out_path, format='JPEG')
        if i % 100 == 0:
            print(f'Processed {i}/{len(files)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compress', type=int, default=85)
    parser.add_argument('--blur', type=float, default=0.0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    process_folder(Path(args.input), Path(args.output), args.compress, args.blur, args.noise, args.limit)


if __name__ == '__main__':
    main()
