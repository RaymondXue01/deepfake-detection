import subprocess
import os
from pathlib import Path

DATASET = "tristanzhang32/ai-generated-images-vs-real-images"

def main():
    out = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "--unzip", "-p", str(out)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
