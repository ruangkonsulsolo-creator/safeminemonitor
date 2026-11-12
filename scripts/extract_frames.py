"""
Extract frames from a video every N seconds (for manual ground-truth labelling
or for adding your own images to the training set).

Usage:  python scripts/extract_frames.py data/samples/site1.mp4 --every 1.0
Output: data/ground_truth/frames/<video-name>/<t>.jpg   (file name = time in s)
"""
import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main(video, every):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = ROOT / "data" / "ground_truth" / "frames" / Path(video).stem
    out.mkdir(parents=True, exist_ok=True)
    i = n = 0
    step = max(1, int(round(fps * every)))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(out / f"{i / fps:08.2f}.jpg"), f)
            n += 1
        i += 1
    print(f"saved {n} frames -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args()
    main(a.video, a.every)
