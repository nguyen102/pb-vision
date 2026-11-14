"""Step 3: fine-tune YOLOv8 on the full-resolution crop dataset from labels.py.

The crops are native-resolution, so imgsz 640 trains at the ball's real size and the pipeline runs the
model at native resolution (imgsz 1920 for 1080p). Scale augmentation is mild so the ball stays a few
pixels, not one.

Usage: python training/train.py training/data/ds-r0/data.yaml [--model yolov8s.pt --epochs 40 --name r1]
--resume continues an interrupted run (training/run_train.sh uses it after a hang).
Outputs training/runs/<name>/weights/best.pt (and best-<size>.onnx with --onnx 640 1280).
"""
import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv.tracking import DEVICE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="r1")
    ap.add_argument("--resume", action="store_true", help="continue training/runs/<name>/weights/last.pt if it exists")
    ap.add_argument("--onnx", type=int, nargs="*", default=[], help="also export ONNX at these sizes")
    args = ap.parse_args()

    last = Path("training/runs") / args.name / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"resuming {last}")
        YOLO(str(last)).train(resume=True)
    else:
        _train(args)
    _finish(args)


def _train(args):
    model = YOLO(args.model)
    model.train(
        data=str(args.data), imgsz=640, epochs=args.epochs, batch=args.batch, device=DEVICE or 0,
        project=str(Path("training/runs").resolve()), name=args.name, exist_ok=True, patience=10,
        workers=args.workers, plots=False,
        # Native-scale crops: little zoom so the ball neither vanishes nor balloons; no vertical flips
        # (gravity); mosaic stitches 4 crops at the same scale.
        scale=0.2, mosaic=1.0, close_mosaic=5, degrees=0.0, flipud=0.0, fliplr=0.5, translate=0.1,
    )


def _finish(args):
    best = Path("training/runs") / args.name / "weights" / "best.pt"
    for size in args.onnx:
        out = YOLO(str(best)).export(format="onnx", imgsz=size, simplify=True, opset=17)
        shutil.move(out, best.with_name(f"best-{size}.onnx"))
    print(f"done: {best}")


if __name__ == "__main__":
    main()
