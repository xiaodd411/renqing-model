from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="中文人际场景的训练、评测与类型化推理")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/demo")
    t.add_argument("--output", default="runs/renqing-v1")
    t.add_argument("--base-model", default="hfl/rbt3")
    t.add_argument("--revision", default="main")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=12)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--max-length", type=int, default=256)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--threshold", type=float, default=0.8)
    t.add_argument("--balance-tasks", action="store_true", help="按任务类型均衡采样，并按验证集任务平均 NLL 选轮次")
    p = sub.add_parser("predict")
    p.add_argument("--model", default="runs/renqing-v1")
    p.add_argument("--input", required=True, help="JSON 文件路径；- 代表 stdin")
    p.add_argument("--output")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threshold", type=float, default=0.8)
    v = sub.add_parser("validate")
    v.add_argument("--data", default="data/demo")
    e = sub.add_parser("evaluate")
    e.add_argument("--model", default="runs/renqing-v1")
    e.add_argument("--data", default="data/demo/test.jsonl")
    e.add_argument("--output", required=True)
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    try:
        if args.command == "train":
            from .train import train
            train(args)
            return
        if args.command == "validate":
            from .schema import read_examples, validate_splits
            result = validate_splits({s: read_examples(Path(args.data) / f"{s}.jsonl")
                                      for s in ("train", "validation", "calibration", "test")})
        elif args.command == "predict":
            from .model import Predictor
            raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8-sig")
            result = Predictor(args.model, args.device).predict(json.loads(raw), args.threshold)
        else:
            from .model import Predictor, collect_logits
            from .schema import read_examples
            from .train import sliced_metrics
            predictor = Predictor(args.model, args.device)
            examples = read_examples(args.data)
            logits = collect_logits(predictor.model, predictor.tokenizer, examples, predictor.meta["max_length"])
            temperature = predictor.calibration["temperature"] if predictor.calibration else 1.0
            result = sliced_metrics(examples, logits, temperature, args.threshold)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if getattr(args, "output", None):
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Refuse overwriting a user's review/export file.
            with target.open("x", encoding="utf-8") as f:
                f.write(encoded + "\n")
        print(encoded)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
