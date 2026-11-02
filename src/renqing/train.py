"""Supervised decision training + separate scalar temperature calibration."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from .metrics import evaluate, fit_temperature
from .model import DecisionModel, batch_examples, choose_device, collect_logits
from .schema import read_examples, validate_splits


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sliced_metrics(examples, logits, temperature, threshold):
    result = {"overall": evaluate(logits, [x.target for x in examples], temperature, threshold)}
    for field, getter in (("by_type", lambda x: x.question.kind), ("by_domain", lambda x: x.domain)):
        result[field] = {}
        for group in sorted({getter(x) for x in examples}):
            indices = [i for i, x in enumerate(examples) if getter(x) == group]
            result[field][group] = evaluate([logits[i] for i in indices], [examples[i].target for i in indices], temperature, threshold)
    return result


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        raise ValueError("epochs、batch_size、lr、patience 必须为正数")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold 必须在 [0,1] 内")
    if not 32 <= args.max_length <= 512:
        raise ValueError("max_length 必须在 [32,512] 内")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} 非空，请选择新的输出目录")
    data = Path(args.data)
    splits = {name: read_examples(data / f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")}
    stats = validate_splits(splits)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    encoder = AutoModel.from_pretrained(args.base_model, revision=args.revision)
    model = DecisionModel(encoder).to(device)
    source_file = Path(args.base_model) / "source.json"
    source = json.loads(source_file.read_text(encoding="utf-8")) if source_file.is_file() else {}
    metadata = {"base_model": source.get("model", args.base_model),
                "base_revision": source.get("revision", encoder.config._commit_hash or args.revision),
                "seed": args.seed, "max_length": args.max_length, "split_stats": stats,
                "dataset_manifest": manifest, "parameters": sum(x.numel() for x in model.parameters()),
                "data_sha256": {name: hashlib.sha256((data / f"{name}.jsonl").read_bytes()).hexdigest() for name in splits},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "transformers": transformers.__version__, "device": str(device),
                                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                "hyperparameters": {k: v for k, v in vars(args).items() if k != "func"}}
    print(json.dumps({"device": str(device), "parameters": metadata["parameters"], "splits": stats}, ensure_ascii=False), flush=True)
    start = time.perf_counter()
    # Predeclared baseline; never used for epoch or temperature selection.
    baseline_logits = collect_logits(model, tokenizer, splits["test"], args.max_length, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(splits["train"]) / args.batch_size)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history, best_nll, best_epoch, stale, step = [], float("inf"), 0, 0, 0
    generator = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        examples = splits["train"].copy()
        generator.shuffle(examples)
        total_loss = 0.0
        for offset in range(0, len(examples), args.batch_size):
            batch = examples[offset:offset + args.batch_size]
            inputs, sizes = batch_examples(batch, tokenizer, args.max_length, device)
            optimizer.zero_grad(set_to_none=True)
            warmup = max(1, int(total_steps * 0.1))
            scale = min((step + 1) / warmup, max(0.05, (total_steps - step) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                chunks = model(**inputs).split(sizes)
                losses = [-(torch.tensor(ex.target, device=device, dtype=torch.float32) * logits.float().log_softmax(0)).sum()
                          for ex, logits in zip(batch, chunks)]
                loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失非有限数，请检查数据和学习率")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            step += 1
        logits = collect_logits(model, tokenizer, splits["validation"], args.max_length, args.batch_size)
        metrics = evaluate(logits, [x.target for x in splits["validation"]])
        entry = {"epoch": epoch, "train_loss": total_loss / len(examples), "validation_nll": metrics["nll"],
                 "validation_accuracy": metrics["accuracy_against_target_mode"]}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["nll"] < best_nll - 1e-5:
            best_nll, best_epoch, stale = metrics["nll"], epoch, 0
            model.save(output, tokenizer, {**metadata, "best_epoch": epoch})
        else:
            stale += 1
        save_json(output / "history.json", history)
        if stale >= args.patience:
            break
    # Free training tensors before loading the saved best checkpoint.
    del model, optimizer, encoder, chunks, losses, loss, inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, tokenizer, metadata = DecisionModel.load(output, args.device)
    cal_logits = collect_logits(model, tokenizer, splits["calibration"], args.max_length, args.batch_size)
    cal_targets = [x.target for x in splits["calibration"]]
    temperature = fit_temperature(cal_logits, cal_targets)
    calibration = {"method": "scalar_temperature_nll_grid", "temperature": temperature,
                   "fit_split": "calibration", "fit_examples": len(cal_targets),
                   "before": evaluate(cal_logits, cal_targets), "after": evaluate(cal_logits, cal_targets, temperature),
                   "scope": "synthetic_demo_only" if manifest.get("human_reviewed") is False else "heldout_dataset_only"}
    save_json(output / "calibration.json", calibration)
    test_logits = collect_logits(model, tokenizer, splits["test"], args.max_length, args.batch_size)
    test_targets = [x.target for x in splits["test"]]
    report = {"test_scope": "heldout scenario groups; author-assigned synthetic labels; not a real-world benchmark",
              "best_epoch": best_epoch, "training_seconds": time.perf_counter() - start,
              "threshold": args.threshold, "temperature": temperature,
              "uniform_baseline": evaluate([np.zeros(len(x.target)) for x in splits["test"]], test_targets, threshold=args.threshold),
              "pretrained_untrained_head": sliced_metrics(splits["test"], baseline_logits, 1.0, args.threshold),
              "trained_uncalibrated": sliced_metrics(splits["test"], test_logits, 1.0, args.threshold),
              "trained_calibrated": sliced_metrics(splits["test"], test_logits, temperature, args.threshold)}
    save_json(output / "evaluation.json", report)
    predictions = []
    for ex, logits in zip(splits["test"], test_logits):
        from .metrics import softmax
        predictions.append({**ex.as_dict(), "probabilities": softmax(logits, temperature).tolist()})
    save_json(output / "test_predictions.json", predictions)
    print(json.dumps({"output": str(output), "evaluation": report["trained_calibrated"]["overall"]}, ensure_ascii=False), flush=True)
