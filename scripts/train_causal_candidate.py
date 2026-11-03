"""Source-balanced bounded-decision LoRA, development selection, no test reads."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model

from renqing.causal import CausalPredictor, render_prompt
from renqing.metrics import evaluate
from renqing.schema import read_examples


def select_development(examples, limit):
    rng, strata = random.Random(128), {}
    for ex in examples:
        strata.setdefault((ex.metadata.get("source", "demo"), ex.question.kind), []).append(ex)
    selected = []
    for group in strata.values():
        selected.extend(rng.sample(group, min(len(group), max(1, limit // len(strata)))))
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-candidate", default="runs/causal-round1-base")
    p.add_argument("--data", default="data/expanded-v2")
    p.add_argument("--output", default="runs/causal-round1-lora")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=0.1)
    p.add_argument("--dev-limit", type=int, default=360)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.dev_limit) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.anchor <= 1:
        raise ValueError("非法训练参数")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train = read_examples(Path(args.data) / "train.jsonl")
    dev = select_development(read_examples(Path(args.data) / "validation.jsonl"), args.dev_limit)
    train_groups, dev_groups = {x.group_id for x in train}, {x.group_id for x in dev}
    if train_groups & dev_groups:
        raise ValueError("训练与验证组重叠")
    strata = {}
    for ex in train:
        strata.setdefault(ex.question.kind, {}).setdefault(ex.metadata.get("source", "demo"), []).append(ex)
    weights = {"original_demo": 1.0, "synthetic_contrast_v2": 2.0, "dureader_yesno": 1.0}
    predictor = CausalPredictor(args.base_candidate)
    metadata = {**predictor.meta, "model_id": "renqing-causal-round1-lora", "adapter": True,
                "hyperparameters": vars(args), "selection": "macro task development NLL only", "test_read_during_training": False,
                "source_sampling_weights": weights, "development_ids": [x.id for x in dev],
                "data_sha256": {s: hashlib.sha256((Path(args.data) / f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "validation")}}
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def assess():
        predictor.model.eval()
        scores = predictor.score_examples(dev)
        by_type = {}
        for kind in strata:
            indices = [i for i, x in enumerate(dev) if x.question.kind == kind]
            by_type[kind] = evaluate([scores[i] for i in indices], [dev[i].target for i in indices])
        return float(np.mean([m["nll"] for m in by_type.values()])), by_type

    baseline, baseline_types = assess()
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    predictor.model = get_peft_model(predictor.model, config)
    predictor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    predictor.model.enable_input_require_grads()
    trainable = sum(x.numel() for x in predictor.model.parameters() if x.requires_grad)
    total = sum(x.numel() for x in predictor.model.parameters())
    if not 0 < trainable < total * 0.01:
        raise AssertionError("不应更新完整底座权重")
    optimizer = torch.optim.AdamW([x for x in predictor.model.parameters() if x.requires_grad], lr=args.lr, weight_decay=0.01)
    history, best, best_step = [], float("inf"), None
    kinds = sorted(strata)
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        predictor.model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(args.accumulation):
            batch = []
            for b in range(args.batch_size):
                kind = kinds[((step - 1) * args.accumulation * args.batch_size + micro * args.batch_size + b) % len(kinds)]
                sources = sorted(strata[kind])
                source = rng.choices(sources, weights=[weights.get(s, 1.0) for s in sources], k=1)[0]
                batch.append(rng.choice(strata[kind][source]))
            prompts, targets, sizes = [], [], []
            for ex in batch:
                order = list(range(len(ex.target)))
                rng.shuffle(order)
                prompts.append(render_prompt(predictor.tokenizer, ex.state, ex.question, order))
                targets.append(torch.tensor([ex.target[i] for i in order], device=predictor.device, dtype=torch.float32))
                sizes.append(len(order))
            inputs = predictor.encode(prompts)
            with predictor.model.disable_adapter(), torch.no_grad():
                predictor.model.eval()
                teacher = predictor.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            predictor.model.train()
            logits = predictor.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            losses = []
            for row, base, target, size in zip(logits, teacher, targets, sizes):
                selected = row[predictor.letter_ids[:size]].log_softmax(-1)
                base_log = base[predictor.letter_ids[:size]].log_softmax(-1)
                loss = -(target * selected).sum() + args.anchor * (base_log.exp() * (base_log - selected)).sum()
                losses.append(loss)
            loss = torch.stack(losses).mean() / args.accumulation
            if not torch.isfinite(loss):
                raise ValueError("训练出现非有限损失")
            loss.backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([x for x in predictor.model.parameters() if x.requires_grad], 1.0)
        progress = step / args.steps
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(step / 8, max(0.1, 1 - progress))
        optimizer.step()
        entry = {"step": step, "loss": total_loss}
        if step % 16 == 0:
            print(json.dumps({**entry, "seconds": time.perf_counter() - start_time}), flush=True)
        if step % 64 == 0 or step == args.steps:
            selection, metrics = assess()
            entry.update(development_macro_nll=selection, development_by_type=metrics)
            print(json.dumps({"step": step, "development_macro_nll": selection,
                              "development_accuracy": {k: v["accuracy_against_target_mode"] for k, v in metrics.items()}}), flush=True)
            if selection < best:
                best, best_step = selection, step
                predictor.model.peft_config["default"].base_model_name_or_path = metadata["base_model"]
                predictor.model.peft_config["default"].revision = metadata["base_revision"]
                predictor.model.save_pretrained(root, safe_serialization=True, save_embedding_layers=False)
        history.append(entry)
        (root / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata.update(best_step=best_step, development_macro_nll=best, baseline_development_macro_nll=baseline,
                    adapter_improved_development_nll=best < baseline, trainable_parameters=trainable, total_parameters=total,
                    training_seconds=time.perf_counter() - start_time)
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_step": best_step, "baseline_nll": baseline, "best_nll": best, "trainable_parameters": trainable}), flush=True)


if __name__ == "__main__":
    main()
