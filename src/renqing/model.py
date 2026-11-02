"""One shared Chinese encoder scores (state, question, candidate) pairs."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file, save_file


def choose_device(name="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


class DecisionModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(encoder.config.hidden_size, 1))

    def forward(self, **inputs):
        out = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)

    def save(self, path, tokenizer, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder", safe_serialization=True)
        tokenizer.save_pretrained(path / "tokenizer")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}, path / "head.safetensors")
        (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, device="auto"):
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(AutoModel.from_pretrained(path / "encoder", local_files_only=True))
        model.head.load_state_dict(load_file(path / "head.safetensors"))
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        return model.to(choose_device(device)).eval(), tokenizer, meta


def encode_pairs(tokenizer, states, queries, max_length=256):
    if max_length < 32:
        raise ValueError("max_length 必须至少为 32")
    # Preserve all evidence: reject oversized inputs instead of silently losing context.
    query_lengths = [len(x) for x in tokenizer(queries, add_special_tokens=False)["input_ids"]]
    if max(query_lengths) + tokenizer.num_special_tokens_to_add(pair=True) >= max_length:
        raise ValueError("问题和候选项过长，请缩短文本或增大 max_length")
    state_lengths = [len(x) for x in tokenizer(states, add_special_tokens=False)["input_ids"]]
    special = tokenizer.num_special_tokens_to_add(pair=True)
    if any(a + b + special > max_length for a, b in zip(state_lengths, query_lengths)):
        raise ValueError("场景、问题和候选项合计超过模型长度限制，请精简场景；不会静默截断重要上下文")
    return tokenizer(states, queries, padding=True, truncation="only_first", max_length=max_length, return_tensors="pt")


def batch_examples(examples, tokenizer, max_length, device):
    states, queries, sizes = [], [], []
    for ex in examples:
        qs = ex.question.queries()
        states.extend([ex.state] * len(qs))
        queries.extend(qs)
        sizes.append(len(qs))
    inputs = encode_pairs(tokenizer, states, queries, max_length)
    return {k: v.to(device) for k, v in inputs.items()}, sizes


@torch.inference_mode()
def collect_logits(model, tokenizer, examples, max_length, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    logits = []
    for start in range(0, len(examples), batch_size):
        inputs, sizes = batch_examples(examples[start:start + batch_size], tokenizer, max_length, device)
        logits.extend(x.float().cpu().numpy() for x in model(**inputs).split(sizes))
    return logits


class Predictor:
    def __init__(self, model_dir, device="auto"):
        self.model, self.tokenizer, self.meta = DecisionModel.load(model_dir, device)
        calibration = Path(model_dir) / "calibration.json"
        self.calibration = json.loads(calibration.read_text(encoding="utf-8")) if calibration.exists() else None

    @torch.inference_mode()
    def predict(self, data, threshold=0.8):
        from .schema import parse_request
        from .metrics import decision_result
        if not 0 <= threshold <= 1:
            raise ValueError("threshold 必须在 [0,1] 内")
        state, questions = parse_request(data)
        states, queries, sizes = [], [], []
        for q in questions.values():
            qs = q.queries()
            states.extend([state] * len(qs))
            queries.extend(qs)
            sizes.append(len(qs))
        # Bound peak memory regardless of number of questions/options.
        outputs = []
        device = next(self.model.parameters()).device
        for start in range(0, len(queries), 32):
            inputs = encode_pairs(self.tokenizer, states[start:start + 32], queries[start:start + 32], self.meta["max_length"])
            outputs.append(self.model(**{k: v.to(device) for k, v in inputs.items()}).float().cpu())
        chunks = torch.cat(outputs).split(sizes)
        temperature = self.calibration["temperature"] if self.calibration else 1.0
        calibration_scope = self.calibration.get("scope", "heldout_dataset_only") if self.calibration else "uncalibrated"
        return {"model": "renqing-0.1", "answers": {
            key: decision_result(q, logit.numpy(), temperature, threshold, self.calibration is not None, calibration_scope)
            for (key, q), logit in zip(questions.items(), chunks)}}
