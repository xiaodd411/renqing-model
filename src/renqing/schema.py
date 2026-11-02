"""Small, strict local schema; deliberately not an official Jev wire protocol."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def text(value: Any, field: str, limit: int = 6000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} 必须是 1–{limit} 字符的非空字符串")
    return value


@dataclass(frozen=True)
class Question:
    kind: str
    prompt: str
    options: tuple[str, ...]
    values: tuple[float, ...] = ()

    @classmethod
    def parse(cls, data: dict) -> "Question":
        if not isinstance(data, dict):
            raise ValueError("question 必须是对象")
        kind = data.get("type")
        prompt = text(data.get("prompt"), "prompt", 800)
        if kind == "noul":
            return cls(kind, prompt, ("是", "否"))
        if kind not in ("choice", "score"):
            raise ValueError("type 必须是 choice、score 或 noul")
        options = data.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 16:
            raise ValueError("options 必须包含 2–16 个候选项")
        options = tuple(text(x, "option", 600) for x in options)
        if len(set(options)) != len(options):
            raise ValueError("候选项不得重复")
        values = ()
        if kind == "score":
            raw = data.get("values")
            if not isinstance(raw, list) or len(raw) != len(options):
                raise ValueError("score.values 必须与 options 一一对应")
            if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in raw):
                raise ValueError("values 必须是有限数字")
            values = tuple(float(x) for x in raw)
            if any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError("score.values 必须严格递增")
        return cls(kind, prompt, options, values)

    def as_dict(self) -> dict:
        out = {"type": self.kind, "prompt": self.prompt}
        if self.kind != "noul":
            out["options"] = list(self.options)
        if self.values:
            out["values"] = list(self.values)
        return out

    def queries(self) -> list[str]:
        # No position embeddings for candidate order: each is independently scored.
        return [f"任务：{self.kind}。问题：{self.prompt}。候选判断：{option}" for option in self.options]


@dataclass(frozen=True)
class Example:
    id: str
    group_id: str
    domain: str
    state: str
    question: Question
    target: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, data: dict) -> "Example":
        if not isinstance(data, dict):
            raise ValueError("每行样本必须是对象")
        q = Question.parse(data.get("question"))
        target = data.get("target")
        if not isinstance(target, list) or len(target) != len(q.options):
            raise ValueError("target 必须与候选项一一对应（noul 为 [P(是), P(否)]）")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in target):
            raise ValueError("target 必须是非负有限概率")
        if not math.isclose(sum(target), 1.0, abs_tol=1e-5):
            raise ValueError("target 概率之和必须为 1")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata 必须是对象")
        return cls(text(data.get("id"), "id", 200), text(data.get("group_id"), "group_id", 200),
                   text(data.get("domain"), "domain", 100), text(data.get("state"), "state"), q, tuple(target), dict(metadata))

    def as_dict(self) -> dict:
        result = {"id": self.id, "group_id": self.group_id, "domain": self.domain, "state": self.state,
                  "question": self.question.as_dict(), "target": list(self.target)}
        if self.metadata:
            result["metadata"] = self.metadata
        return result


def read_examples(path: str | Path) -> list[Example]:
    result = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    result.append(Example.parse(json.loads(line)))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{n}: {exc}") from exc
    if not result:
        raise ValueError(f"{path}: 数据集为空")
    ids = [x.id for x in result]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: id 重复")
    return result


def validate_splits(splits: dict[str, list[Example]]) -> dict:
    groups, states, identities = {}, {}, set()
    for split, examples in splits.items():
        for ex in examples:
            if ex.id in identities:
                raise ValueError(f"跨集合 id 重复：{ex.id}")
            identities.add(ex.id)
            normalized = "".join(ex.state.split())
            for key, index, name in ((ex.group_id, groups, "场景组"), (normalized, states, "相同场景")):
                if key in index and index[key] != split:
                    raise ValueError(f"{name}跨集合泄漏：{key[:80]}")
                index[key] = split
    return {s: {"examples": len(xs), "groups": len({x.group_id for x in xs}),
                "types": {k: sum(x.question.kind == k for x in xs) for k in ("choice", "score", "noul")}}
            for s, xs in splits.items()}


def parse_request(data: dict) -> tuple[str, dict[str, Question]]:
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象")
    state = text(data.get("state"), "state")
    raw = data.get("questions")
    if not isinstance(raw, dict) or not 1 <= len(raw) <= 32:
        raise ValueError("questions 必须是包含 1–32 个问题的对象")
    return state, {text(key, "question id", 100): Question.parse(value) for key, value in raw.items()}
