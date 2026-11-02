"""Build traceable v2 data; group original question families before splitting."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
import unicodedata
import urllib.request

import numpy as np
from transformers import AutoTokenizer
from renqing.schema import Example, validate_splits


URL = "https://bj.bcebos.com/paddlenlp/datasets/dureader_yesno-data.tar.gz"
MD5 = "30c744d65e87fdce00cdc707fd008138"
EXPECTED_SHA256 = "c6fdbc76771e0eb3c36de78017da64a1097992536d204e98eb5d7e32abe1c44d"
DOMAINS = {
    "家庭": "父母 妈妈 爸爸 亲戚 婆婆 家人 兄弟 姐妹 长辈".split(),
    "伴侣": "男朋友 女朋友 老公 老婆 恋爱 丈夫 妻子 夫妻 伴侣 男友 女友 分手".split(),
    "职场": "同事 领导 老板 上司 下属 职场".split(),
    "朋友": "朋友 友谊 友情 闺蜜".split(),
    "校园": "同学 老师 同桌 室友".split(),
    "服务": "客户 顾客 客服 服务员 卖家 买家".split(),
    "邻里": "邻居 合租 房东 租客".split(),
    "协作": "合作 队友 搭档 团队 组员".split(),
    "线上": "网友 群聊 拉黑 私信 评论 点赞 微信 聊天".split(),
    "沟通": "拒绝 道歉 尊重 礼貌 委婉 隐私 冷战 吵架 嫉妒 送礼 人情 面子 借钱 生气 伤心 讨厌 误会 信任 倾诉 安慰".split(),
}
PII = re.compile(r"https?://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|(?:QQ|qq|微信号|手机号)\s*[:：]?\s*[A-Za-z0-9_-]{5,}")
OFF_TOPIC = re.compile("治疗|手术|癌症|药物|艾滋|阴茎|性交|做爱|自杀|自残|股票|贷款|保险理赔")


def canonical(value):
    value = unicodedata.normalize("NFKC", value).lower()
    return "".join(c for c in value if c.isalnum())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print("Downloading official DuReader YesNo archive", flush=True)
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=120) as source, partial.open("wb") as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
        partial.replace(path)
    payload = path.read_bytes()
    if hashlib.md5(payload).hexdigest() != MD5 or hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("官方数据校验失败；不会使用未核验的下载文件")


def near_question_clusters(questions, threshold=0.85):
    """Deterministic 32-permutation, 8-band character-trigram MinHash LSH.

    Approximate lexical screening, not semantic equivalence detection.
    """
    keys = sorted(set(questions))
    parents = {x: x for x in keys}
    grams = {x: {x[i:i + 3] for i in range(max(1, len(x) - 2))} for x in keys}
    rng = np.random.default_rng(2048)
    a = rng.integers(1, 2**31, size=32, dtype=np.uint64)[:, None]
    b = rng.integers(0, 2**31, size=32, dtype=np.uint64)[:, None]
    cache, buckets, checked, merged = {}, collections.defaultdict(list), set(), 0

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for key in keys:
        hashes = []
        for gram in grams[key]:
            if gram not in cache:
                cache[gram] = int.from_bytes(hashlib.blake2s(gram.encode(), digest_size=4).digest(), "little")
            hashes.append(cache[gram])
        signatures = ((a * np.asarray(hashes, dtype=np.uint64) + b) % 4294967311).min(axis=1)
        candidates = set()
        for band in range(8):
            bucket = (band, tuple(int(x) for x in signatures[band * 4:band * 4 + 4]))
            candidates.update(buckets[bucket])
            buckets[bucket].append(key)
        for other in candidates:
            pair = (other, key)
            if pair in checked:
                continue
            checked.add(pair)
            similarity = len(grams[key] & grams[other]) / max(1, len(grams[key] | grams[other]))
            if similarity >= threshold:
                left, right = find(key), find(other)
                if left != right:
                    parents[max(left, right)] = min(left, right)
                    merged += 1
    return {x: find(x) for x in keys}, {"questions": len(keys), "candidate_pairs_checked": len(checked), "near_question_merges": merged,
                                       "method": "character_trigram_minhash_lsh", "jaccard_threshold": threshold,
                                       "limitation": "approximate lexical matches only; no claim of complete semantic deduplication"}


def source_rows(archive_path, general_limit, dev_limit, seed):
    audit = collections.Counter()
    candidates = {"train": [], "dev": []}
    seen, conflicts = {}, set()
    with tarfile.open(archive_path) as archive:
        for split in candidates:
            for line in archive.extractfile(f"dureader_yesno-data/{split}.json"):
                row = json.loads(line)
                audit[f"{split}_raw"] += 1
                q, answer, label = row.get("question"), row.get("answer"), row.get("yesno_answer")
                if not isinstance(q, str) or not isinstance(answer, str) or label not in ("Yes", "No", "Depends"):
                    audit["invalid"] += 1
                    continue
                if not q.strip() or not answer.strip() or len(q) + len(answer) > 200:
                    audit["empty_or_over_200_characters"] += 1
                    continue
                if PII.search(q + " " + answer):
                    audit["contact_or_url_filtered"] += 1
                    continue
                if OFF_TOPIC.search(q):
                    audit["off_topic_filtered"] += 1
                    continue
                key = (canonical(q), canonical(answer))
                if key in seen:
                    if seen[key] != label:
                        conflicts.add(key)
                    audit["duplicate_pairs"] += 1
                    continue
                seen[key] = label
                domain = next((d for d, terms in DOMAINS.items() if any(t in q for t in terms)), "一般观点")
                candidates[split].append({"question": q, "answer": answer, "label": label, "domain": domain,
                                          "source_id": str(row["id"]), "source_split": split, "key": key})
    for split in candidates:
        candidates[split] = [x for x in candidates[split] if x["key"] not in conflicts]
    audit["conflicting_pairs_removed"] = len(conflicts)
    rng = random.Random(seed)
    selected = []
    for split, rows in candidates.items():
        social = [x for x in rows if x["domain"] != "一般观点"]
        general = [x for x in rows if x["domain"] == "一般观点"]
        rng.shuffle(general)
        limit = general_limit if split == "train" else dev_limit
        per_label = max(1, limit // 3)
        chosen = social + [x for label in ("Yes", "No", "Depends") for x in [r for r in general if r["label"] == label][:per_label]]
        selected.extend(chosen)
    clusters, near_audit = near_question_clusters([x["key"][0] for x in selected])
    dev_groups = {clusters[x["key"][0]] for x in selected if x["source_split"] == "dev"}
    kept = []
    for row in selected:
        group = clusters[row["key"][0]]
        if row["source_split"] == "train" and group in dev_groups:
            audit["train_rows_overlapping_dev_question_cluster"] += 1
            continue
        row["group_id"] = "dureader-" + digest(group)
        kept.append(row)
    return kept, {**dict(audit), "near_duplicates": near_audit}


def public_examples(rows):
    for row in rows:
        state = f"提问：{row['question']}\n一段回答：{row['answer']}"
        group = row["group_id"]
        meta = {"source": "dureader_yesno", "source_split": row["source_split"], "source_id": row["source_id"],
                "label_kind": "original_opinion_polarity_human_annotation", "scope": "answer_stance_not_world_truth",
                "question_cluster": group}
        q = {"type": "choice", "prompt": "这段回答对所提问题表达了哪种立场？",
             "options": ["明确肯定或赞同", "明确否定或反对", "无法确定或需要分情况"]}
        labels = ("Yes", "No", "Depends")
        definitions = [("stance", q, [float(row["label"] == x) for x in labels])]
        for polarity, prompt in (("Yes", "这段回答是否明确赞同所提问题？"), ("No", "这段回答是否明确否定所提问题？"),
                                  ("Depends", "这段回答的立场是否不确定或需要分情况？")):
            yes = float(row["label"] == polarity)
            definitions.append((polarity.lower(), {"type": "noul", "prompt": prompt}, [yes, 1 - yes]))
        if row["source_split"] == "dev":
            split = "test"
        else:
            bucket = int(digest(group)[:8], 16) % 100
            split = "train" if bucket < 84 else "validation" if bucket < 92 else "calibration"
        for task, question, target in definitions:
            yield split, Example.parse({"id": f"dy-{row['source_split']}-{row['source_id']}-{task}", "group_id": group,
                                        "domain": row["domain"], "state": state, "question": question, "target": target, "metadata": meta})


def synthetic_examples(families, seed):
    rng = random.Random(seed)
    by_domain = collections.defaultdict(list)
    for domain, person, request in families:
        by_domain[domain].append((person, request))
    for domain, rows in by_domain.items():
        rng.shuffle(rows)
        for i, (person, request) in enumerate(rows):
            split = "train" if i < 4 else "validation" if i == 4 else "calibration"
            group = "contrast-" + digest(domain + person + request)
            options = [f"可以，我愿意{request}，我们把时间和范围确认一下。",
                       f"这次我没有合适条件{request}，需要换一个时间或另找安排。",
                       "我先不直接答应，想确认具体时间、范围和你需要我做的部分。",
                       "我已经明确表达过拒绝，请停止施压，之后只讨论双方同意的安排。"]
            variants = [
                (f"我愿意帮忙，时间和能力都允许，具体范围也已确认。{person}说明完全自愿，我希望把这件事安排好。", 0, 0, 0.98, 0.02),
                (f"这件事的时间和范围已讲清，但我当天已有不能调整的安排，确定接不了。我想礼貌说明限制。{person}表示可以拒绝。", 1, 0, 0.02, 0.02),
                (f"对方还没说清时间和工作量。我并未确定是否有空，想先了解具体需求。没有人催促我当场表态。", 2, 0, 0.50, 0.02),
                (f"我已多次明确说明自己不愿承担，{person}仍不断催促，并用关系和面子要求我必须答应。我希望他停止施压。", 3, 2, 0.02, 0.98),
                (f"我愿意帮，但目前提出的时间有冲突。双方对安排还存在分歧，正在商量替代时段，还没有谈妥，也没有威胁。", 2, 1, 0.50, 0.02),
                (f"我已说明近期无法承担，{person}认可这个限制，没有继续纠缠。我希望确认这次不接，并保持正常联系。", 1, 0, 0.02, 0.02)]
            for v, (condition, best, risk, can_accept, pressure) in enumerate(variants):
                renderings = [f"{person}请我{request}。{condition}",
                              f"当前需要考虑的请求是：{request}。提出请求的人是{person}。实际情况：{condition}",
                              f"我正在和{person}沟通。对方希望我{request}。补充背景：{condition}"]
                for rendering, state in enumerate(renderings):
                    prefix = f"{group}-{v}-{rendering}"
                    order = list(range(4))
                    rng.shuffle(order)
                    target = [0.88 if j == best else 0.04 for j in order]
                    risk_target = [[0.9, 0.09, 0.01], [0.08, 0.84, 0.08], [0.01, 0.09, 0.9]][risk]
                    definitions = [
                        ("choice", {"type": "choice", "prompt": "结合我的目标、实际条件和既往沟通，哪种回应最合适？",
                                    "options": [options[j] for j in order]}, target),
                        ("risk", {"type": "score", "prompt": "当前互动中，反复越界或冲突升级的迹象有多明显？",
                                  "options": ["低：没有明显越界或冲突", "中：已有分歧，需要协商", "高：已有反复越界或明显施压"],
                                  "values": [0, 0.5, 1]}, risk_target),
                        ("available", {"type": "noul", "prompt": "根据已知信息，我目前是否愿意且有条件接受这项请求？"}, [can_accept, 1 - can_accept]),
                        ("pressure", {"type": "noul", "prompt": "对方是否在我明确拒绝后仍反复施压？"}, [pressure, 1 - pressure]),
                        ("notpressure", {"type": "noul", "prompt": "这次互动中，对方是否没有在明确拒绝后继续施压？"}, [1 - pressure, pressure])]
                    meta = {"source": "synthetic_contrast_v2", "template_family": group, "variant": v,
                            "rendering": rendering, "label_kind": "author_assigned_not_observed_frequency", "human_reviewed": False}
                    for task, question, target in definitions:
                        yield split, Example.parse({"id": f"{prefix}-{task}", "group_id": group, "domain": domain,
                                                    "state": state, "question": question, "target": target, "metadata": meta})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="cache/dureader_yesno-data.tar.gz")
    parser.add_argument("--output", default="data/expanded-v2")
    parser.add_argument("--families", default="data/contrast_families.json")
    parser.add_argument("--demo", default="data/demo")
    parser.add_argument("--challenge", default="data/challenge_v2.jsonl")
    parser.add_argument("--tokenizer", default="hfl/rbt3")
    parser.add_argument("--general-train-rows", type=int, default=3000)
    parser.add_argument("--general-test-rows", type=int, default=450)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空，不覆盖现有标注或数据")
    download(args.archive)
    rows, source_audit = source_rows(args.archive, args.general_train_rows, args.general_test_rows, args.seed)
    candidates = list(public_examples(rows))
    families = json.loads(Path(args.families).read_text(encoding="utf-8"))["families"]
    candidates.extend(synthetic_examples(families, args.seed))
    # V1 validation/calibration/test are never added to v2 training.
    for line in (Path(args.demo) / "train.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        item["metadata"] = {"source": "original_demo", "label_kind": "author_assigned_not_observed_frequency", "human_reviewed": False}
        candidates.append(("train", Example.parse(item)))
    challenge = Path(args.challenge)
    if challenge.exists():
        candidates.extend(("test", Example.parse(json.loads(line))) for line in challenge.read_text(encoding="utf-8").splitlines() if line.strip())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    texts = sorted({s for _, ex in candidates for s in [ex.state] + ex.question.queries()})
    lengths = {}
    for start in range(0, len(texts), 256):
        batch = texts[start:start + 256]
        lengths.update(zip(batch, map(len, tokenizer(batch, add_special_tokens=False)["input_ids"])))
    splits = {s: [] for s in ("train", "validation", "calibration", "test")}
    overlength, seen = 0, set()
    for split, ex in candidates:
        if any(lengths[ex.state] + lengths[q] + tokenizer.num_special_tokens_to_add(pair=True) > args.max_length for q in ex.question.queries()):
            overlength += 1
            continue
        identity = (canonical(ex.state), ex.question.prompt, ex.question.options)
        if identity in seen:
            raise ValueError("转换后发现相同任务重复；需在来源层排查")
        seen.add(identity)
        splits[split].append(ex)
    stats = validate_splits(splits)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split, examples in splits.items():
        rng.shuffle(examples)
        (output / f"{split}.jsonl").write_text("".join(json.dumps(x.as_dict(), ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")
    source_counts = {s: dict(collections.Counter(ex.metadata.get("source", "unknown") for ex in xs)) for s, xs in splits.items()}
    manifest = {"version": "0.2.0", "seed": args.seed, "source": "mixed_public_opinion_polarity_and_original_synthetic_social_contrasts",
                "human_reviewed": False, "review_note": "public polarity labels are human sourced; synthetic social labels are not externally reviewed",
                "max_length": args.max_length, "splits": stats, "source_counts": source_counts,
                "unique_state_count": len({canonical(x.state) for xs in splits.values() for x in xs}),
                "independent_group_count": len({x.group_id for xs in splits.values() for x in xs}),
                "synthetic_family_count": len(families), "overlength_tasks_removed": overlength,
                "source_audit": source_audit, "upstream": {"url": URL, "md5": MD5, "sha256": EXPECTED_SHA256,
                  "project": "https://github.com/baidu/DuReader", "license": "Apache-2.0",
                  "label_meaning": "stance of an answer, NOT objective truth of question or social appropriateness"},
                "quality_limitations": ["lexical near-duplicate screening is approximate", "synthetic variants do not increase independent template-family count",
                                        "opinion recognition is auxiliary and not an empathy or social intelligence benchmark",
                                        "synthetic target probabilities are author assignments, not measured human agreement"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"splits": stats, "source_counts": source_counts, "overlength_removed": overlength}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
