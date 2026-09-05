#!/usr/bin/env python3
"""Evaluate unigram, first-order and second-order next-vowel models on Shona verbs.

The Hayes & Wilson learning data are fetched at run time. The source data and per-word
vowel sequences are never written to the repository; only aggregate statistics are emitted.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

VOWELS = ("a", "e", "i", "o", "u")
VOWEL_SET = set(VOWELS)
DATA_URL = "https://brucehayes.org/Phonotactics/files/ShonaLearningData.txt"
SEED = 20260905
ALPHA = 0.5

# Hayes & Wilson (2008), Table 6: Shona vowel distribution, raw counts.
TABLE6 = {
    "aa": 1443, "ae": 3, "ao": 0, "ai": 500, "au": 568,
    "ea": 639, "ee": 587, "eo": 0, "ei": 2, "eu": 260,
    "oa": 638, "oe": 153, "oo": 694, "oi": 23, "ou": 20,
    "ia": 1130, "ie": 0, "io": 0, "ii": 478, "iu": 175,
    "ua": 1737, "ue": 4, "uo": 1, "ui": 175, "uu": 811,
}


@dataclass(frozen=True)
class Corpus:
    sha256: str
    source_rows: int
    entries: int
    form_counts: Mapping[tuple[str, ...], int]


def parse(raw: bytes) -> Corpus:
    counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    source_rows = entries = 0
    for lineno, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"line {lineno}: missing count")
        try:
            n = int(fields[-1])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: invalid count {fields[-1]!r}") from exc
        if n <= 0:
            raise ValueError(f"line {lineno}: count must be positive")
        counts[tuple(fields[:-1])] += n
        source_rows += 1
        entries += n
    return Corpus(hashlib.sha256(raw).hexdigest(), source_rows, entries, dict(counts))


def vowels(form: Sequence[str]) -> tuple[str, ...]:
    return tuple(x for x in form if x in VOWEL_SET)


def split_forms(form_counts: Mapping[tuple[str, ...], int], seed: int) -> dict[str, tuple[tuple[str, ...], ...]]:
    """80/10/10 split, grouping exact duplicate forms to avoid train/test leakage."""
    groups = list(form_counts.items())
    random.Random(seed).shuffle(groups)
    total = sum(n for _, n in groups)
    cuts = (0.8 * total, 0.9 * total)
    out = {"train": [], "validation": [], "test": []}
    assigned = 0
    for form, n in groups:
        midpoint = assigned + n / 2
        bucket = "train" if midpoint < cuts[0] else "validation" if midpoint < cuts[1] else "test"
        out[bucket].extend([form] * n)
        assigned += n
    return {k: tuple(v) for k, v in out.items()}


def examples(seq: Sequence[str], exclude_final: bool) -> Iterable[tuple[tuple[str, ...], str]]:
    for i in range(1, len(seq)):
        if exclude_final and i == len(seq) - 1:
            continue
        yield tuple(seq[:i]), seq[i]


class Model:
    def __init__(self, order: int, alpha: float):
        if order not in (0, 1, 2) or alpha <= 0:
            raise ValueError("order must be 0..2 and alpha > 0")
        self.order, self.alpha = order, alpha
        self.counts: collections.defaultdict[tuple[str, ...], collections.Counter[str]] = collections.defaultdict(collections.Counter)

    def context(self, history: Sequence[str]) -> tuple[str, ...]:
        if self.order == 0:
            return ()
        return tuple(history[-min(self.order, len(history)):])

    def fit(self, seqs: Iterable[Sequence[str]], exclude_final: bool) -> None:
        self.counts.clear()
        for seq in seqs:
            for history, target in examples(seq, exclude_final):
                self.counts[self.context(history)][target] += 1

    def probs(self, history: Sequence[str]) -> dict[str, float]:
        c = self.counts.get(self.context(history), collections.Counter())
        z = sum(c.values()) + self.alpha * len(VOWELS)
        return {v: (c[v] + self.alpha) / z for v in VOWELS}


@dataclass(frozen=True)
class Metrics:
    targets: int
    nll_nats: float
    bits: float
    perplexity: float
    accuracy: float


def evaluate(model: Model, seqs: Iterable[Sequence[str]], exclude_final: bool) -> Metrics:
    nll = 0.0
    correct = targets = 0
    for seq in seqs:
        for history, target in examples(seq, exclude_final):
            p = model.probs(history)
            nll -= math.log(p[target])
            prediction = max(VOWELS, key=lambda v: (p[v], -VOWELS.index(v)))
            correct += prediction == target
            targets += 1
    if not targets:
        raise ValueError("no evaluation targets")
    bits = nll / targets / math.log(2)
    return Metrics(targets, nll, bits, 2**bits, correct / targets)


def pair_counts(seqs: Iterable[Sequence[str]]) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    for seq in seqs:
        c.update(a + b for a, b in zip(seq, seq[1:]))
    return c


def run(corpus: Corpus, seed: int, alpha: float) -> dict:
    split = split_forms(corpus.form_counts, seed)
    seqs = {k: tuple(vowels(form) for form in forms) for k, forms in split.items()}
    allseq = tuple(vowels(form) for form, n in corpus.form_counts.items() for _ in range(n))
    if any(not seq for seq in allseq):
        raise ValueError("source contains an entry without a/e/i/o/u")

    vf = collections.Counter(v for seq in allseq for v in seq)
    lengths = collections.Counter(map(len, allseq))
    pairs = pair_counts(allseq)
    table6_diff = {pair: {"expected": n, "actual": pairs[pair]} for pair, n in TABLE6.items() if pairs[pair] != n}

    conditions = {}
    for name, exclude_final in (("include-final", False), ("exclude-final-target", True)):
        models = {}
        metrics = {}
        for order in (0, 1, 2):
            model = Model(order, alpha)
            model.fit(seqs["train"], exclude_final)
            models[order] = model
            metrics[order] = evaluate(model, seqs["test"], exclude_final)
        sensitivity = {}
        for a in (0.1, 0.5, 1.0):
            sensitivity[str(a)] = {}
            for order in (0, 1, 2):
                model = Model(order, a)
                model.fit(seqs["train"], exclude_final)
                sensitivity[str(a)][order] = evaluate(model, seqs["test"], exclude_final).bits
        conditions[name] = {"models": models, "metrics": metrics, "sensitivity": sensitivity}

    return {
        "split": split, "seqs": seqs, "allseq": allseq, "vowels": vf, "lengths": lengths,
        "pairs": pairs, "table6_diff": table6_diff, "conditions": conditions,
        "seed": seed, "alpha": alpha,
    }


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def probability_table(model: Model) -> str:
    contexts: list[tuple[str, ...]]
    if model.order == 0:
        contexts = [()]
    elif model.order == 1:
        contexts = [(v,) for v in VOWELS]
    else:
        contexts = [(a, b) for a in VOWELS for b in VOWELS]
    rows = []
    for context in contexts:
        p = model.probs(context)
        rows.append(["".join(context) or "(none)", *(f"{p[v]:.4f}" for v in VOWELS)])
    return table(["context", *(f"P({v})" for v in VOWELS)], rows)


def render(corpus: Corpus, r: Mapping[str, object], source: str) -> str:
    split = r["split"]
    vf = r["vowels"]
    lengths = r["lengths"]
    pairs = r["pairs"]
    conditions = r["conditions"]
    total_vowels = sum(vf.values())
    lines = [
        "# Shona 母音列の低次 Markov 予測評価", "",
        "元4,399語および全件の母音列は収録せず、公開データから再計算できる集計統計だけを記録する。", "",
        "## データと方法", "",
        f"- learning data: <{source}>",
        "- Bruce Hayes & Colin Wilson (2008), *Linguistic Inquiry* 39(3), 379–440, DOI: <https://doi.org/10.1162/ling.2008.39.3.379>",
        f"- source SHA-256: `{corpus.sha256}`",
        f"- 非空ソース行: {corpus.source_rows:,}; count 展開後: {corpus.entries:,}; 完全一致音素列: {len(corpus.form_counts):,}",
        f"- split: 80/10/10、seed `{r['seed']}`。完全一致音素列は同じ split にまとめた。",
        f"- additive smoothing α={r['alpha']}。α=0.1/0.5/1.0 でも感度確認する。",
        "- BOS/EOSは使わない。2次モデルは履歴が1母音しかない最初の遷移だけ1次文脈を使う。", "",
        table(["split", "語エントリ"], [[k, f"{len(v):,}"] for k, v in split.items()]), "",
        "## 基本統計", "", "### 母音数による語長", "",
        table(["母音数", "語数"], [[n, f"{lengths[n]:,}"] for n in sorted(lengths)]), "",
        "### 母音頻度", "",
        table(["母音", "回数", "割合"], [[v, f"{vf[v]:,}", f"{vf[v]/total_vowels:.4f}"] for v in VOWELS]), "",
        "### 5×5 遷移回数", "",
        table(["現在\\次", *VOWELS], [[a, *(f"{pairs[a+b]:,}" for b in VOWELS)] for a in VOWELS]), "",
        "### 5×5 遷移確率", "",
        table(["現在\\次", *VOWELS], [
            [a, *(f"{pairs[a+b]/sum(pairs[a+x] for x in VOWELS):.4f}" for b in VOWELS)] for a in VOWELS
        ]), "", "### Hayes & Wilson Table 6 照合", "",
    ]
    if r["table6_diff"]:
        lines += ["**不一致あり。**", "", table(["pair", "Table 6", "取得データ"], [
            [p, x["expected"], x["actual"]] for p, x in sorted(r["table6_diff"].items())
        ])]
    else:
        lines += ["**25通りすべて一致。** 公開 learning data の隣接母音対は論文 Table 6 を再現した。"]

    lines += ["", "## 予測性能", ""]
    names = {0: "unigram", 1: "1次 Markov", 2: "2次 Markov"}
    for cname, c in conditions.items():
        title = "語末 a を含む通常条件" if cname == "include-final" else "最終母音への予測を除外"
        m = c["metrics"]
        lines += [f"### {title}", "", table(
            ["モデル", "test targets", "NLL (nats)", "bits/token", "perplexity", "top-1 accuracy"],
            [[names[o], f"{m[o].targets:,}", f"{m[o].nll_nats:.3f}", f"{m[o].bits:.4f}", f"{m[o].perplexity:.4f}", f"{m[o].accuracy:.4f}"] for o in (0, 1, 2)]
        ), "",
        f"- unigram → 1次: **{m[0].bits-m[1].bits:+.4f} bits/token**",
        f"- 1次 → 2次: **{m[1].bits-m[2].bits:+.4f} bits/token**", "",
        "#### smoothing 感度", "", table(
            ["α", "unigram", "1次", "2次", "uni→1次", "1次→2次"],
            [[a, f"{c['sensitivity'][a][0]:.4f}", f"{c['sensitivity'][a][1]:.4f}", f"{c['sensitivity'][a][2]:.4f}",
              f"{c['sensitivity'][a][0]-c['sensitivity'][a][1]:+.4f}", f"{c['sensitivity'][a][1]-c['sensitivity'][a][2]:+.4f}"] for a in ("0.1", "0.5", "1.0")]
        ), "", "#### train から推定した予測分布", ""]
        for order in (0, 1, 2):
            lines += [f"**{names[order]}**", "", probability_table(c["models"][order]), ""]

    inc = conditions["include-final"]["metrics"]
    exc = conditions["exclude-final-target"]["metrics"]
    lines += [
        "## 教材適性の判断材料", "",
        f"- 通常条件: unigram→1次 {inc[0].bits-inc[1].bits:+.4f} bits/token、1次→2次 {inc[1].bits-inc[2].bits:+.4f} bits/token。",
        f"- 語末予測除外: unigram→1次 {exc[0].bits-exc[1].bits:+.4f} bits/token、1次→2次 {exc[1].bits-exc[2].bits:+.4f} bits/token。",
        "- 5×5表で母音調和由来の極端に少ない遷移が直接見えるため、頻度→条件付き頻度の導入に使いやすい。",
        "- 採用判断では、1次での改善が明瞭か、2次の追加改善が相対的に小さいか、語末 a を除いても同じ傾向かを確認する。", "",
        "## 再実行", "", "```bash",
        "python scripts/evaluate_shona_markov.py --output docs/reports/shona-markov-evaluation.md --json-output artifacts/shona-markov-results.json",
        "```", "",
    ]
    return "\n".join(lines) + "\n"


def as_json(corpus: Corpus, r: Mapping[str, object], source: str) -> dict:
    return {
        "source_url": source, "source_sha256": corpus.sha256, "source_rows": corpus.source_rows,
        "entries": corpus.entries, "unique_exact_forms": len(corpus.form_counts), "seed": r["seed"],
        "split_sizes": {k: len(v) for k, v in r["split"].items()},
        "vowel_lengths": dict(sorted(r["lengths"].items())),
        "vowel_frequencies": {v: r["vowels"][v] for v in VOWELS},
        "transition_counts": {a+b: r["pairs"][a+b] for a in VOWELS for b in VOWELS},
        "table6_matches": not bool(r["table6_diff"]), "table6_differences": r["table6_diff"],
        "conditions": {name: {
            "metrics": {str(o): vars(c["metrics"][o]) for o in (0, 1, 2)},
            "smoothing_sensitivity_bits": c["sensitivity"],
        } for name, c in r["conditions"].items()},
    }


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tinyxfmr-shona-evaluation/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-url", default=DATA_URL)
    p.add_argument("--data-file", type=Path)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--output", type=Path, default=Path("artifacts/shona-markov-evaluation.md"))
    p.add_argument("--json-output", type=Path, default=Path("artifacts/shona-markov-results.json"))
    args = p.parse_args(argv)

    raw = args.data_file.read_bytes() if args.data_file else fetch(args.data_url)
    corpus = parse(raw)
    result = run(corpus, args.seed, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(corpus, result, args.data_url), encoding="utf-8")
    args.json_output.write_text(json.dumps(as_json(corpus, result, args.data_url), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if corpus.entries != 4399:
        print(f"ERROR: expected 4,399 entries, got {corpus.entries}", file=sys.stderr)
        return 2
    if result["table6_diff"]:
        print(f"ERROR: Hayes & Wilson Table 6 mismatch: {result['table6_diff']}", file=sys.stderr)
        return 3
    print(f"wrote {args.output} and {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
