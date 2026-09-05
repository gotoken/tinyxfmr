#!/usr/bin/env python3
"""Evaluate low-order next-vowel models on Hayes & Wilson's published Shona data.

The source corpus is fetched at run time. tinyxfmr stores only code and aggregate
statistics, never the source lexicon or the full derived vowel sequences.
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
# The paper says 4,399 verbs. The file currently served by the authors has 4,395
# non-empty rows; later work using the Hayes/Wilson Shona data also reports 4,395.
EXPECTED_ROWS = 4395
EXPECTED_SHA256 = "ecee3f5fe3ede70871530eb8a4cd1496b63283b49f568e7116bda61200448fbb"

# Hayes & Wilson (2008), Table 6. This is a reference comparison, not an
# assertion about the currently served file: the two releases differ slightly.
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
    rows: int
    form_counts: Mapping[tuple[str, ...], int]


def parse(raw: bytes) -> Corpus:
    """Parse one space-separated phoneme sequence plus its final frequency field."""
    form_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    rows = 0
    for lineno, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"line {lineno}: expected phonemes followed by a frequency")
        try:
            frequency = int(fields[-1])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: invalid frequency {fields[-1]!r}") from exc
        if frequency <= 0:
            raise ValueError(f"line {lineno}: frequency must be positive")
        form_counts[tuple(fields[:-1])] += frequency
        rows += 1
    return Corpus(hashlib.sha256(raw).hexdigest(), rows, dict(form_counts))


def entry_count(corpus: Corpus) -> int:
    return sum(corpus.form_counts.values())


def vowels(form: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in form if token in VOWEL_SET)


def split_forms(form_counts: Mapping[tuple[str, ...], int], seed: int) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Split 80/10/10 by exact source form, keeping duplicates in one split."""
    groups = list(form_counts.items())
    random.Random(seed).shuffle(groups)
    total = sum(n for _, n in groups)
    train_cut, val_cut = total * 0.8, total * 0.9
    result: dict[str, list[tuple[str, ...]]] = {"train": [], "validation": [], "test": []}
    assigned = 0
    for form, n in groups:
        midpoint = assigned + n / 2
        split = "train" if midpoint < train_cut else "validation" if midpoint < val_cut else "test"
        result[split].extend([form] * n)
        assigned += n
    return {name: tuple(items) for name, items in result.items()}


def examples(seq: Sequence[str], exclude_final: bool) -> Iterable[tuple[tuple[str, ...], str]]:
    for target_index in range(1, len(seq)):
        if exclude_final and target_index == len(seq) - 1:
            continue
        yield tuple(seq[:target_index]), seq[target_index]


class Model:
    """0/1/2-order additive-smoothed Markov model.

    A second-order model also accumulates first-order counts. Therefore the first
    predictable vowel, which has only one preceding vowel and no BOS symbol, uses
    the same first-order statistics rather than gaining an implicit position cue.
    """

    def __init__(self, order: int, alpha: float):
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.counts: collections.defaultdict[tuple[str, ...], collections.Counter[str]] = collections.defaultdict(collections.Counter)

    def fit(self, seqs: Iterable[Sequence[str]], exclude_final: bool) -> None:
        self.counts.clear()
        for seq in seqs:
            for history, target in examples(seq, exclude_final):
                max_width = min(self.order, len(history))
                # Store all usable suffix contexts so lower-order backoff is estimated
                # from all training targets, not only word-initial positions.
                for width in range(max_width + 1):
                    context = () if width == 0 else tuple(history[-width:])
                    self.counts[context][target] += 1

    def context(self, history: Sequence[str]) -> tuple[str, ...]:
        width = min(self.order, len(history))
        return () if width == 0 else tuple(history[-width:])

    def probs(self, history: Sequence[str]) -> dict[str, float]:
        context = self.context(history)
        counts = self.counts.get(context, collections.Counter())
        denominator = sum(counts.values()) + self.alpha * len(VOWELS)
        return {v: (counts[v] + self.alpha) / denominator for v in VOWELS}


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
    result: collections.Counter[str] = collections.Counter()
    for seq in seqs:
        result.update(a + b for a, b in zip(seq, seq[1:]))
    return result


def run(corpus: Corpus, seed: int, alpha: float) -> dict:
    split = split_forms(corpus.form_counts, seed)
    seqs = {name: tuple(vowels(form) for form in forms) for name, forms in split.items()}
    allseq = tuple(vowels(form) for form, n in corpus.form_counts.items() for _ in range(n))
    if any(not seq for seq in allseq):
        raise ValueError("source contains an entry without a/e/i/o/u")

    vowel_counts = collections.Counter(v for seq in allseq for v in seq)
    lengths = collections.Counter(map(len, allseq))
    pairs = pair_counts(allseq)
    table6_diff = {
        pair: {"table6": expected, "current_file": pairs[pair], "delta": pairs[pair] - expected}
        for pair, expected in TABLE6.items() if pairs[pair] != expected
    }

    conditions = {}
    for name, exclude_final in (("include-final", False), ("exclude-final-target", True)):
        models, metrics = {}, {}
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
        "split": split,
        "vowels": vowel_counts,
        "lengths": lengths,
        "pairs": pairs,
        "table6_diff": table6_diff,
        "conditions": conditions,
        "seed": seed,
        "alpha": alpha,
    }


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def probability_table(model: Model) -> str:
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
    split, vc, lengths, pairs = r["split"], r["vowels"], r["lengths"], r["pairs"]
    conditions = r["conditions"]
    total_vowels = sum(vc.values())
    entries = entry_count(corpus)
    lines = [
        "# Shona 母音列の低次 Markov 予測評価", "",
        "Hayes & Wilson の公開 Shona learning data を外部取得して評価した。元データや全件の派生母音列はリポジトリへ収録しない。", "",
        "## データと再現条件", "",
        f"- learning data: <{source}>",
        "- Bruce Hayes & Colin Wilson (2008), *Linguistic Inquiry* 39(3), 379–440, DOI: <https://doi.org/10.1162/ling.2008.39.3.379>",
        f"- 取得ファイル SHA-256: `{corpus.sha256}`",
        f"- 現行公開ファイル: 非空 {corpus.rows:,} 行、frequency 展開後 {entries:,} エントリ、完全一致音素列 {len(corpus.form_counts):,} 種。",
        "- 論文本文は 4,399 動詞と記載する一方、現行公開ファイルは 4,395 行である。2019年の同データ利用研究も Shona の size を 4,395 と報告しているため、現行公開版を 4,395 語版として評価する。",
        "- 2019 reference: <https://aclanthology.org/W19-42.pdf>（Table 1 で Shona = 4,395）。",
        f"- split: train/validation/test = 80/10/10、seed `{r['seed']}`。完全一致音素列は同じ split にまとめ、重複リークを防ぐ。",
        f"- additive smoothing α={r['alpha']}。α=0.1/0.5/1.0 でも感度確認する。",
        "- BOS/EOS は使わない。2次モデルの最初の予測は、全訓練遷移から推定した1次分布へ戻る。", "",
        table(["split", "語エントリ"], [[name, f"{len(forms):,}"] for name, forms in split.items()]), "",
        "## 基本統計", "", "### 母音数による語長", "",
        table(["母音数", "語数"], [[n, f"{lengths[n]:,}"] for n in sorted(lengths)]), "",
        "### 5母音の出現頻度", "",
        table(["母音", "回数", "割合"], [[v, f"{vc[v]:,}", f"{vc[v]/total_vowels:.4f}"] for v in VOWELS]), "",
        "### 5×5 二母音遷移回数", "",
        table(["現在\\次", *VOWELS], [[a, *(f"{pairs[a+b]:,}" for b in VOWELS)] for a in VOWELS]), "",
        "### 5×5 二母音遷移確率", "",
        table(["現在\\次", *VOWELS], [[a, *(f"{pairs[a+b]/sum(pairs[a+x] for x in VOWELS):.4f}" for b in VOWELS)] for a in VOWELS]), "",
        "### Hayes & Wilson (2008) Table 6 との照合", "",
        "現行公開ファイルは論文 Table 6 と完全一致しない。これは失敗条件とはせず、公開版の差として記録する。", "",
    ]
    if r["table6_diff"]:
        lines.append(table(["pair", "Table 6", "現行ファイル", "差"], [
            [pair, values["table6"], values["current_file"], f"{values['delta']:+d}"]
            for pair, values in sorted(r["table6_diff"].items())
        ]))
    else:
        lines.append("25通りすべて一致した。")

    names = {0: "unigram", 1: "1次 Markov", 2: "2次 Markov"}
    lines += ["", "## 予測性能", ""]
    for cname, condition in conditions.items():
        title = "語末 a を含む通常条件" if cname == "include-final" else "最終母音への予測を除外"
        m = condition["metrics"]
        lines += [f"### {title}", "", table(
            ["モデル", "test targets", "NLL (nats)", "cross entropy (bits/token)", "perplexity", "top-1 accuracy"],
            [[names[o], f"{m[o].targets:,}", f"{m[o].nll_nats:.3f}", f"{m[o].bits:.4f}", f"{m[o].perplexity:.4f}", f"{m[o].accuracy:.4f}"] for o in (0, 1, 2)]
        ), "",
        f"- unigram → 1次: **{m[0].bits-m[1].bits:+.4f} bits/token**",
        f"- 1次 → 2次: **{m[1].bits-m[2].bits:+.4f} bits/token**", "",
        "#### smoothing 感度", "", table(
            ["α", "unigram", "1次", "2次", "uni→1次", "1次→2次"],
            [[a, f"{condition['sensitivity'][a][0]:.4f}", f"{condition['sensitivity'][a][1]:.4f}", f"{condition['sensitivity'][a][2]:.4f}",
              f"{condition['sensitivity'][a][0]-condition['sensitivity'][a][1]:+.4f}",
              f"{condition['sensitivity'][a][1]-condition['sensitivity'][a][2]:+.4f}"] for a in ("0.1", "0.5", "1.0")]
        ), "", "#### train から推定した予測分布", ""]
        for order in (0, 1, 2):
            lines += [f"**{names[order]}**", "", probability_table(condition["models"][order]), ""]

    inc, exc = conditions["include-final"]["metrics"], conditions["exclude-final-target"]["metrics"]
    lines += [
        "## 教材としての結論", "",
        f"通常条件では unigram→1次が **{inc[0].bits-inc[1].bits:.4f} bits/token**、1次→2次が **{inc[1].bits-inc[2].bits:.4f} bits/token** 改善する。",
        f"語末への予測を除くと unigram→1次は **{exc[0].bits-exc[1].bits:.4f} bits/token** 改善する一方、1次→2次は **{exc[1].bits-exc[2].bits:.4f} bits/token** に縮む。", "",
        "したがって **Shona 母音列は第1〜2回の主要データとして採用する価値が高い**。特に語末 `-a` の予測を除いた条件は、無条件頻度の限界が大きく、直前1母音を見ることで大幅に改善し、それより長い履歴の追加利得は小さいという教材上望ましい形を示す。", "",
        "5×5表にも `a→o`, `e→o`, `i→e`, `i→o` などゼロまたはほぼゼロのセルが現れ、母音調和を規則の事前説明なしに視覚的に発見できる。語末 `-a` を含めた通常条件では2次の追加利得がやや大きくなるため、教材の主たる評価・可視化では **最終母音への予測を除外**し、語末 `-a` の影響自体を補足実験として示すのが妥当である。", "",
        "注意点として、論文の4,399語/Table 6と現行公開4,395語ファイルには小さな差がある。教材では『Hayes & Wilson (2008) が公開した現行 learning data 4,395行を利用した』と版を明示し、論文表との不一致を隠さない。", "",
        "## 再実行", "", "```bash",
        "python scripts/evaluate_shona_markov.py --output docs/reports/shona-markov-evaluation.md --json-output artifacts/shona-markov-results.json",
        "```", "",
    ]
    return "\n".join(lines) + "\n"


def as_json(corpus: Corpus, r: Mapping[str, object], source: str) -> dict:
    return {
        "source_url": source,
        "source_sha256": corpus.sha256,
        "source_rows": corpus.rows,
        "entries": entry_count(corpus),
        "unique_exact_forms": len(corpus.form_counts),
        "seed": r["seed"],
        "split_sizes": {name: len(forms) for name, forms in r["split"].items()},
        "vowel_lengths": dict(sorted(r["lengths"].items())),
        "vowel_frequencies": {v: r["vowels"][v] for v in VOWELS},
        "transition_counts": {a+b: r["pairs"][a+b] for a in VOWELS for b in VOWELS},
        "table6_matches": not bool(r["table6_diff"]),
        "table6_differences": r["table6_diff"],
        "conditions": {name: {
            "metrics": {str(order): vars(condition["metrics"][order]) for order in (0, 1, 2)},
            "smoothing_sensitivity_bits": condition["sensitivity"],
        } for name, condition in r["conditions"].items()},
    }


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tinyxfmr-shona-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--output", type=Path, default=Path("artifacts/shona-markov-evaluation.md"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/shona-markov-results.json"))
    parser.add_argument("--accept-source-change", action="store_true", help="run even if the public source fingerprint changed")
    args = parser.parse_args(argv)

    raw = args.data_file.read_bytes() if args.data_file else fetch(args.data_url)
    corpus = parse(raw)
    if not args.accept_source_change and (corpus.rows != EXPECTED_ROWS or corpus.sha256 != EXPECTED_SHA256):
        print(
            f"ERROR: public source changed: rows={corpus.rows}, sha256={corpus.sha256}; "
            f"expected rows={EXPECTED_ROWS}, sha256={EXPECTED_SHA256}", file=sys.stderr,
        )
        return 2

    result = run(corpus, args.seed, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(corpus, result, args.data_url), encoding="utf-8")
    args.json_output.write_text(json.dumps(as_json(corpus, result, args.data_url), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
