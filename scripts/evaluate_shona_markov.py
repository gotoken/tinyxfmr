#!/usr/bin/env python3
"""Evaluate low-order Markov next-vowel models on Hayes & Wilson's Shona data.

The source corpus is fetched at evaluation time and is intentionally not vendored in
this repository. The script writes only aggregate statistics and model metrics.
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
DEFAULT_URL = "https://brucehayes.org/Phonotactics/files/ShonaLearningData.txt"
DEFAULT_SEED = 20260905
DEFAULT_ALPHA = 0.5
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# Hayes & Wilson (2008), Table 6. The published table orders rows a,e,o,i,u;
# we store the same raw counts by pair while rendering tables in VOWELS order.
HAYES_WILSON_TABLE6: dict[tuple[str, str], int] = {
    ("a", "a"): 1443, ("a", "e"): 3, ("a", "o"): 0, ("a", "i"): 500, ("a", "u"): 568,
    ("e", "a"): 639, ("e", "e"): 587, ("e", "o"): 0, ("e", "i"): 2, ("e", "u"): 260,
    ("o", "a"): 638, ("o", "e"): 153, ("o", "o"): 694, ("o", "i"): 23, ("o", "u"): 20,
    ("i", "a"): 1130, ("i", "e"): 0, ("i", "o"): 0, ("i", "i"): 478, ("i", "u"): 175,
    ("u", "a"): 1737, ("u", "e"): 4, ("u", "o"): 1, ("u", "i"): 175, ("u", "u"): 811,
}


@dataclass(frozen=True)
class Corpus:
    raw_sha256: str
    source_rows: int
    weighted_entries: int
    forms: tuple[tuple[str, ...], ...]
    form_counts: Mapping[tuple[str, ...], int]


def parse_learning_data(raw: bytes) -> Corpus:
    """Parse whitespace-separated phoneme rows followed by an integer weight.

    Exact phoneme-token sequences are grouped so duplicate dictionary rows never cross
    train/validation/test boundaries. The final integer is treated as a multiplicity.
    """
    text = raw.decode("utf-8-sig")
    form_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    source_rows = 0
    weighted_entries = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"line {lineno}: expected phonemes followed by a count")
        try:
            weight = int(fields[-1])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: final field is not an integer count: {fields[-1]!r}") from exc
        if weight <= 0:
            raise ValueError(f"line {lineno}: count must be positive")
        phonemes = tuple(fields[:-1])
        if not phonemes:
            raise ValueError(f"line {lineno}: empty phoneme sequence")
        form_counts[phonemes] += weight
        source_rows += 1
        weighted_entries += weight

    forms: list[tuple[str, ...]] = []
    for form, count in form_counts.items():
        forms.extend([form] * count)
    return Corpus(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        source_rows=source_rows,
        weighted_entries=weighted_entries,
        forms=tuple(forms),
        form_counts=dict(form_counts),
    )


def extract_vowels(form: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in form if token in VOWEL_SET)


def split_grouped_forms(
    form_counts: Mapping[tuple[str, ...], int],
    seed: int,
    ratios: Sequence[float] = SPLIT_RATIOS,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Split by exact source form while approximately preserving requested entry ratios."""
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ratios must contain train/validation/test fractions summing to 1")
    groups = list(form_counts.items())
    random.Random(seed).shuffle(groups)
    total = sum(count for _, count in groups)
    target_train = total * ratios[0]
    target_val = total * (ratios[0] + ratios[1])
    buckets: dict[str, list[tuple[str, ...]]] = {"train": [], "validation": [], "test": []}
    assigned = 0
    for form, count in groups:
        midpoint = assigned + count / 2
        if midpoint < target_train:
            bucket = "train"
        elif midpoint < target_val:
            bucket = "validation"
        else:
            bucket = "test"
        buckets[bucket].extend([form] * count)
        assigned += count
    return {name: tuple(items) for name, items in buckets.items()}


def sequence_targets(vowels: Sequence[str], exclude_final_target: bool) -> list[tuple[tuple[str, ...], str]]:
    """Return (history, target) pairs for all positions that have at least one prior vowel."""
    result: list[tuple[tuple[str, ...], str]] = []
    for target_index in range(1, len(vowels)):
        if exclude_final_target and target_index == len(vowels) - 1:
            continue
        result.append((tuple(vowels[:target_index]), vowels[target_index]))
    return result


class MarkovModel:
    """Maximum-context n-gram model with additive smoothing and shorter-history fallback.

    order=0 is an unconditional next-vowel distribution. order=1 conditions on the
    current vowel. order=2 uses the previous two vowels when available, and the last
    vowel for the first transition because the experiment intentionally has no BOS token.
    """

    def __init__(self, order: int, alpha: float = DEFAULT_ALPHA):
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.counts: collections.defaultdict[tuple[str, ...], collections.Counter[str]] = collections.defaultdict(collections.Counter)

    def _context(self, history: Sequence[str]) -> tuple[str, ...]:
        if self.order == 0:
            return ()
        width = min(self.order, len(history))
        return tuple(history[-width:])

    def fit(self, vowel_sequences: Iterable[Sequence[str]], exclude_final_target: bool) -> None:
        self.counts.clear()
        for seq in vowel_sequences:
            for history, target in sequence_targets(seq, exclude_final_target):
                self.counts[self._context(history)][target] += 1

    def distribution(self, history: Sequence[str]) -> dict[str, float]:
        context = self._context(history)
        counts = self.counts.get(context, collections.Counter())
        denominator = sum(counts.values()) + self.alpha * len(VOWELS)
        return {vowel: (counts[vowel] + self.alpha) / denominator for vowel in VOWELS}


@dataclass(frozen=True)
class Metrics:
    targets: int
    nll_nats: float
    cross_entropy_bits: float
    perplexity: float
    accuracy: float


def evaluate(model: MarkovModel, vowel_sequences: Iterable[Sequence[str]], exclude_final_target: bool) -> Metrics:
    nll = 0.0
    correct = 0
    targets = 0
    vowel_rank = {v: i for i, v in enumerate(VOWELS)}
    for seq in vowel_sequences:
        for history, target in sequence_targets(seq, exclude_final_target):
            distribution = model.distribution(history)
            probability = distribution[target]
            nll -= math.log(probability)
            prediction = max(VOWELS, key=lambda v: (distribution[v], -vowel_rank[v]))
            correct += prediction == target
            targets += 1
    if targets == 0:
        raise ValueError("no evaluation targets")
    ce_nats = nll / targets
    return Metrics(
        targets=targets,
        nll_nats=nll,
        cross_entropy_bits=ce_nats / math.log(2),
        perplexity=math.exp(ce_nats),
        accuracy=correct / targets,
    )


def vowel_length_distribution(sequences: Iterable[Sequence[str]]) -> collections.Counter[int]:
    return collections.Counter(len(seq) for seq in sequences)


def vowel_frequencies(sequences: Iterable[Sequence[str]]) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for seq in sequences:
        result.update(seq)
    return result


def transition_counts(sequences: Iterable[Sequence[str]]) -> collections.Counter[tuple[str, str]]:
    result: collections.Counter[tuple[str, str]] = collections.Counter()
    for seq in sequences:
        result.update(zip(seq, seq[1:]))
    return result


def transition_probabilities(counts: Mapping[tuple[str, str], int]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for current in VOWELS:
        total = sum(counts.get((current, nxt), 0) for nxt in VOWELS)
        for nxt in VOWELS:
            result[(current, nxt)] = counts.get((current, nxt), 0) / total if total else 0.0
    return result


def table6_differences(counts: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    differences = {}
    for pair, expected in HAYES_WILSON_TABLE6.items():
        actual = counts.get(pair, 0)
        if actual != expected:
            differences[" ".join(pair)] = {"expected": expected, "actual": actual}
    return differences


def md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def model_probability_rows(model: MarkovModel) -> list[list[str]]:
    rows: list[list[str]] = []
    if model.order == 0:
        dist = model.distribution(())
        rows.append(["(none)"] + [f"{dist[v]:.4f}" for v in VOWELS])
    elif model.order == 1:
        for current in VOWELS:
            dist = model.distribution((current,))
            rows.append([current] + [f"{dist[v]:.4f}" for v in VOWELS])
    else:
        for prev in VOWELS:
            for current in VOWELS:
                dist = model.distribution((prev, current))
                rows.append([prev + current] + [f"{dist[v]:.4f}" for v in VOWELS])
    return rows


def run_analysis(corpus: Corpus, seed: int, alpha: float) -> dict:
    splits = split_grouped_forms(corpus.form_counts, seed)
    vowel_splits = {name: tuple(extract_vowels(form) for form in forms) for name, forms in splits.items()}
    all_vowels = tuple(extract_vowels(form) for form in corpus.forms)
    empty_vowel_entries = sum(not seq for seq in all_vowels)
    if empty_vowel_entries:
        raise ValueError(f"{empty_vowel_entries} entries contain no a/e/i/o/u vowels")

    full_counts = transition_counts(all_vowels)
    table6_diff = table6_differences(full_counts)
    conditions: dict[str, dict] = {}
    for condition_name, exclude_final in (("include-final", False), ("exclude-final-target", True)):
        models = {order: MarkovModel(order, alpha) for order in (0, 1, 2)}
        metrics: dict[int, Metrics] = {}
        for order, model in models.items():
            model.fit(vowel_splits["train"], exclude_final)
            metrics[order] = evaluate(model, vowel_splits["test"], exclude_final)
        sensitivity = {}
        for sensitivity_alpha in (0.1, 0.5, 1.0):
            sensitivity_metrics = {}
            for order in (0, 1, 2):
                model = MarkovModel(order, sensitivity_alpha)
                model.fit(vowel_splits["train"], exclude_final)
                sensitivity_metrics[order] = evaluate(model, vowel_splits["test"], exclude_final)
            sensitivity[str(sensitivity_alpha)] = sensitivity_metrics
        conditions[condition_name] = {
            "exclude_final": exclude_final,
            "models": models,
            "metrics": metrics,
            "sensitivity": sensitivity,
        }

    return {
        "corpus": corpus,
        "splits": splits,
        "vowel_splits": vowel_splits,
        "all_vowels": all_vowels,
        "vowel_lengths": vowel_length_distribution(all_vowels),
        "vowel_freqs": vowel_frequencies(all_vowels),
        "transition_counts": full_counts,
        "transition_probs": transition_probabilities(full_counts),
        "table6_diff": table6_diff,
        "conditions": conditions,
        "seed": seed,
        "alpha": alpha,
    }


def render_report(result: Mapping[str, object], source_url: str) -> str:
    corpus: Corpus = result["corpus"]  # type: ignore[assignment]
    splits = result["splits"]  # type: ignore[assignment]
    vowel_lengths = result["vowel_lengths"]  # type: ignore[assignment]
    vowel_freqs = result["vowel_freqs"]  # type: ignore[assignment]
    counts = result["transition_counts"]  # type: ignore[assignment]
    probs = result["transition_probs"]  # type: ignore[assignment]
    conditions = result["conditions"]  # type: ignore[assignment]
    table6_diff = result["table6_diff"]  # type: ignore[assignment]
    seed = result["seed"]
    alpha = result["alpha"]

    duplicate_extra = corpus.weighted_entries - len(corpus.form_counts)
    lines = [
        "# Shona 母音列の低次 Markov 予測評価",
        "",
        "このレポートは `scripts/evaluate_shona_markov.py` が生成した集計結果である。元の4,399語および全件の母音列は再配布しない。",
        "",
        "## データと再現条件",
        "",
        f"- 公開元: <{source_url}>",
        "- 出典: Bruce Hayes & Colin Wilson, \"A Maximum Entropy Model of Phonotactics and Phonotactic Learning\", *Linguistic Inquiry* 39(3), 379–440, 2008.",
        "- DOI: <https://doi.org/10.1162/ling.2008.39.3.379>",
        f"- 元ファイル SHA-256: `{corpus.raw_sha256}`",
        f"- 非空ソース行: {corpus.source_rows:,}",
        f"- count を展開した語エントリ: {corpus.weighted_entries:,}",
        f"- 完全一致する音素列をまとめた種類数: {len(corpus.form_counts):,}（重複分 {duplicate_extra:,}）",
        f"- split: train/validation/test = 80/10/10、seed = `{seed}`",
        "- 重複語形のリークを避けるため、完全一致する音素列は必ず同じ split に入れる。",
        f"- 主評価の smoothing: additive smoothing α = {alpha}",
        "- 2次モデルは履歴が1母音しかない最初の遷移では1次の文脈へ短縮する。BOS/EOS記号は加えない。",
        "",
        "### split 件数",
        "",
        md_table(["split", "語エントリ"], [[name, f"{len(forms):,}"] for name, forms in splits.items()]),
        "",
        "## 基本統計",
        "",
        "### 母音数による語長分布",
        "",
        md_table(["母音数", "語数"], [[length, f"{vowel_lengths[length]:,}"] for length in sorted(vowel_lengths)]),
        "",
        "### 5母音の出現頻度",
        "",
        md_table(["母音", "回数", "割合"], [
            [v, f"{vowel_freqs[v]:,}", f"{vowel_freqs[v] / sum(vowel_freqs.values()):.4f}"] for v in VOWELS
        ]),
        "",
        "### 5×5 二母音遷移回数",
        "",
        md_table(["現在\\次"] + list(VOWELS), [
            [current] + [f"{counts.get((current, nxt), 0):,}" for nxt in VOWELS] for current in VOWELS
        ]),
        "",
        "### 5×5 二母音遷移確率",
        "",
        md_table(["現在\\次"] + list(VOWELS), [
            [current] + [f"{probs[(current, nxt)]:.4f}" for nxt in VOWELS] for current in VOWELS
        ]),
        "",
        "### Hayes & Wilson Table 6 との照合",
        "",
    ]
    if table6_diff:
        lines.append("**不一致あり。** 以下の組み合わせが論文 Table 6 と一致しなかった。")
        lines.append("")
        lines.append(md_table(["pair", "Table 6", "取得データ"], [
            [pair, values["expected"], values["actual"]] for pair, values in sorted(table6_diff.items())
        ]))
    else:
        lines.append("**25通りすべて一致した。** 公開 learning data から母音だけを抽出して得た隣接母音対の回数は Hayes & Wilson (2008) Table 6 を再現する。")

    lines.extend(["", "## 予測性能", ""])
    for condition_name, condition in conditions.items():
        title = "語末 a を含む通常条件" if condition_name == "include-final" else "最終母音への予測を除外"
        metrics = condition["metrics"]
        lines.extend([
            f"### {title}",
            "",
            md_table(
                ["モデル", "test targets", "NLL (nats)", "cross entropy (bits/token)", "perplexity", "top-1 accuracy"],
                [
                    [
                        {0: "unigram", 1: "1次 Markov", 2: "2次 Markov"}[order],
                        f"{metrics[order].targets:,}",
                        f"{metrics[order].nll_nats:.3f}",
                        f"{metrics[order].cross_entropy_bits:.4f}",
                        f"{metrics[order].perplexity:.4f}",
                        f"{metrics[order].accuracy:.4f}",
                    ]
                    for order in (0, 1, 2)
                ],
            ),
            "",
            f"- unigram → 1次: {metrics[0].cross_entropy_bits - metrics[1].cross_entropy_bits:+.4f} bits/token 改善",
            f"- 1次 → 2次: {metrics[1].cross_entropy_bits - metrics[2].cross_entropy_bits:+.4f} bits/token 改善",
            "",
            "#### smoothing 感度",
            "",
            md_table(
                ["α", "unigram bits/token", "1次 bits/token", "2次 bits/token", "uni→1次改善", "1次→2次改善"],
                [
                    [
                        a,
                        f"{condition['sensitivity'][a][0].cross_entropy_bits:.4f}",
                        f"{condition['sensitivity'][a][1].cross_entropy_bits:.4f}",
                        f"{condition['sensitivity'][a][2].cross_entropy_bits:.4f}",
                        f"{condition['sensitivity'][a][0].cross_entropy_bits - condition['sensitivity'][a][1].cross_entropy_bits:+.4f}",
                        f"{condition['sensitivity'][a][1].cross_entropy_bits - condition['sensitivity'][a][2].cross_entropy_bits:+.4f}",
                    ]
                    for a in ("0.1", "0.5", "1.0")
                ],
            ),
            "",
            "#### train から推定した予測分布",
            "",
        ])
        for order in (0, 1, 2):
            model = condition["models"][order]
            lines.extend([
                f"**{{0: 'unigram', 1: '1次 Markov', 2: '2次 Markov'}[order]}**",
                "",
                md_table(["context"] + [f"P({v})" for v in VOWELS], model_probability_rows(model)),
                "",
            ])

    include_metrics = conditions["include-final"]["metrics"]
    exclude_metrics = conditions["exclude-final-target"]["metrics"]
    include_gain_1 = include_metrics[0].cross_entropy_bits - include_metrics[1].cross_entropy_bits
    include_gain_2 = include_metrics[1].cross_entropy_bits - include_metrics[2].cross_entropy_bits
    exclude_gain_1 = exclude_metrics[0].cross_entropy_bits - exclude_metrics[1].cross_entropy_bits
    exclude_gain_2 = exclude_metrics[1].cross_entropy_bits - exclude_metrics[2].cross_entropy_bits

    lines.extend([
        "## 教材適性の判断材料",
        "",
        f"- 通常条件の unigram→1次改善: **{include_gain_1:.4f} bits/token**。",
        f"- 通常条件の 1次→2次改善: **{include_gain_2:.4f} bits/token**。",
        f"- 語末予測除外時の unigram→1次改善: **{exclude_gain_1:.4f} bits/token**。",
        f"- 語末予測除外時の 1次→2次改善: **{exclude_gain_2:.4f} bits/token**。",
        "- 5×5遷移表では、論文が説明する母音調和に対応して極端に少ない遷移がそのまま見える。",
        "- smoothing の α を 0.1 / 0.5 / 1.0 に変えたときも改善の符号と大きさが大きく変わらないかを上表で確認する。",
        "",
        "最終的な採用判断は、上記の実測値を教材ストーリー（unigram の限界→直前母音を見る必要性→さらに長い文脈の必要性）と照合して行う。",
        "",
        "## 再実行",
        "",
        "```bash",
        "python scripts/evaluate_shona_markov.py --output docs/reports/shona-markov-evaluation.md --json-output artifacts/shona-markov-results.json",
        "```",
        "",
        "GitHub Actions では同じスクリプトを固定 seed で実行し、Markdown と JSON を artifact として保存する。",
    ])
    return "\n".join(lines) + "\n"


def jsonable_result(result: Mapping[str, object], source_url: str) -> dict:
    corpus: Corpus = result["corpus"]  # type: ignore[assignment]
    counts = result["transition_counts"]  # type: ignore[assignment]
    probs = result["transition_probs"]  # type: ignore[assignment]
    conditions = result["conditions"]  # type: ignore[assignment]
    return {
        "source_url": source_url,
        "source_sha256": corpus.raw_sha256,
        "source_rows": corpus.source_rows,
        "weighted_entries": corpus.weighted_entries,
        "unique_exact_forms": len(corpus.form_counts),
        "seed": result["seed"],
        "split_sizes": {name: len(forms) for name, forms in result["splits"].items()},  # type: ignore[union-attr]
        "vowel_length_distribution": {str(k): v for k, v in sorted(result["vowel_lengths"].items())},  # type: ignore[union-attr]
        "vowel_frequencies": {v: result["vowel_freqs"][v] for v in VOWELS},  # type: ignore[index]
        "transition_counts": {f"{a}{b}": counts.get((a, b), 0) for a in VOWELS for b in VOWELS},
        "transition_probabilities": {f"{a}{b}": probs[(a, b)] for a in VOWELS for b in VOWELS},
        "table6_matches": not bool(result["table6_diff"]),
        "table6_differences": result["table6_diff"],
        "conditions": {
            name: {
                "metrics": {
                    str(order): {
                        "targets": condition["metrics"][order].targets,
                        "nll_nats": condition["metrics"][order].nll_nats,
                        "cross_entropy_bits": condition["metrics"][order].cross_entropy_bits,
                        "perplexity": condition["metrics"][order].perplexity,
                        "accuracy": condition["metrics"][order].accuracy,
                    }
                    for order in (0, 1, 2)
                },
                "smoothing_sensitivity": {
                    alpha: {
                        str(order): condition["sensitivity"][alpha][order].cross_entropy_bits
                        for order in (0, 1, 2)
                    }
                    for alpha in ("0.1", "0.5", "1.0")
                },
            }
            for name, condition in conditions.items()
        },
    }


def fetch_source(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tinyxfmr-shona-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-url", default=DEFAULT_URL)
    parser.add_argument("--data-file", type=Path, help="Use a locally downloaded source file instead of fetching the URL")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--output", type=Path, default=Path("artifacts/shona-markov-evaluation.md"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/shona-markov-results.json"))
    args = parser.parse_args(argv)

    raw = args.data_file.read_bytes() if args.data_file else fetch_source(args.data_url)
    corpus = parse_learning_data(raw)
    result = run_analysis(corpus, args.seed, args.alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(result, args.data_url), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(jsonable_result(result, args.data_url), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if corpus.weighted_entries != 4399:
        print(f"ERROR: expected 4,399 weighted entries, got {corpus.weighted_entries}", file=sys.stderr)
        return 2
    if result["table6_diff"]:
        print(f"ERROR: Table 6 mismatch: {result['table6_diff']}", file=sys.stderr)
        return 3
    print(f"wrote {args.output} and {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
