from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "threads_results.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pseudolabeled_threads_5000.csv"
DEFAULT_SUMMARY = ROOT / "output" / "pseudolabel_summary_5000.json"
DEFAULT_RAW_GLOB = str(ROOT / "data" / "raw" / "*.json")


POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "bagus",
    "baik",
    "beautiful",
    "best",
    "brilliant",
    "cool",
    "enak",
    "excellent",
    "excited",
    "fantastic",
    "good",
    "great",
    "happy",
    "hebat",
    "incredible",
    "keren",
    "love",
    "mantap",
    "nice",
    "perfect",
    "powerful",
    "recommend",
    "solid",
    "suka",
    "super",
    "terbaik",
    "thank",
    "thanks",
    "wow",
}


NEGATIVE_WORDS = {
    "annoying",
    "bad",
    "benci",
    "boring",
    "broken",
    "crash",
    "disappoint",
    "disappointed",
    "error",
    "fail",
    "fake",
    "frustrating",
    "hate",
    "horrible",
    "jelek",
    "kecewa",
    "lambat",
    "lag",
    "problem",
    "rusak",
    "scam",
    "slow",
    "terrible",
    "toxic",
    "ugly",
    "worse",
    "worst",
}


TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply pseudo-label sentiment to scraped Threads data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV path from scraping result.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for pseudo-labeled data.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output JSON path for label distribution summary.",
    )
    parser.add_argument(
        "--raw-glob",
        type=str,
        default=DEFAULT_RAW_GLOB,
        help="Glob pattern for raw JSON files used when CSV rows are insufficient.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of rows to pseudo-label.",
    )
    return parser.parse_args()


def tokenize(text: str) -> Iterable[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_row(raw: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": str(raw.get("id") or ""),
        "username": str(raw.get("username") or ""),
        "text": str(raw.get("text") or ""),
        "like_count": float(raw.get("like_count") or 0.0),
        "reply_count": float(raw.get("reply_count") or 0.0),
        "repost_count": float(raw.get("repost_count") or 0.0),
        "created_at": raw.get("created_at") or "",
        "url": str(raw.get("url") or ""),
    }


def collect_seed_rows(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    required_cols = {"text", "like_count", "reply_count", "repost_count"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {', '.join(missing_cols)}")

    for col in ["id", "username", "created_at", "url"]:
        if col not in df.columns:
            df[col] = ""

    df["text"] = df["text"].fillna("").astype(str)
    for col in ["like_count", "reply_count", "repost_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return [normalize_row(record) for record in df.to_dict(orient="records")]


def collect_raw_rows(raw_glob: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pattern = raw_glob
    if not Path(pattern).is_absolute():
        pattern = str(ROOT / pattern)

    for path_str in sorted(glob.glob(pattern)):
        json_path = Path(path_str)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                records = payload["items"]
            elif isinstance(payload.get("data"), list):
                records = payload["data"]
            else:
                records = [payload]
        else:
            records = []

        for record in records:
            if isinstance(record, dict):
                rows.append(normalize_row(record))
    return rows


def label_text(text: str, like_count: float, reply_count: float, repost_count: float) -> Dict[str, float | str]:
    tokens = list(tokenize(text))

    pos_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    lexical_score = float(pos_hits - neg_hits)

    engagement_total = max(0.0, like_count) + (2.0 * max(0.0, reply_count)) + (2.0 * max(0.0, repost_count))
    engagement_score = math.log1p(engagement_total) / 5.0

    if lexical_score > 0:
        combined_score = lexical_score + (0.2 * engagement_score)
    elif lexical_score < 0:
        combined_score = lexical_score - (0.2 * engagement_score)
    else:
        combined_score = 0.1 * engagement_score

    if combined_score >= 1.0:
        label = "positive"
    elif combined_score <= -1.0:
        label = "negative"
    else:
        label = "neutral"

    confidence = min(0.99, 0.50 + min(3.0, abs(combined_score)) / 5.0)

    return {
        "pseudo_label": label,
        "pseudo_score": round(combined_score, 4),
        "pseudo_confidence": round(confidence, 4),
        "positive_hits": int(pos_hits),
        "negative_hits": int(neg_hits),
    }


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    seed_rows = collect_seed_rows(args.input)
    raw_rows = collect_raw_rows(args.raw_glob) if len(seed_rows) < args.limit else []

    combined_rows: List[Dict[str, object]] = []
    seen_ids = set()

    def add_rows(rows: List[Dict[str, object]]) -> None:
        for row in rows:
            item_id = str(row.get("id") or "")
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            combined_rows.append(row)
            if len(combined_rows) >= args.limit:
                break

    add_rows(seed_rows)
    if len(combined_rows) < args.limit:
        add_rows(raw_rows)

    if len(combined_rows) < args.limit:
        raise ValueError(
            f"Requested {args.limit} rows but only found {len(combined_rows)} unique rows from CSV+raw sources"
        )

    working_df = pd.DataFrame(combined_rows[: args.limit]).copy()

    labels = working_df.apply(
        lambda row: label_text(
            text=row["text"],
            like_count=float(row["like_count"]),
            reply_count=float(row["reply_count"]),
            repost_count=float(row["repost_count"]),
        ),
        axis=1,
    )

    label_df = pd.DataFrame(labels.tolist())
    result_df = pd.concat([working_df.reset_index(drop=True), label_df], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    distribution = result_df["pseudo_label"].value_counts(dropna=False).to_dict()
    summary_payload = {
        "input": str(args.input),
        "raw_glob": args.raw_glob,
        "output": str(args.output),
        "rows_labeled": int(len(result_df)),
        "seed_rows_from_csv": int(len(seed_rows)),
        "rows_scanned_from_raw": int(len(raw_rows)),
        "label_distribution": distribution,
        "average_confidence": float(round(result_df["pseudo_confidence"].mean(), 4)),
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()