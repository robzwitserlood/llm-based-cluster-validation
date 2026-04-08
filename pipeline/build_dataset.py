"""
pipeline/build_dataset.py — generate raw_examples.json from real BERTopic clusters.

Usage:
    python pipeline/build_dataset.py
    python pipeline/build_dataset.py --max-docs 10000 --examples-per-topic 4
    python pipeline/build_dataset.py --skip-bertopic   # re-generate from existing topics.json

Steps:
1. Stream stanford-oval/ccnews (2024 config) from HuggingFace, keeping only
   Dutch (language == "nl") articles, up to --max-docs plain_text values.
2. Run BERTopic (multilingual embeddings, Dutch stop words) to extract topics.
3. Save topic representations (top-N keywords per topic) to --out (default: pipeline/topics.json).
4. Build raw examples: for each topic, emit --examples-per-topic rows where
   five keywords come from that topic and one intruder keyword is drawn from a
   randomly chosen *different* topic.
5. Save examples to --examples-out (default: pipeline/raw_examples.json).
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
OUTPUTS_DIR = REPO_DIR / "outputs"

DUTCH_STOP_WORDS = [
    "de", "het", "een", "en", "van", "in", "is", "dat", "op", "te",
    "zijn", "er", "maar", "om", "met", "aan", "voor", "niet", "ook",
    "als", "bij", "nog", "dan", "naar", "zo", "kan", "worden", "door",
    "wel", "hebben", "heeft", "was", "worden", "ze", "dit", "die",
    "worden", "worden", "hij", "ze", "we", "ik", "je", "u", "ze",
    "hun", "hem", "haar", "ons", "jullie", "hen", "zij", "wij",
]


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def build_topic_model() -> BERTopic:
    umap_model = UMAP(n_neighbors=5, n_components=3, min_dist=0.05, random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=60, min_samples=30, prediction_data=True, gen_min_span_tree=True,
    )
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words=DUTCH_STOP_WORDS, max_df=0.8, min_df=5,
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        language="dutch",
        calculate_probabilities=True,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dutch_docs(max_docs: int = 5_000, min_text_len: int = 30) -> list[str]:
    print(f"Streaming stanford-oval/ccnews (2024, nl) — collecting up to {max_docs} docs …")
    ds = load_dataset("stanford-oval/ccnews", "2024", split="train", streaming=True)
    docs: list[str] = []
    scanned = 0
    for row in ds:
        scanned += 1
        if row.get("language") == "nl":
            text = row.get("plain_text") or ""
            if len(text) > min_text_len:
                docs.append(text)
                if len(docs) >= max_docs:
                    break
        if scanned % 50_000 == 0:
            print(f"  … scanned {scanned:,} rows, collected {len(docs):,} Dutch docs")
    print(f"  Done — {len(docs)} Dutch documents from {scanned:,} rows scanned.")
    return docs


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def extract_topics(docs: list[str], top_n: int = 10) -> dict[int, list[str]]:
    """Return {topic_id: [keyword, ...]} for all non-outlier topics."""
    model = build_topic_model()
    topics, _ = model.fit_transform(docs)
    topic_info = model.get_topic_info()
    topic_keywords: dict[int, list[str]] = {}
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words_scores = model.get_topic(tid)
        keywords = [w for w, _ in words_scores[:top_n]]
        if keywords:
            topic_keywords[tid] = keywords
    return topic_keywords


def save_topics(topic_keywords: dict[int, list[str]], path: Path) -> None:
    serialisable = {str(k): v for k, v in topic_keywords.items()}
    path.write_text(json.dumps(serialisable, indent=2))
    print(f"Topic representations saved to {path}")


def load_topics(path: Path) -> dict[int, list[str]]:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Build examples
# ---------------------------------------------------------------------------

def build_raw_examples(
    topic_keywords: dict[int, list[str]],
    examples_per_topic: int = 4,
    seed: int = 42,
) -> list[dict]:
    """For each topic, generate `examples_per_topic` intruder-detection rows."""
    rng = random.Random(seed)
    topic_ids = list(topic_keywords.keys())

    if len(topic_ids) < 2:
        raise ValueError(
            f"Need at least 2 topics to create intruder examples, got {len(topic_ids)}."
        )

    examples: list[dict] = []
    for tid in topic_ids:
        pool = topic_keywords[tid]
        if len(pool) < 5:
            print(f"  Skipping topic {tid}: only {len(pool)} keywords (need ≥5).")
            continue

        other_ids = [t for t in topic_ids if t != tid]
        for _ in range(examples_per_topic):
            cluster_kws = rng.sample(pool, 5)
            intruder_topic = rng.choice(other_ids)
            intruder_kw = rng.choice(topic_keywords[intruder_topic])
            insert_pos = rng.randint(0, 5)
            six = cluster_kws[:insert_pos] + [intruder_kw] + cluster_kws[insert_pos:]
            examples.append({
                "keyword_1": six[0], "keyword_2": six[1], "keyword_3": six[2],
                "keyword_4": six[3], "keyword_5": six[4], "keyword_6": six[5],
                "intruder": intruder_kw,
                "source_topic": tid,
                "intruder_topic": intruder_topic,
            })
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_build_dataset(
    out: Path = DATA_DIR / "topics.json",
    examples_out: Path = DATA_DIR / "raw_examples.json",
    max_docs: int = 5_000,
    examples_per_topic: int = 4,
    top_n: int = 10,
    seed: int = 42,
    skip_bertopic: bool = False,
) -> list[dict]:
    """Build the dataset. Returns the list of raw examples."""
    if skip_bertopic:
        print(f"Loading existing topics from {out} …")
        topic_keywords = load_topics(Path(out))
    else:
        docs = load_dutch_docs(max_docs=max_docs)
        topic_keywords = extract_topics(docs, top_n=top_n)
        save_topics(topic_keywords, Path(out))

    print(f"\nFound {len(topic_keywords)} topics:")
    for tid, kws in sorted(topic_keywords.items()):
        print(f"  Topic {tid}: {kws}")

    examples = build_raw_examples(topic_keywords, examples_per_topic=examples_per_topic, seed=seed)
    Path(examples_out).write_text(json.dumps(examples, indent=2))
    print(f"{len(examples)} examples saved to {examples_out}")
    print("\nDone. Run `python pipeline/evaluate.py` to evaluate ClusterIntruderValidator.")
    return examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",                default=str(DATA_DIR / "topics.json"))
    p.add_argument("--examples-out",       default=str(DATA_DIR / "raw_examples.json"))
    p.add_argument("--max-docs",           type=int, default=5_000)
    p.add_argument("--examples-per-topic", type=int, default=4)
    p.add_argument("--top-n",              type=int, default=10)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--skip-bertopic",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_build_dataset(**{k.replace("-", "_"): v for k, v in vars(args).items()})
