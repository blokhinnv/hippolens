#!/usr/bin/env python3
"""Index the demo corpus into a HippoRAG save_dir (one-time API setup)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_CORPUS = Path(__file__).resolve().parent / "demo_corpus.json"
_DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "demo_index"


def load_corpus(corpus_path: Path) -> list[str]:
    with corpus_path.open(encoding="utf-8") as f:
        records = json.load(f)
    return [record["text"] for record in records]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from examples.demo_config import load_demo_env, make_demo_config
    from hipporag import HippoRAG

    parser = argparse.ArgumentParser(description="Index demo corpus with HippoRAG + OpenRouter.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help=f"JSON corpus file (default: {_DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=_DEFAULT_SAVE_DIR,
        help=f"HippoRAG save_dir (default: {_DEFAULT_SAVE_DIR})",
    )
    args = parser.parse_args()

    load_demo_env()
    docs = load_corpus(args.corpus)
    save_dir = str(args.save_dir.resolve())

    print(f"Indexing {len(docs)} passages into {save_dir} …")
    hr = HippoRAG(global_config=make_demo_config(save_dir))
    hr.index(docs)
    print(f"Done. Graph: {hr._graph_pickle_filename}")
    print(f"Working dir: {hr.working_dir}")


if __name__ == "__main__":
    main()
