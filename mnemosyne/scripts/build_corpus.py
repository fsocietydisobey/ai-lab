#!/usr/bin/env python3
"""Assemble a continued-pretraining (CPT) corpus from raw source repos.

Walks one or more repo roots, keeps source + docs, drops noise (binaries,
lockfiles, vendored deps, VCS, build output), prepends a path header to each
file so the model learns repo structure, and splits at the FILE level into
train/eval. File-level (not line-level) split is the point: eval perplexity
then measures generalization to UNSEEN files, not memorization of seen lines.

Usage:
  python scripts/build_corpus.py --repo /home/_3ntropy/dev/khimaira \
      --out-dir corpora/khimaira --eval-frac 0.1
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

INCLUDE_EXT = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".yaml", ".yml", ".toml",
    ".json", ".sh", ".css", ".scss", ".html", ".sql", ".rs", ".go", ".jinja",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", "target", ".turbo", "coverage",
    ".ruff_cache", "models", "adapters", "corpora", ".cache", ".idea", ".vscode",
}
SKIP_FILE_SUBSTR = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "uv.lock", ".min.js", ".min.css", ".lock",
}
MAX_FILE_BYTES = 500_000


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix.lower() not in INCLUDE_EXT:
            continue
        if any(s in p.name for s in SKIP_FILE_SUBSTR):
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield p


def in_eval(relpath: str, eval_frac: float) -> bool:
    """Deterministic file-level split keyed on path hash (reproducible)."""
    h = int(hashlib.sha1(relpath.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a CPT corpus from raw source.")
    ap.add_argument("--repo", action="append", required=True, help="Repo root (repeatable)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_parts: list[str] = []
    eval_parts: list[str] = []
    n_train = n_eval = 0
    bytes_train = bytes_eval = 0

    for repo in args.repo:
        root = Path(repo).resolve()
        label = root.name
        for f in iter_files(root):
            rel = f.relative_to(root).as_posix()
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not text.strip():
                continue
            key = f"{label}/{rel}"
            block = f"# File: {key}\n{text}\n\n"
            if in_eval(key, args.eval_frac):
                eval_parts.append(block)
                n_eval += 1
                bytes_eval += len(block)
            else:
                train_parts.append(block)
                n_train += 1
                bytes_train += len(block)

    (out / "corpus_train.txt").write_text("".join(train_parts), encoding="utf-8")
    (out / "corpus_eval.txt").write_text("".join(eval_parts), encoding="utf-8")

    print(f"[corpus] repos: {', '.join(Path(r).name for r in args.repo)}")
    print(f"[corpus] train: {n_train} files, {bytes_train/1e6:.2f} MB → {out/'corpus_train.txt'}")
    print(f"[corpus] eval:  {n_eval} files, {bytes_eval/1e6:.2f} MB → {out/'corpus_eval.txt'}")


if __name__ == "__main__":
    main()
