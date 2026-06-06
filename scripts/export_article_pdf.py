#!/usr/bin/env python3
"""Compile the LaTeX article PDF and publish it beside the interactive article."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = ROOT / "docs" / "article_latex"
TEX = LATEX_DIR / "rune_matrix_arithmetic_article.tex"
LATEX_PDF = LATEX_DIR / "rune_matrix_arithmetic_article.pdf"
OUT = ROOT / "docs" / "article_interactive" / "rune_matrix_arithmetic_article.pdf"


def main() -> int:
    if not TEX.exists():
        raise SystemExit(f"missing LaTeX source: {TEX}")
    if shutil.which("tectonic") is None:
        raise SystemExit("tectonic is required to build the article PDF")

    subprocess.run(
        ["tectonic", "--outdir", str(LATEX_DIR), str(TEX)],
        cwd=LATEX_DIR,
        check=True,
    )
    if not LATEX_PDF.exists():
        raise SystemExit(f"tectonic did not produce {LATEX_PDF}")
    shutil.copy2(LATEX_PDF, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
