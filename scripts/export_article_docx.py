#!/usr/bin/env python3
"""Export the LaTeX article to Word with TikZ figures rendered as images."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = ROOT / "docs" / "article_latex"
TEX = LATEX_DIR / "rune_matrix_arithmetic_article.tex"
BIB = LATEX_DIR / "rune_matrix_arithmetic_article.bib"
OUT = ROOT / "docs" / "article_interactive" / "rune_matrix_arithmetic_article.docx"
CACHE = ROOT / ".cache" / "article_docx"
FIG_DIR = CACHE / "docx_figures"
WORK_TEX = CACHE / "rune_matrix_arithmetic_article_docx.tex"


FIGURE_RE = re.compile(r"\\begin\{figure\*?\}\[[^\]]*\].*?\\end\{figure\*?\}", re.S)
TIKZ_RE = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def required(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is required")
    return path


def tikz_document(tikz: str) -> str:
    return rf"""\documentclass[tikz,border=8pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta,positioning,calc}}
\begin{{document}}
{tikz}
\end{{document}}
"""


def render_tikz_figure(index: int, tikz: str) -> Path:
    tex = CACHE / f"tikz_figure_{index:02d}.tex"
    pdf = CACHE / f"tikz_figure_{index:02d}.pdf"
    png = FIG_DIR / f"tikz_figure_{index:02d}.png"
    tex.write_text(tikz_document(tikz), encoding="utf-8")
    run(["tectonic", "--outdir", str(CACHE), str(tex)], cwd=CACHE)
    if not pdf.exists():
        raise SystemExit(f"tectonic did not produce {pdf}")
    run(["pdftoppm", "-singlefile", "-png", "-r", "220", str(pdf), str(png.with_suffix(""))])
    if not png.exists():
        raise SystemExit(f"pdftoppm did not produce {png}")
    return png


def replace_tikz_figures(source: str) -> str:
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        block = match.group(0)
        tikz_match = TIKZ_RE.search(block)
        if tikz_match is None:
            return block
        counter += 1
        png = render_tikz_figure(counter, tikz_match.group(0))
        replacement = (
            "\\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{docx_figures/{png.name}}}"
        )
        return block[: tikz_match.start()] + replacement + block[tikz_match.end() :]

    rewritten = FIGURE_RE.sub(replace, source)
    if counter == 0:
        raise SystemExit("found no TikZ figures to render")
    print(f"rendered {counter} TikZ figures")
    return rewritten


def main() -> int:
    required("tectonic")
    required("pdftoppm")
    required("pandoc")
    if not TEX.exists():
        raise SystemExit(f"missing LaTeX source: {TEX}")
    if not BIB.exists():
        raise SystemExit(f"missing bibliography: {BIB}")

    CACHE.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIG_DIR.glob("tikz_figure_*.png"):
        path.unlink()

    source = TEX.read_text(encoding="utf-8")
    rewritten = replace_tikz_figures(source)
    WORK_TEX.write_text(rewritten, encoding="utf-8")

    run(
        [
            "pandoc",
            str(WORK_TEX),
            "--from",
            "latex",
            "--to",
            "docx",
            "--bibliography",
            str(BIB),
            "--citeproc",
            "--metadata",
            "reference-section-title=References",
            "--resource-path",
            f"{CACHE}:{LATEX_DIR}:{ROOT / 'docs' / 'article_figures'}:{ROOT / 'docs' / 'article_interactive' / 'assets'}",
            "--output",
            str(OUT),
        ],
        cwd=ROOT,
    )
    if not OUT.exists():
        raise SystemExit(f"pandoc did not produce {OUT}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
