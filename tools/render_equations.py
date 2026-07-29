"""Render display LaTeX equations in tutorial Markdown as static SVG assets.

This intentionally removes the web viewer's MathJax layout dependency: each
display equation is stored with intrinsic SVG dimensions and embedded directly.
Run from the repository root with the system Python that provides matplotlib.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import MathTextParser


ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    (ROOT / "docs" / "01-vae-derivation.md", "vae"),
    (ROOT / "docs" / "02-variants.md", "variants"),
]
ASSET_ROOT = ROOT / "assets" / "equations"
BLOCK_RE = re.compile(r"\$\$\s*\n(.*?)\n\s*\$\$", re.DOTALL)
TAG_RE = re.compile(r"\\tag\{([^{}]+)\}")

TEXT_REPLACEMENTS = {
    "期望重构对数似然": "reconstruction",
    "码本损失": "codebook",
    "承诺损失": "commitment",
    "重构": "reconstruction",
}


def strip_annotated_macro(text: str, macro: str) -> str:
    r"""Replace ``\macro{expression}_{label}`` with its expression."""
    token = rf"\{macro}{{"
    while token in text:
        start = text.index(token)
        content_start = start + len(token)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        content = text[content_start : cursor - 1]
        end = cursor
        if text[end : end + 2] == "_{":
            depth = 1
            end += 2
            while end < len(text) and depth:
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                end += 1
        text = text[:start] + content + text[end:]
    return text


def normalize_formula(source: str) -> tuple[list[str], str | None]:
    """Return mathtext-compatible visual lines and the equation tag."""
    tag_match = TAG_RE.search(source)
    tag = tag_match.group(1) if tag_match else None
    text = TAG_RE.sub("", source)
    text = text.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")

    for chinese, english in TEXT_REPLACEMENTS.items():
        text = text.replace(rf"\text{{{chinese}}}", rf"\mathrm{{{english}}}")
    text = strip_annotated_macro(text, "underbrace")
    text = strip_annotated_macro(text, "boxed")

    # Matplotlib mathtext does not implement every AMS macro. These equivalent
    # forms preserve the visible mathematical meaning.
    text = text.replace(r"\operatorname{diag}", r"\mathrm{diag}")
    text = text.replace(r"\operatorname{Var}", r"\mathrm{Var}")
    text = text.replace(r"\operatorname{sigmoid}", r"\mathrm{sigmoid}")
    text = text.replace(r"\operatorname{logsumexp}", r"\mathrm{logsumexp}")
    text = text.replace(r"\operatorname{sg}", r"\mathrm{sg}")
    text = text.replace(r"\text{s.t.}", r"\mathrm{s.t.}")
    text = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", text)
    text = re.sub(r"\\mathbb\s+([A-Za-z])", r"\\mathbb{\1}", text)
    text = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", text)
    text = re.sub(r"\\le(?![A-Za-z])", r"\\leq", text)
    text = text.replace(r"\frac12", r"\frac{1}{2}")
    text = text.replace(r"\frac1{", r"\frac{1}{")
    text = re.sub(r"\\frac([A-Za-z0-9])([A-Za-z0-9])", r"\\frac{\1}{\2}", text)
    text = text.replace(r"^\*", "^*")

    sentinel = "@@ROWBREAK@@"
    text = re.sub(r"\\\\\s*", sentinel, text)
    rows = []
    for row in text.split(sentinel):
        row = " ".join(part.strip() for part in row.splitlines() if part.strip())
        row = row.replace("&", "").strip().rstrip(",")
        if row:
            rows.append(row)
    return rows or [""], tag


def validate(rows: list[str], label: str) -> None:
    parser = MathTextParser("path")
    for index, row in enumerate(rows, start=1):
        try:
            parser.parse(f"${row}$", dpi=160)
        except Exception as exc:  # pragma: no cover - diagnostic path
            raise ValueError(f"{label}, row {index}: {row}\n{exc}") from exc


def render_svg(rows: list[str], tag: str | None, output: Path) -> None:
    longest = max(len(row) for row in rows)
    width = min(13.0, max(4.6, 0.068 * longest + (1.0 if tag else 0.0)))
    height = max(0.72, 0.60 * len(rows) + 0.16)
    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.subplots_adjust(0, 0, 1, 1)

    top = 0.72 if len(rows) == 1 else 0.86
    step = 0 if len(rows) == 1 else 0.72 / max(1, len(rows) - 1)
    for i, row in enumerate(rows):
        y = top - i * step
        fig.text(0.055, y, f"${row}$", fontsize=15.5, ha="left", va="center", color="#111111")
    if tag:
        fig.text(0.965, 0.5, f"({tag})", fontsize=13.5, ha="right", va="center", color="#111111")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.10,
        transparent=False,
        metadata={"Date": None, "Creator": "vae-family-theory-and-code"},
    )
    plt.close(fig)
    svg = output.read_text(encoding="utf-8")
    output.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    manifest: dict[str, dict[str, object]] = {}
    total = 0
    for doc_path, group in DOCS:
        markdown = doc_path.read_text(encoding="utf-8")
        counter = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal counter, total
            counter += 1
            total += 1
            source = match.group(1).strip()
            rows, tag = normalize_formula(source)
            label = tag or f"unnumbered-{counter:02d}"
            validate(rows, f"{doc_path.name} equation {label}")
            filename = f"eq-{counter:02d}.svg"
            output = ASSET_ROOT / group / filename
            render_svg(rows, tag, output)
            key = f"{group}/{filename}"
            manifest[key] = {"tag": tag, "latex": source, "rows": rows}
            alt = f"公式（{tag}）" if tag else f"{group} 推导公式 {counter}"
            return (
                f'<p align="center">\n'
                f'  <img src="../assets/equations/{group}/{filename}" alt="{alt}" />\n'
                f'</p>'
            )

        converted, replacements = BLOCK_RE.subn(replace, markdown)
        if replacements == 0:
            raise RuntimeError(f"No display equations found in {doc_path}")
        doc_path.write_text(converted, encoding="utf-8", newline="\n")

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    (ASSET_ROOT / "formulas.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Rendered {total} display equations to {ASSET_ROOT}")


if __name__ == "__main__":
    main()
