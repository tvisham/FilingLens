"""10-K HTML -> ordered text blocks that keep bold/italic information.

EDGAR filings are mostly inline XBRL: text sits in `<div>`/`<td>` blocks of styled
`<span>`s. Risk-factor headings are recognised by styling, so each block keeps its text
runs with bold/italic flags. Hidden content (the `ix:header`, `display:none`) is dropped,
and repeated page headers/footers are removed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from lxml import html as lxml_html
from lxml.etree import _Comment, _Element, _ProcessingInstruction

BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr",
        "section", "article", "blockquote", "dt", "dd", "pre", "center", "ul", "ol", "body",
    }
)  # fmt: skip
SKIP_TAGS = frozenset({"script", "style", "head", "title", "ix:header"})
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(bold|bolder|[6-9]00)", re.I)
_NOT_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(normal|lighter|[1-5]00)", re.I)
_ITALIC_STYLE = re.compile(r"font-style\s*:\s*(italic|oblique)", re.I)
_NOT_ITALIC_STYLE = re.compile(r"font-style\s*:\s*normal", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Run:
    text: str
    bold: bool
    italic: bool


@dataclass(frozen=True)
class Block:
    text: str  # whitespace-normalised
    runs: tuple[Run, ...]

    @property
    def styled_ratio(self) -> float:
        """Share of non-space characters that are bold or italic."""
        total = sum(len(r.text.replace(" ", "")) for r in self.runs)
        styled = sum(len(r.text.replace(" ", "")) for r in self.runs if r.bold or r.italic)
        return styled / total if total else 0.0

    @property
    def leading_styled_text(self) -> str:
        """Text of the bold/italic runs at the start of the block (a run-in heading)."""
        out: list[str] = []
        for r in self.runs:
            if r.bold or r.italic:
                out.append(r.text)
            elif r.text.strip():
                break
            else:
                out.append(r.text)
        return normalise("".join(out))


def dominant_style(runs: tuple[Run, ...] | list[Run]) -> tuple[bool, bool]:
    """(bold, italic) of the styled text in `runs`, by majority of characters."""
    styled = [r for r in runs if r.text.strip() and (r.bold or r.italic)]
    total = sum(len(r.text) for r in styled) or 1
    bold = sum(len(r.text) for r in styled if r.bold) / total >= 0.5
    italic = sum(len(r.text) for r in styled if r.italic) / total >= 0.5
    return bold, italic


def is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def normalise(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ").replace("​", "")).strip()


def _flags(el: _Element, bold: bool, italic: bool) -> tuple[bool, bool]:
    tag = str(el.tag).lower()
    if tag in ("b", "strong"):
        bold = True
    if tag in ("i", "em"):
        italic = True
    style = el.get("style") or ""
    if _BOLD_STYLE.search(style):
        bold = True
    elif _NOT_BOLD_STYLE.search(style):
        bold = False
    if _ITALIC_STYLE.search(style):
        italic = True
    elif _NOT_ITALIC_STYLE.search(style):
        italic = False
    return bold, italic


def _hidden(el: _Element) -> bool:
    return str(el.tag).lower() in SKIP_TAGS or bool(_HIDDEN_STYLE.search(el.get("style") or ""))


class _Walker:
    """Depth-first walk that turns inline runs into blocks at every block boundary.

    Handles mixed content (text next to a nested table inside one `<div>`), which a
    leaf-block approach would drop.
    """

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._runs: list[Run] = []

    def flush(self) -> None:
        text = normalise("".join(r.text for r in self._runs))
        if text:
            self.blocks.append(Block(text=text, runs=tuple(self._runs)))
        self._runs = []

    def walk(self, el: _Element, bold: bool, italic: bool) -> None:
        b, i = _flags(el, bold, italic)
        block = str(el.tag).lower() in BLOCK_TAGS
        if block:
            self.flush()
        if el.text:
            self._runs.append(Run(el.text, b, i))
        for child in el:
            if isinstance(child, (_Comment, _ProcessingInstruction)) or _hidden(child):
                pass
            elif str(child.tag).lower() == "br":
                self._runs.append(Run(" ", b, i))
            else:
                self.walk(child, b, i)
            if child.tail:
                self._runs.append(Run(child.tail, b, i))  # tail text belongs to `el`, not `child`
        if block:
            self.flush()


def html_to_blocks(document: str) -> list[Block]:
    """Text blocks in document order, each with its styled runs."""
    root = lxml_html.document_fromstring(document.encode("utf-8", errors="replace"))
    walker = _Walker()
    walker.walk(root, False, False)
    walker.flush()
    return _drop_page_furniture(walker.blocks)


_EDGE_NUMBER = re.compile(r"^\d+\b\s*|\s*\b\d+$")
_ITEM_OR_PART = re.compile(r"^(item|part)\s", re.I)


def _drop_page_furniture(blocks: list[Block]) -> list[Block]:
    """Remove page numbers and short lines repeated across pages (running headers/footers).

    Only leading/trailing numbers are ignored when spotting repeats (page numbers sit at the
    edges), so "Item 1A." and "Item 7A." stay distinct; Item/Part headings are never dropped.
    """
    keys = [_EDGE_NUMBER.sub("#", b.text.lower()) for b in blocks]
    counts = Counter(k for k, b in zip(keys, blocks, strict=True) if len(b.text) <= 80)
    repeated = {k for k, n in counts.items() if n >= 4}
    return [
        b
        for b, k in zip(blocks, keys, strict=True)
        if _ITEM_OR_PART.match(b.text)
        or (
            k not in repeated
            and not re.fullmatch(r"[#\s\-\u2013\u2014|]*", k)
            and k not in {"table of contents", "index"}
        )
    ]
