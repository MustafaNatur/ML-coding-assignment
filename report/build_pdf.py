"""Render report.md to HTML (Mermaid diagrams included), then print it to PDF with headless Chrome.

Usage:  python build_pdf.py
Requires: `markdown` (pip), Google Chrome, and assets/mermaid.min.js (vendored, so no network).
"""
from __future__ import annotations

import base64
import html
import mimetypes
import re
import subprocess
from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

STYLESHEET = """
@page { size: A4; margin: 15mm 17mm; }

html { font-size: 9.7pt; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  line-height: 1.32;
  color: #111;
  margin: 0;
  text-align: justify;
  hyphens: auto;
}

h1 { font-size: 1.55rem; line-height: 1.2; margin: 0 0 .2rem; text-align: left; }
h2 {
  font-size: 1.14rem; margin: .85rem 0 .35rem; text-align: left;
  border-bottom: 1px solid #ccc; padding-bottom: .12rem;
  break-after: avoid; page-break-after: avoid;
}
h3 { font-size: 1rem; margin: .6rem 0 .25rem; text-align: left;
     break-after: avoid; page-break-after: avoid; }
p { margin: 0 0 .45rem; }
hr { border: none; border-top: 1px solid #ccc; margin: .7rem 0 .8rem; }

ul { margin: .3rem 0 .7rem; padding-left: 1.1rem; }
li { margin-bottom: .3rem; }

/* the references are the only ordered list; keep them compact */
ol { margin: .3rem 0 .3rem; padding-left: 1.3rem; font-size: .93rem; }
ol li { margin-bottom: .1rem; }

code {
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: .84em; background: #f4f4f4; padding: .05em .25em; border-radius: 3px;
}
pre {
  background: #f7f7f7; border: 1px solid #e2e2e2; border-radius: 4px;
  padding: .5rem .7rem; margin: .5rem 0 .8rem; overflow: hidden;
  break-inside: avoid; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: .8rem; line-height: 1.35; }

table {
  border-collapse: collapse; width: 100%; margin: .5rem 0 .9rem; font-size: .9rem;
  break-inside: avoid; page-break-inside: avoid;
}
th, td { border-bottom: 1px solid #ddd; padding: .28rem .45rem; text-align: left; }
th { border-bottom: 1.5px solid #999; font-weight: 600; }
.num { text-align: right; }

figure { margin: .5rem 0 .8rem; text-align: center;
         break-inside: avoid; page-break-inside: avoid; }
figure img { max-width: 70%; }
figcaption { font-size: .82rem; color: #444; margin-top: .3rem;
             text-align: left; line-height: 1.35; }

em { color: #000; }

.diagram {
  margin: .4rem auto .6rem; text-align: center;
  break-inside: avoid; page-break-inside: avoid;
}
.mermaid svg { max-width: 100%; height: auto; }

/* a diagram set beside the text that explains it */
.row {
  display: flex; gap: .9rem; align-items: flex-start; margin: .4rem 0 .7rem;
  break-inside: avoid; page-break-inside: avoid;
}
.row-text { flex: 1 1 auto; min-width: 0; }
.row-text > *:first-child { margin-top: 0; }
.row .diagram { margin: 0; }
.row ol, .row ul { margin-top: 0; }
"""

# ```mermaid w=60  ->  diagram rendered at 60% of the text width
MERMAID_BLOCK = re.compile(r"^```mermaid(?P<opts>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.S | re.M)


def inline_images(html: str) -> str:
    """Replace <img src="..."> with base64 data URIs so the HTML stands alone."""

    def replace(match: re.Match) -> str:
        src = match.group("src")
        path = (REPORT_DIR / src).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return match.group(0).replace(src, f"data:{mime};base64,{encoded}")

    return re.sub(r'<img[^>]*src="(?P<src>[^"]+)"[^>]*>', replace, html)


def promote_figures(html: str) -> str:
    """Turn a paragraph holding a single image into <figure> + <figcaption> from its alt text."""

    def replace(match: re.Match) -> str:
        img, alt = match.group(0), match.group("alt")
        return f"<figure>{img}<figcaption>{alt}</figcaption></figure>"

    return re.sub(
        r'<p><img[^>]*alt="(?P<alt>[^"]*)"[^>]*/?></p>',
        replace,
        html,
    )


CELL = re.compile(r"<(?P<tag>td|th)(?P<attrs>[^>]*)>(?P<text>.*?)</(?P=tag)>", re.S)
NUMBER = re.compile(r"[<>≈±~]?\s*[-+]?\d[\d.,]*(\s*(%|×|nats|e-?\d+))?\s*")


def align_numeric_columns(html: str) -> str:
    """Right-align table columns whose body cells are all numbers; leave prose columns alone."""

    def bare(cell: str) -> str:
        return re.sub(r"<[^>]+>", "", cell).replace("&nbsp;", " ").strip()

    def fix_table(match: re.Match) -> str:
        table = match.group(0)
        rows = re.findall(r"<tr>(.*?)</tr>", table, re.S)
        body = [[bare(c.group("text")) for c in CELL.finditer(r)] for r in rows[1:]]
        if not body:
            return table
        columns = range(min(len(r) for r in body))
        numeric = {
            i for i in columns
            if any(row[i] for row in body)
            and all(not row[i] or NUMBER.fullmatch(row[i]) for row in body)
        }

        index = -1

        def fix_cell(cell: re.Match) -> str:
            nonlocal index
            index += 1
            return cell.group(0) if index not in numeric else (
                f'<{cell.group("tag")} class="num">{cell.group("text")}</{cell.group("tag")}>'
            )

        out, position = [], 0
        for row in re.finditer(r"<tr>(.*?)</tr>", table, re.S):
            index = -1
            out.append(table[position:row.start()])
            out.append(CELL.sub(fix_cell, row.group(0)))
            position = row.end()
        out.append(table[position:])
        return "".join(out)

    return re.sub(r"<table>.*?</table>", fix_table, html, flags=re.S)


def stash_mermaid(source: str) -> tuple[str, list[str]]:
    """Pull ```mermaid blocks out before conversion, so Markdown does not escape their syntax."""
    diagrams: list[str] = []

    def replace(match: re.Match) -> str:
        opts = match.group("opts")
        side = re.search(r"side=(\d+)", opts)
        width = re.search(r"w=(\d+)", opts)
        percent = (side or width).group(1) if (side or width) else "100"
        # Escape the source: the browser parses the div's content as HTML first, and
        # would otherwise swallow mermaid syntax such as <<base>> as an unknown tag.
        diagrams.append(
            f'<div class="diagram" style="max-width:{percent if not side else 100}%">'
            f'<div class="mermaid">{html.escape(match.group("body"))}</div></div>'
        )
        sides.append(int(side.group(1)) if side else 0)
        return f"\n@@MERMAID{len(diagrams) - 1}@@\n"

    sides: list[int] = []
    return MERMAID_BLOCK.sub(replace, source), diagrams, sides


def place_beside_text(body: str, index: int, diagram: str, width: int) -> str:
    """Lay a diagram out next to the prose that follows it, up to the next heading."""
    marker = f"<p>@@MERMAID{index}@@</p>"
    start = body.index(marker)
    rest = body[start + len(marker):]
    stop = re.search(r"<h[1-6]|@@MERMAID|<table", rest)
    text, tail = (rest[:stop.start()], rest[stop.start():]) if stop else (rest, "")
    return (
        f'{body[:start]}<div class="row">'
        f'<div style="flex:0 0 {width}%">{diagram}</div>'
        f'<div class="row-text">{text}</div></div>{tail}'
    )


def main() -> None:
    source = (REPORT_DIR / "report.md").read_text()
    source, diagrams, sides = stash_mermaid(source)

    body = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    body = promote_figures(inline_images(align_numeric_columns(body)))
    for index, (diagram, side) in enumerate(zip(diagrams, sides)):
        if side:
            body = place_beside_text(body, index, diagram, side)
        else:
            body = body.replace(f"<p>@@MERMAID{index}@@</p>", diagram)

    mermaid_js = (REPORT_DIR / "assets" / "mermaid.min.js").read_text()
    html_path = REPORT_DIR / "report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Character-Level GPT from Scratch</title><style>{STYLESHEET}</style>"
        f"</head><body>{body}"
        f"<script>{mermaid_js}</script>"
        "<script>mermaid.initialize({startOnLoad:true, theme:'neutral', "
        "themeVariables:{fontSize:'13px', fontFamily:'Helvetica, Arial, sans-serif'}, "
        "flowchart:{useMaxWidth:true, htmlLabels:true, nodeSpacing:28, rankSpacing:26, padding:6}, "
        "sequence:{useMaxWidth:true, mirrorActors:false, boxMargin:5, actorMargin:40, "
        "height:34, messageMargin:26, bottomMarginAdj:0}, "
        "class:{useMaxWidth:true, nodeSpacing:22, rankSpacing:34}});</script>"
        "</body></html>"
    )

    pdf_path = REPORT_DIR / "report.pdf"
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=30000", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )

    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"{pdf_path.name}: {pages} pages, {pdf_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
