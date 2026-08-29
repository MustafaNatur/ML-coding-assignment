"""Render report.md to HTML with Mermaid diagrams as PNG figures, then print to PDF.

Usage:  python build_pdf.py
Requires: markdown, Pillow, Google Chrome, and assets/mermaid.min.js.
"""
from __future__ import annotations

import base64
import html
import mimetypes
import re
import subprocess
import tempfile
from pathlib import Path

import markdown
from PIL import Image

REPORT_DIR = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIGURES_DIR = REPORT_DIR / "figures"

STYLESHEET = """
@page { size: A4; margin: 14mm 16mm; }

html { font-size: 9.6pt; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  line-height: 1.30;
  color: #111;
  margin: 0;
  text-align: justify;
  hyphens: auto;
}

h1 { font-size: 1.5rem; line-height: 1.2; margin: 0 0 .18rem; text-align: left; }
h2 {
  font-size: 1.12rem; margin: .7rem 0 .3rem; text-align: left;
  break-after: avoid; page-break-after: avoid;
}
h3 { font-size: .98rem; margin: .5rem 0 .2rem; text-align: left;
     break-after: avoid; page-break-after: avoid; }
p { margin: 0 0 .38rem; }
hr { display: none; }

ul { margin: .25rem 0 .55rem; padding-left: 1.1rem; }
li { margin-bottom: .22rem; }

ol { margin: .25rem 0 .25rem; padding-left: 1.3rem; font-size: .93rem; }
ol li { margin-bottom: .08rem; }

code {
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: .84em; background: #f4f4f4; padding: .05em .25em; border-radius: 3px;
}
pre {
  background: #f7f7f7; border: 1px solid #e2e2e2; border-radius: 4px;
  padding: .4rem .6rem; margin: .4rem 0 .6rem; overflow: hidden;
  break-inside: avoid; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: .78rem; line-height: 1.3; }

table {
  border-collapse: collapse; width: 100%; margin: .4rem 0 .7rem; font-size: .88rem;
  break-inside: avoid; page-break-inside: avoid;
}
th, td { border-bottom: 1px solid #ddd; padding: .24rem .4rem; text-align: left; }
th { border-bottom: 1.5px solid #999; font-weight: 600; }
.num { text-align: right; }

figure { margin: .4rem 0 .65rem; text-align: center;
         break-inside: avoid; page-break-inside: avoid; }
figure img { max-width: 72%; height: auto; }
figcaption { font-size: .8rem; color: #444; margin-top: .25rem;
             text-align: left; line-height: 1.3; }

em { color: #000; }

.diagram {
  margin: .35rem auto .5rem; text-align: center;
  break-inside: avoid; page-break-inside: avoid;
}
.diagram img {
  max-width: 100%; height: auto; display: block; margin: 0 auto;
  background: #fff;
}

.row {
  display: flex; gap: .8rem; align-items: flex-start; margin: .35rem 0 .55rem;
  break-inside: avoid; page-break-inside: avoid;
}
.row-text { flex: 1 1 auto; min-width: 0; }
.row-text > *:first-child { margin-top: 0; }
.row .diagram { margin: 0; }
.row ol, .row ul { margin-top: 0; }
"""

MERMAID_BLOCK = re.compile(r"^```mermaid(?P<opts>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.S | re.M)
CELL = re.compile(r"<(?P<tag>td|th)(?P<attrs>[^>]*)>(?P<text>.*?)</(?P=tag)>", re.S)
NUMBER = re.compile(r"[<>≈±~]?\s*[-+]?\d[\d.,]*(\s*(%|×|nats|e-?\d+))?\s*")


def inline_images(markup: str) -> str:
    def replace(match: re.Match) -> str:
        src = match.group("src")
        if src.startswith("data:"):
            return match.group(0)
        path = (REPORT_DIR / src).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return match.group(0).replace(src, f"data:{mime};base64,{encoded}")

    return re.sub(r'<img[^>]*src="(?P<src>[^"]+)"[^>]*>', replace, markup)


def promote_figures(markup: str) -> str:
    def replace(match: re.Match) -> str:
        img, alt = match.group(0), match.group("alt")
        return f"<figure>{img}<figcaption>{alt}</figcaption></figure>"

    return re.sub(
        r'<p><img[^>]*alt="(?P<alt>[^"]*)"[^>]*/?></p>',
        replace,
        markup,
    )


def align_numeric_columns(markup: str) -> str:
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

        def fix_cell(cell: re.Match, index: list[int]) -> str:
            index[0] += 1
            return cell.group(0) if index[0] not in numeric else (
                f'<{cell.group("tag")} class="num">{cell.group("text")}</{cell.group("tag")}>'
            )

        out, position = [], 0
        for row in re.finditer(r"<tr>(.*?)</tr>", table, re.S):
            index = [-1]
            out.append(table[position:row.start()])
            out.append(CELL.sub(lambda m: fix_cell(m, index), row.group(0)))
            position = row.end()
        out.append(table[position:])
        return "".join(out)

    return re.sub(r"<table>.*?</table>", fix_table, markup, flags=re.S)


def crop_whitespace(image_path: Path, pad: int = 12) -> None:
    image = Image.open(image_path).convert("RGB")
    pixels = image.load()
    width, height = image.size
    left, top, right, bottom = width, height, 0, 0
    threshold = 248
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r < threshold or g < threshold or b < threshold:
                left, top = min(left, x), min(top, y)
                right, bottom = max(right, x), max(bottom, y)
    if right <= left or bottom <= top:
        return
    box = (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad + 1),
        min(height, bottom + pad + 1),
    )
    image.crop(box).save(image_path, "PNG")


def render_mermaid_png(source: str, destination: Path) -> None:
    mermaid_js = (REPORT_DIR / "assets" / "mermaid.min.js").read_text()
    page = Path(tempfile.mkstemp(suffix=".html")[1])
    page.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>html,body{margin:0;padding:24px;background:#fff}"
        ".mermaid svg{max-width:none}</style>"
        f'<div class="mermaid">{html.escape(source)}</div>'
        f"<script>{mermaid_js}</script>"
        "<script>mermaid.initialize({startOnLoad:true, theme:'neutral',"
        "themeVariables:{fontSize:'15px', fontFamily:'Helvetica, Arial, sans-serif'},"
        "flowchart:{useMaxWidth:false, htmlLabels:true, nodeSpacing:32, rankSpacing:28},"
        "class:{useMaxWidth:false, nodeSpacing:24, rankSpacing:36}});</script>"
    )
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
         "--default-background-color=ffffff", "--window-size=1600,2200",
         "--virtual-time-budget=20000", f"--screenshot={destination}",
         page.as_uri()],
        check=True, capture_output=True,
    )
    page.unlink(missing_ok=True)
    crop_whitespace(destination)


def stash_mermaid(source: str) -> tuple[str, list[str], list[int]]:
    diagrams: list[str] = []
    sides: list[int] = []
    FIGURES_DIR.mkdir(exist_ok=True)

    def replace(match: re.Match) -> str:
        opts = match.group("opts")
        side = re.search(r"side=(\d+)", opts)
        width = re.search(r"w=(\d+)", opts)
        percent = int((side or width).group(1)) if (side or width) else 100
        index = len(diagrams)
        png_path = FIGURES_DIR / f"diagram_{index + 1}.png"
        render_mermaid_png(match.group("body"), png_path)
        style = f'style="max-width:{percent}%"' if not side else ""
        diagrams.append(
            f'<div class="diagram" {style}>'
            f'<img src="{png_path.relative_to(REPORT_DIR).as_posix()}" alt="Diagram {index + 1}">'
            f"</div>"
        )
        sides.append(int(side.group(1)) if side else 0)
        return f"\n@@MERMAID{index}@@\n"

    return MERMAID_BLOCK.sub(replace, source), diagrams, sides


def place_beside_text(body: str, index: int, diagram: str, width: int) -> str:
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
    body = inline_images(body)

    html_path = REPORT_DIR / "report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Character-Level GPT from Scratch</title><style>{STYLESHEET}</style>"
        f"</head><body>{body}</body></html>"
    )

    pdf_path = REPORT_DIR / "report.pdf"
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )

    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"{pdf_path.name}: {pages} pages, {pdf_path.stat().st_size / 1024:.0f} KB")
    print("diagrams:", ", ".join(p.name for p in sorted(FIGURES_DIR.glob("diagram_*.png"))))


if __name__ == "__main__":
    main()
