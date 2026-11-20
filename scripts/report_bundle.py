"""
report_bundle: the interactive tables and the report zip (stdlib only),
adapted from NanoPulldown's nanopulldown.report.bundle.

  tables  datavzrd over a config whose relative table paths resolve against
          the bundle dir's parent, then the dark skin (report/datavzrd/
          datavzrd-dark.css) inlined into every page: datavzrd has no dark
          mode, and once the bundle sits inside the Snakemake report zip the
          parent stylesheet can't reach it.
  build   `snakemake --report <zip> --report-stylesheet report/custom.css`,
          then the dark-theme fixes the Vega charts can't take from a
          stylesheet. Called from the Snakefile's onsuccess: never fails.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARK_OPEN, MARK_CLOSE = "<style data-ab-dark>", "</style>"
MARK = "/* cfrandom:style_report */"
INK, INK2, RULE = "#e8eef5", "#93a4b6", "#2c3e52"
VEGA_CONFIG = {"background": None,
               "axis": {"labelColor": INK2, "titleColor": INK, "domainColor": INK2, "tickColor": INK2, "gridColor": RULE},
               "legend": {"labelColor": INK, "titleColor": INK}, "title": {"color": INK},
               "view": {"stroke": RULE}, "text": {"fill": INK}}


def style_datavzrd(root: Path, css: Path) -> int:
    block = f"{MARK_OPEN}\n{css.read_text()}\n{MARK_CLOSE}"
    pages = sorted(root.rglob("*.html"))
    for page in pages:
        html = page.read_text(encoding="utf-8")
        if MARK_OPEN in html:
            pre, _, rest = html.partition(MARK_OPEN)
            html = pre + rest.partition(MARK_CLOSE)[2]
        low = html.lower()
        meta = low.find("<meta")
        idx = html.find(">", meta) + 1 if meta != -1 else 0
        page.write_text(html[:idx] + block + html[idx:], encoding="utf-8")
    print(f"[datavzrd] skinned {len(pages)} page(s) under {root}")
    return len(pages)


def patch_html(html: str) -> str:
    if MARK in html:
        return html
    html = re.sub(r'(rulegraph_spec = \{.*?)("fill": \{"value": )"black"',
                  lambda m: m.group(1) + m.group(2) + f'"{INK}"', html, count=1, flags=re.S)
    html = re.sub(r'(rulegraph_spec = \{.*?\};)',
                  lambda m: m.group(1).replace('"stroke": {"value": "#ccc"}', f'"stroke": {{"value": "{INK2}"}}'),
                  html, count=1, flags=re.S)
    cfg = json.dumps(VEGA_CONFIG)
    for name in ("rulegraph_spec", "runtimes_spec", "timeline_spec"):
        html = html.replace(f"var {name} = {{", f'var {name} = {{"config": {cfg}, ', 1)
    style = f"<style>{MARK} :root {{ --fig-card: #ffffff; }}</style>"
    return html.replace("</head>", style + "</head>", 1) if "</head>" in html else style + html


def style_report(zpath: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / zpath.name
        with zipfile.ZipFile(zpath) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith("report.html"):
                    data = patch_html(data.decode("utf-8")).encode("utf-8")
                zout.writestr(item, data)
        shutil.move(out, zpath)


def tables(a) -> int:
    a.outdir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["datavzrd", str(a.config.resolve()), "--overwrite-output", "--output", a.outdir.name],
                   cwd=a.outdir.parent, check=True)
    style_datavzrd(a.outdir, a.css)
    return 0


def build(a) -> int:
    out = a.out if a.out.is_absolute() else REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["snakemake", "-s", str(REPO / "Snakefile"), "--report", str(out),
           "--report-stylesheet", str(REPO / "report" / "custom.css"), *a.extra]
    print("[report]", " ".join(cmd), flush=True)
    if shutil.which("snakemake") is None or subprocess.run(cmd, cwd=REPO).returncode != 0:
        print("[report] WARNING: report build failed (non-fatal)", file=sys.stderr)
        return 0
    try:
        style_report(out)
    except Exception as e:  # noqa: BLE001
        print(f"[report] WARNING: dark-theme patch failed: {e}", file=sys.stderr)
    print(f"[report] ok: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--outdir", type=Path, required=True)
    t.add_argument("--css", type=Path, default=REPO / "report" / "datavzrd" / "datavzrd-dark.css")
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path("reports/report.zip"))
    b.add_argument("extra", nargs="*", help="extra snakemake args (e.g. --configfile ...)")
    a = ap.parse_args()
    return tables(a) if a.cmd == "tables" else build(a)


if __name__ == "__main__":
    raise SystemExit(main())
