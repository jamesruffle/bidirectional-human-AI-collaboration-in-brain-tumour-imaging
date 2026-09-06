#!/usr/bin/env python3
"""Supplementary Figure 2: Human-AI collaboration paradigms.

Unlike the other figures this one is not a data plot. It is a static schematic
that was authored as a React component; the published image was a screenshot of
that component rendered in a browser. Nothing about it is computed, so this
script reproduces the same markup as a self-contained HTML page and renders it
with headless Chrome.

Two deliberate choices make the reproduction offline and deterministic:
  * the component's Tailwind utility classes are written out as explicit CSS
    rather than pulled from the Tailwind CDN at run time, and
  * the six lucide icons are inlined as SVG path data.
so no network access is needed and the output cannot drift with an upstream
release. The original component is deposited alongside this script as
`human_ai_paradigm.jsx`; it is kept for reference and is not executed.

Rendered at a 724 CSS px viewport, which is below Tailwind's `md` breakpoint and
therefore produces the single-column stacked layout of the published figure.

Requires: google-chrome (or chromium) on PATH.
Output: data/figures/Supplementary_Figure_2.png  (and .svg, .pdf)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'figures')
COMPONENT = os.path.join(HERE, 'human_ai_paradigm.jsx')   # reference only, not executed

VIEWPORT_W = 724          # CSS px; below the md: breakpoint, so panels stack
PAGE_H_IN = 14.2188       # content height, pinned so the PDF is a single page
SCALE = 7                 # device pixel ratio -> 5068 px wide, ~698 dpi at A4 text width

# Tailwind v3 palette entries used by the component.
C = {
    'white': '#ffffff',
    'slate-500': '#64748b', 'slate-600': '#475569', 'slate-700': '#334155', 'slate-800': '#1e293b',
    'blue-50': '#eff6ff', 'blue-100': '#dbeafe', 'blue-200': '#bfdbfe',
    'blue-600': '#2563eb', 'blue-700': '#1d4ed8', 'blue-900': '#1e3a8a',
    'purple-50': '#faf5ff', 'purple-100': '#f3e8ff', 'purple-200': '#e9d5ff',
    'purple-600': '#9333ea', 'purple-700': '#7e22ce', 'purple-900': '#581c87',
    'green-50': '#f0fdf4', 'green-300': '#86efac',
    'green-600': '#16a34a', 'green-700': '#15803d', 'green-800': '#166534',
    'amber-50': '#fffbeb', 'amber-300': '#fcd34d',
    'amber-600': '#d97706', 'amber-700': '#b45309', 'amber-800': '#92400e',
}

ICONS = {
    'car': '<path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/>',
    'plane': '<path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/>',
    'user': '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    'cpu': '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    'shield': '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    'eye': '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
}


def icon(name: str, px: float, colour: str) -> str:
    return (f'<svg width="{px}" height="{px}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex:none">{ICONS[name]}</svg>')


def css() -> str:
    """The Tailwind utilities the component uses, written out explicitly."""
    return f"""
*,*::before,*::after{{box-sizing:border-box;border:0 solid #e5e7eb}}
html,body{{margin:0;padding:0;background:{C['white']}}}
body{{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
 "Helvetica Neue",Arial,"Noto Sans",sans-serif;line-height:1.5;-webkit-font-smoothing:antialiased}}
p,h1,h2,ul{{margin:0}} ul{{padding:0;list-style:none}}
svg{{display:block;vertical-align:middle}}
.wrap{{width:100%;max-width:72rem;margin:0 auto;padding:2rem;background:{C['white']}}}
.title{{font-size:1.875rem;line-height:2.25rem;font-weight:700;text-align:center;
 margin-bottom:3rem;color:{C['slate-800']}}}
.grid{{display:grid;gap:2rem}}
.card{{background:{C['white']};border-radius:.75rem;padding:1.5rem;border-width:2px;
 box-shadow:0 10px 15px -3px rgb(0 0 0/.1),0 4px 6px -4px rgb(0 0 0/.1)}}
.iconwrap{{display:flex;align-items:center;justify-content:center;margin-bottom:1rem}}
.iconcirc{{padding:1rem;border-radius:9999px}}
.h2{{font-size:1.5rem;line-height:2rem;font-weight:700;text-align:center;margin-bottom:.5rem}}
.sub{{text-align:center;font-size:1.125rem;line-height:1.75rem;font-weight:600;margin-bottom:1.5rem}}
.stack>*+*{{margin-top:1.5rem}}
.flowrow{{display:flex;align-items:center;justify-content:center;gap:1rem}}
.col{{display:flex;flex-direction:column;align-items:center}}
.pill{{color:{C['white']};padding:1rem;border-radius:.5rem;display:flex;align-items:center;
 gap:.5rem;width:10rem;justify-content:center;font-weight:600;
 box-shadow:0 4px 6px -1px rgb(0 0 0/.1),0 2px 4px -2px rgb(0 0 0/.1)}}
.caption{{font-size:.75rem;line-height:1rem;color:{C['slate-600']};margin-top:.5rem;text-align:center}}
.arrow{{font-size:1.875rem;line-height:2.25rem}}
.arrowlab{{font-size:.75rem;line-height:1rem;color:{C['slate-500']};margin-top:.25rem}}
.callout{{display:flex;align-items:center;gap:.75rem;border-width:2px;border-radius:.5rem;padding:.75rem}}
.calname{{display:flex;align-items:center;gap:.5rem;font-weight:600}}
.caldesc{{font-size:.75rem;line-height:1rem}}
.notes{{border-radius:.5rem;padding:1rem;border-width:1px}}
.notes li{{display:flex;align-items:flex-start;gap:.5rem;font-size:.875rem;line-height:1.25rem;
 color:{C['slate-700']}}}
.notes li+li{{margin-top:.5rem}}
.bullet{{font-weight:700;margin-top:.25rem}}
"""


def panel(letter, title, sub, circ_icon, circ_bg, accent, border, title_col, sub_col,
          ctrl_name, ctrl_icon, ctrl_bg, ctrl_cap, task_icon, task_cap,
          s_icon, s_person, s_name, s_desc, s_bg, s_border, s_icon_col, s_text_col, s_name_col,
          notes_bg, notes_border, items):
    lis = ''.join(
        f'<li><span class="bullet" style="color:{accent}">&bull;</span>'
        f'<span><strong>{a}</strong> {b}</span></li>' for a, b in items)
    return f"""
<div class="card" style="border-color:{border}">
  <div class="iconwrap"><div class="iconcirc" style="background:{circ_bg}">{icon(circ_icon,48,accent)}</div></div>
  <h2 class="h2" style="color:{title_col}">{letter}) {title}</h2>
  <p class="sub" style="color:{sub_col}">{sub}</p>
  <div class="stack">
    <div class="flowrow">
      <div class="col"><div class="pill" style="background:{ctrl_bg}">{icon(ctrl_icon,20,C['white'])}<span>{ctrl_name}</span></div>
        <p class="caption">{ctrl_cap}</p></div>
      <div class="col"><div class="arrow" style="color:{sub_col}">&rarr;</div><div class="arrowlab">controls</div></div>
      <div class="col"><div class="pill" style="background:{C['slate-700']}">{icon(task_icon,20,C['white'])}<span>Task</span></div>
        <p class="caption">{task_cap}</p></div>
    </div>
    <div class="flowrow">
      <div class="callout" style="background:{s_bg};border-color:{s_border}">{icon(s_icon,24,s_icon_col)}
        <div><div class="calname" style="color:{s_name_col};font-size:.875rem">{icon(s_person,16,s_icon_col)}<span>{s_name}</span></div>
        <p class="caldesc" style="color:{s_text_col}">{s_desc}</p></div></div>
    </div>
    <div class="notes" style="background:{notes_bg};border-color:{notes_border}"><ul>{lis}</ul></div>
  </div>
</div>"""


def build_html() -> str:
    a = panel('a', 'AI-Assisted Human Agent', 'Car Lane Assist', 'car', C['blue-100'],
              C['blue-600'], C['blue-200'], C['blue-900'], C['blue-700'],
              'Human', 'user', C['blue-600'], 'Driver', 'car', 'Driving',
              'shield', 'cpu', 'AI Agent', 'Provides safety assistance',
              C['green-50'], C['green-300'], C['green-600'], C['green-700'], C['green-800'],
              C['blue-50'], C['blue-200'],
              [('Human is in control:', 'Driver steers, accelerates, brakes'),
               ('AI assists:', 'Monitors lane position, alerts, gentle corrections'),
               ('Human responsibility:', 'Driver accountable for all outcomes'),
               ('AI role:', 'Safety net and enhancement')])
    b = panel('b', 'Human-Assisted AI Agent', 'Plane Autopilot', 'plane', C['purple-100'],
              C['purple-600'], C['purple-200'], C['purple-900'], C['purple-700'],
              'AI Agent', 'cpu', C['purple-600'], 'Autopilot', 'plane', 'Flying',
              'eye', 'user', 'Human Agent', 'Monitors and intervenes',
              C['amber-50'], C['amber-300'], C['amber-600'], C['amber-700'], C['amber-800'],
              C['purple-50'], C['purple-200'],
              [('AI is in control:', 'Autopilot maintains altitude, heading, speed'),
               ('Human monitors:', 'Pilot oversees systems, ready to intervene'),
               ('AI responsibility:', 'System handles routine operations'),
               ('Human role:', 'Strategic oversight and edge cases')])
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css()}\n'
            f'@page{{size:{VIEWPORT_W/96:.4f}in {PAGE_H_IN}in;margin:0}}</style></head><body>'
            f'<div class="wrap"><h1 class="title">Human-AI Collaboration Paradigms</h1>'
            f'<div class="grid">{a}{b}</div></div></body></html>')


def find_chrome() -> str:
    for name in ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'):
        p = shutil.which(name)
        if p:
            return p
    sys.exit('error: needs google-chrome or chromium on PATH to render this figure')


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    chrome = find_chrome()
    print(f'Renderer: {chrome}')
    tmp = tempfile.mkdtemp(prefix='suppfig2_')
    page = os.path.join(tmp, 'supplementary_figure_2.html')
    with open(page, 'w') as fh:
        fh.write(build_html())
    print(f'Wrote self-contained page ({os.path.getsize(page)} bytes), no network needed')

    common = [chrome, '--headless', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
              '--virtual-time-budget=25000', f'--window-size={VIEWPORT_W},1400']
    png = os.path.join(OUT_DIR, 'Supplementary_Figure_2.png')
    pdf = os.path.join(OUT_DIR, 'Supplementary_Figure_2.pdf')
    subprocess.run(common + [f'--force-device-scale-factor={SCALE}',
                             f'--screenshot={png}', f'file://{page}'],
                   check=True, capture_output=True)
    subprocess.run(common + [f'--print-to-pdf={pdf}', '--no-pdf-header-footer',
                             f'file://{page}'], check=True, capture_output=True)

    # trim the screenshot to the artwork plus the page's own 2rem padding
    from PIL import Image
    import numpy as np
    im = Image.open(png).convert('RGB')
    arr = np.array(im.convert('L'))
    mask = arr < 250
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    pad = int(32 * SCALE)
    im = im.crop((max(0, cols.min() - pad), max(0, rows.min() - pad),
                  min(im.width, cols.max() + 1 + pad), min(im.height, rows.max() + 1 + pad)))
    im.save(png)
    print(f'Wrote {png}  {im.width}x{im.height} px  '
          f'({im.width / 7.26:.0f} dpi at A4 text width)')

    svg = os.path.join(OUT_DIR, 'Supplementary_Figure_2.svg')
    if shutil.which('pdftocairo'):
        # pdftocairo converts every glyph to an outline path, so the SVG is vector
        # artwork but its text is not editable. The PDF written above keeps the text
        # editable, and the supplementary figures are supplied inside the single
        # Supplementary Information PDF rather than as individual vector files.
        subprocess.run(['pdftocairo', '-svg', pdf, svg], check=True, capture_output=True)
        print(f'Wrote {svg}  (vector artwork; glyphs outlined by pdftocairo)')
    else:
        print('note: pdftocairo not found, skipping the SVG')
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
