#!/usr/bin/env python3
"""Render MAGNET incremental TRACE logs to a self-contained HTML viewer.

Usage:
  python runnable_code/render_magnet_incremental_trace.py \\
    [--log outputs/magnet_incremental_trace_phase_c.log] \\
    [--out outputs/magnet_incremental_trace_phase_c.html]

TEMPORARY companion to MAGNET_INC_TRACE logging.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

LINE_RE = re.compile(r"^\[magnet-inc-trace\] #(\d+) (\S+)(?: \| (.*))?$")
FEAT_PART_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+):shape=(?P<shape>\([^)]+\))"
    r":min=(?P<min>[^:]+):max=(?P<max>[^:]+):mean=(?P<mean>\S+)"
)


def _parse_scalar(raw: str):
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw in ("True", "False"):
        return raw == "True"
    if raw == "None":
        return None
    try:
        if any(c in raw for c in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_fields(rest: str) -> dict:
    fields: dict = {}
    i = 0
    n = len(rest)
    while i < n:
        while i < n and rest[i].isspace():
            i += 1
        if i >= n:
            break
        eq = rest.find("=", i)
        if eq < 0:
            break
        key = rest[i:eq]
        j = eq + 1
        if j < n and rest[j] in "'\"":
            quote = rest[j]
            j += 1
            start = j
            while j < n and rest[j] != quote:
                if rest[j] == "\\" and j + 1 < n:
                    j += 2
                else:
                    j += 1
            val = rest[start:j]
            j += 1  # closing quote
            fields[key] = val
        else:
            start = j
            while j < n and not rest[j].isspace():
                j += 1
            fields[key] = _parse_scalar(rest[start:j])
        i = j

    feats = fields.get("features")
    if isinstance(feats, str):
        for match in FEAT_PART_RE.finditer(feats):
            name = match.group("name")
            fields[f"{name}_shape"] = match.group("shape")
            fields[f"{name}_min"] = float(match.group("min"))
            fields[f"{name}_max"] = float(match.group("max"))
            fields[f"{name}_mean"] = float(match.group("mean"))
    return fields


def parse_log(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        event = m.group(2)
        fields = _parse_fields(m.group(3) or "")
        rows.append({"step": step, "event": event, **fields})
    return rows


def build_cycles(rows: list[dict]) -> list[dict]:
    predicts = [r for r in rows if r["event"] == "MagnetSession.predict_step"]
    feat_at = [r for r in rows if r["event"] == "FeatureState.features_at"]
    seis_q = [r for r in rows if r["event"] == "SeisHash.features_at_time"]
    recent_q = [r for r in rows if r["event"] == "RecentSliding.features_at_time"]
    ingests = [r for r in rows if r["event"] == "FeatureState.ingest"]

    cycles = []
    for i, p in enumerate(predicts):
        t = p.get("t")
        fa = next(
            (r for r in reversed(feat_at) if r["step"] < p["step"] and r.get("t") == t),
            {},
        )
        sq = next(
            (r for r in reversed(seis_q) if r["step"] < p["step"] and r.get("t") == t),
            {},
        )
        rq = next(
            (
                r
                for r in reversed(recent_q)
                if r["step"] < p["step"] and r.get("t") == t
            ),
            {},
        )
        ing = next(
            (r for r in ingests if r["step"] > p["step"] and r.get("t") == t),
            {},
        )
        cycles.append(
            {
                "i": i + 1,
                "t": t,
                "lng": p.get("lng"),
                "lat": p.get("lat"),
                "sampled_mag": p.get("sampled_mag"),
                "n_catalog": ing.get("n_catalog") or fa.get("n_catalog"),
                "n_candidates": sq.get("n_candidates"),
                "n_cells": sq.get("n_cells_scanned"),
                "n_real": rq.get("n_real"),
                "n_history": rq.get("n_history"),
                "recent_mean": rq.get("recent_mean")
                or fa.get("recent_earthquakes_mean"),
                "seis_mean": sq.get("seis_mean") or fa.get("seismicity_rate_mean"),
                "seis_max": sq.get("seis_max") or fa.get("seismicity_rate_max"),
                "catalog_mean": fa.get("catalog_earthquakes_mean"),
            }
        )
    return cycles


def render_html(rows: list[dict], cycles: list[dict], log_name: str) -> str:
    counts = Counter(r["event"] for r in rows)
    sliding = next((r.get("sliding") for r in rows if "sliding" in r), None)
    payload = {
        "log_name": log_name,
        "sliding": sliding,
        "n_lines": len(rows),
        "n_cycles": len(cycles),
        "event_counts": dict(counts),
        "cycles": cycles,
        "rows": rows,
    }
    data_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>MAGNET incremental TRACE — {html.escape(log_name)}</title>
<style>
  :root {{
    --bg: #f7f4ef;
    --ink: #1c1a16;
    --muted: #6a645a;
    --card: #fffdf8;
    --line: #ddd4c6;
    --accent: #0b6e4f;
    --accent2: #b35c1e;
    --chip: #ebe4d8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.4;
  }}
  header {{
    padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--line);
    background: var(--card);
  }}
  header h1 {{ margin: 0 0 .35rem; font-size: 1.35rem; font-weight: 650; }}
  header p {{ margin: 0; color: var(--muted); font-size: .95rem; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; padding: 1rem 1.25rem 2.5rem; }}
  .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(4, 1fr); }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 0.9rem 1rem;
  }}
  .card h2 {{ margin: 0 0 .6rem; font-size: .85rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }}
  .stat {{ font-size: 1.6rem; font-weight: 700; }}
  .stat small {{ font-size: .85rem; color: var(--muted); font-weight: 500; }}
  canvas.plot {{ width: 100%; height: 220px; display: block; }}
  .wide {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th, td {{ padding: .4rem .45rem; border-bottom: 1px solid var(--line);
    text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; position: sticky; top: 0;
    background: var(--card); }}
  .scroll {{ max-height: 420px; overflow: auto; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: .4rem; }}
  .chip {{ background: var(--chip); border-radius: 999px; padding: .2rem .55rem;
    font-size: .78rem; color: var(--ink); }}
  .filters {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .75rem; }}
  .filters label {{ font-size: .85rem; color: var(--muted); }}
  select, input[type=search] {{
    border: 1px solid var(--line); border-radius: 6px; padding: .35rem .5rem;
    background: #fff; font: inherit;
  }}
  .timeline {{ display: flex; flex-direction: column; gap: .35rem; }}
  .tl-row {{
    display: grid; grid-template-columns: 64px 1fr; gap: .6rem; align-items: start;
    font-size: .82rem;
  }}
  .tl-step {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
  .tl-body {{
    background: #f3eee5; border-radius: 6px; padding: .35rem .5rem;
    border-left: 3px solid var(--accent);
  }}
  .tl-body.query {{ border-left-color: var(--accent); }}
  .tl-body.ingest {{ border-left-color: var(--accent2); }}
  .tl-body.other {{ border-left-color: #888; }}
  .tl-event {{ font-weight: 600; }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>MAGNET incremental TRACE viewer</h1>
  <p>Source: <code id="logName"></code> · Phase C query → predict → ingest cycles</p>
</header>
<div class="wrap">
  <div class="grid" id="summary"></div>
  <div class="grid" style="margin-top:1rem">
    <div class="card wide"><h2>Catalog growth &amp; seismicity candidates</h2>
      <canvas class="plot" id="c1" height="220"></canvas></div>
    <div class="card wide"><h2>Sampled magnitude &amp; feature means</h2>
      <canvas class="plot" id="c2" height="220"></canvas></div>
    <div class="card wide"><h2>Query map (lng / lat)</h2>
      <canvas class="plot" id="c3" height="260"></canvas></div>
    <div class="card wide">
      <h2>Per-cycle table</h2>
      <div class="scroll"><table id="cycleTable"></table></div>
    </div>
    <div class="card wide">
      <h2>Raw TRACE events</h2>
      <div class="filters">
        <label>Filter
          <select id="eventFilter"><option value="">all events</option></select>
        </label>
        <label>Search <input id="search" type="search" placeholder="t=, mag, cell…"/></label>
      </div>
      <div class="timeline scroll" id="timeline" style="max-height:520px"></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};

function el(tag, attrs={{}}, kids=[]) {{
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k,v]) => {{
    if (k === 'className') n.className = v;
    else if (k === 'text') n.textContent = v;
    else n.setAttribute(k, v);
  }});
  (Array.isArray(kids) ? kids : [kids]).forEach(c => {{
    if (c == null) return;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }});
  return n;
}}

function drawLine(canvas, series, opts={{}}) {{
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0,0,W,H);
  const pad = {{l:44, r:16, t:16, b:36}};
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const xs = series[0].x;
  let ymin = Infinity, ymax = -Infinity;
  series.forEach(s => s.y.forEach(v => {{
    if (v == null || Number.isNaN(v)) return;
    ymin = Math.min(ymin, v); ymax = Math.max(ymax, v);
  }}));
  if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) return;
  if (ymin === ymax) {{ ymin -= 1; ymax += 1; }}
  const x0 = xs[0], x1 = xs[xs.length-1] || 1;
  const xmap = x => pad.l + ((x - x0) / (x1 - x0 || 1)) * iw;
  const ymap = y => pad.t + ih - ((y - ymin) / (ymax - ymin)) * ih;
  ctx.strokeStyle = '#ddd4c6'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ih); ctx.lineTo(pad.l+iw, pad.t+ih); ctx.stroke();
  ctx.fillStyle = '#6a645a'; ctx.font = '11px sans-serif';
  ctx.fillText(opts.ylabel || '', 4, 12);
  ctx.fillText(String(ymin.toPrecision(3)), 4, ymap(ymin));
  ctx.fillText(String(ymax.toPrecision(3)), 4, ymap(ymax)+10);
  const colors = ['#0b6e4f', '#b35c1e', '#2f5d8c', '#7a3e9d'];
  series.forEach((s, si) => {{
    ctx.strokeStyle = colors[si % colors.length];
    ctx.fillStyle = colors[si % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    s.y.forEach((y, i) => {{
      if (y == null || Number.isNaN(y)) {{ started = false; return; }}
      const X = xmap(s.x[i]), Y = ymap(y);
      if (!started) {{ ctx.moveTo(X,Y); started = true; }}
      else ctx.lineTo(X,Y);
    }});
    ctx.stroke();
    s.y.forEach((y, i) => {{
      if (y == null || Number.isNaN(y)) return;
      ctx.beginPath(); ctx.arc(xmap(s.x[i]), ymap(y), 2.5, 0, Math.PI*2); ctx.fill();
    }});
  }});
  // legend
  let lx = pad.l;
  series.forEach((s, si) => {{
    ctx.fillStyle = colors[si % colors.length];
    ctx.fillRect(lx, H - 18, 10, 10);
    ctx.fillStyle = '#1c1a16';
    ctx.fillText(s.label, lx + 14, H - 9);
    lx += ctx.measureText(s.label).width + 36;
  }});
}}

function drawScatter(canvas, points) {{
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0,0,W,H);
  const pad = {{l:48, r:16, t:16, b:36}};
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const xs = points.map(p => p.lng), ys = points.map(p => p.lat);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  const dx = (xmax - xmin) || 1, dy = (ymax - ymin) || 1;
  xmin -= dx*0.05; xmax += dx*0.05; ymin -= dy*0.05; ymax += dy*0.05;
  const xmap = x => pad.l + ((x - xmin)/(xmax-xmin))*iw;
  const ymap = y => pad.t + ih - ((y - ymin)/(ymax-ymin))*ih;
  ctx.strokeStyle = '#ddd4c6'; ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ih); ctx.lineTo(pad.l+iw, pad.t+ih); ctx.stroke();
  ctx.fillStyle = '#6a645a'; ctx.font = '11px sans-serif';
  ctx.fillText('lng', pad.l+iw-20, H-8);
  ctx.fillText('lat', 8, pad.t+10);
  points.forEach((p, i) => {{
    const t = i / Math.max(points.length-1, 1);
    ctx.fillStyle = `rgb(${{Math.round(11+t*160)}}, ${{Math.round(110-t*40)}}, ${{Math.round(79+t*40)}})`;
    ctx.beginPath(); ctx.arc(xmap(p.lng), ymap(p.lat), 4 + (p.sampled_mag-2)*1.2, 0, Math.PI*2); ctx.fill();
  }});
}}

document.getElementById('logName').textContent = DATA.log_name;
const summary = document.getElementById('summary');
[
  ['TRACE lines', DATA.n_lines],
  ['Predict cycles', DATA.n_cycles],
  ['Sliding', DATA.sliding ? 'Phase C' : 'Phase B / unknown'],
  ['Ingests', DATA.event_counts['FeatureState.ingest'] || 0],
].forEach(([k,v]) => {{
  summary.appendChild(el('div', {{className:'card'}}, [
    el('h2', {{text:k}}), el('div', {{className:'stat', text:String(v)}})
  ]));
}});
const chips = el('div', {{className:'card wide'}});
chips.appendChild(el('h2', {{text:'Event counts'}}));
const chipWrap = el('div', {{className:'chips'}});
Object.entries(DATA.event_counts).sort((a,b)=>b[1]-a[1]).forEach(([k,v]) => {{
  chipWrap.appendChild(el('span', {{className:'chip', text:`${{k.split('.').pop()}} × ${{v}}`}}));
}});
chips.appendChild(chipWrap);
summary.appendChild(chips);

const cycles = DATA.cycles;
const xs = cycles.map(c => c.i);
drawLine(document.getElementById('c1'), [
  {{label:'n_catalog (after ingest)', x: xs, y: cycles.map(c => c.n_catalog)}},
  {{label:'seis n_candidates', x: xs, y: cycles.map(c => c.n_candidates)}},
  {{label:'recent n_real', x: xs, y: cycles.map(c => c.n_real)}},
], {{ylabel:'count'}});
drawLine(document.getElementById('c2'), [
  {{label:'sampled_mag', x: xs, y: cycles.map(c => c.sampled_mag)}},
  {{label:'seis_mean', x: xs, y: cycles.map(c => c.seis_mean)}},
  {{label:'recent_mean (scaled down /1e6)', x: xs, y: cycles.map(c => c.recent_mean == null ? null : c.recent_mean/1e6)}},
], {{ylabel:'value'}});
drawScatter(document.getElementById('c3'), cycles);

const table = document.getElementById('cycleTable');
table.appendChild(el('thead', {{}}, el('tr', {{}},
  ['#','t','lng','lat','mag','catalog','candidates','n_real','seis_mean','recent_mean'].map(h => el('th', {{text:h}}))
)));
const tb = el('tbody');
cycles.forEach(c => {{
  tb.appendChild(el('tr', {{}}, [
    c.i, c.t, c.lng?.toFixed?.(3) ?? c.lng, c.lat?.toFixed?.(3) ?? c.lat,
    c.sampled_mag?.toFixed?.(3), c.n_catalog, c.n_candidates, c.n_real,
    c.seis_mean?.toExponential?.(3) ?? '',
    c.recent_mean?.toExponential?.(3) ?? '',
  ].map(v => el('td', {{text: String(v ?? '')}}))));
}});
table.appendChild(tb);

const filter = document.getElementById('eventFilter');
[...new Set(DATA.rows.map(r => r.event))].sort().forEach(ev => {{
  filter.appendChild(el('option', {{value: ev, text: ev}}));
}});
const timeline = document.getElementById('timeline');
function kind(ev) {{
  if (ev.includes('features_at') || ev.includes('features_for_example') || ev.includes('predict_step')) return 'query';
  if (ev.includes('ingest') || ev.includes('append_row')) return 'ingest';
  return 'other';
}}
function renderTimeline() {{
  const q = document.getElementById('search').value.toLowerCase();
  const evf = filter.value;
  timeline.innerHTML = '';
  DATA.rows.forEach(r => {{
    if (evf && r.event !== evf) return;
    const blob = JSON.stringify(r).toLowerCase();
    if (q && !blob.includes(q)) return;
    const {{step, event, ...rest}} = r;
    const bits = Object.entries(rest)
      .filter(([k]) => !k.endsWith('_shape'))
      .slice(0, 12)
      .map(([k,v]) => `${{k}}=${{typeof v === 'number' ? (Math.abs(v)>1e3 || (Math.abs(v)>0 && Math.abs(v)<1e-2) ? Number(v).toExponential(3) : v) : v}}`)
      .join(' · ');
    timeline.appendChild(el('div', {{className:'tl-row'}}, [
      el('div', {{className:'tl-step', text:'#'+step}}),
      el('div', {{className:'tl-body '+kind(event)}}, [
        el('div', {{className:'tl-event', text: event}}),
        el('div', {{className:'muted', text: bits}}),
      ]),
    ]));
  }});
}}
filter.addEventListener('change', renderTimeline);
document.getElementById('search').addEventListener('input', renderTimeline);
renderTimeline();
window.addEventListener('resize', () => {{
  drawLine(document.getElementById('c1'), [
    {{label:'n_catalog (after ingest)', x: xs, y: cycles.map(c => c.n_catalog)}},
    {{label:'seis n_candidates', x: xs, y: cycles.map(c => c.n_candidates)}},
    {{label:'recent n_real', x: xs, y: cycles.map(c => c.n_real)}},
  ], {{ylabel:'count'}});
  drawLine(document.getElementById('c2'), [
    {{label:'sampled_mag', x: xs, y: cycles.map(c => c.sampled_mag)}},
    {{label:'seis_mean', x: xs, y: cycles.map(c => c.seis_mean)}},
    {{label:'recent_mean (/1e6)', x: xs, y: cycles.map(c => c.recent_mean == null ? null : c.recent_mean/1e6)}},
  ], {{ylabel:'value'}});
  drawScatter(document.getElementById('c3'), cycles);
}});
</script>
</body>
</html>
"""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=repo / "outputs" / "magnet_incremental_trace_phase_c.log",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: same stem as --log with .html",
    )
    args = parser.parse_args()
    log_path = args.log
    out_path = args.out or log_path.with_suffix(".html")
    rows = parse_log(log_path.read_text(encoding="utf-8"))
    cycles = build_cycles(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_html(rows, cycles, log_path.name), encoding="utf-8"
    )
    print(f"Wrote {out_path} ({len(rows)} lines, {len(cycles)} cycles)")


if __name__ == "__main__":
    main()
