"""Generate a self-contained HTML tool for hand-labeling sku_presence.csv.

Labeling is not "type sku_ids from memory against 2470 catalog rows". It is:
watch a video, and for each product you recognise, search the catalog by brand /
type and click it. This builds that as one offline HTML file with the whole
catalog embedded, per-video baskets, autosave, and CSV export in the exact
``video,expected_sku_ids`` format the evaluator expects.

    python3 -m ml.make_labeling_tool
    open reports/labeling/label_tool.html

Progress autosaves in the browser (localStorage); "Скачать CSV" downloads
``sku_presence.csv``; "Копировать CSV" copies it so you can paste it back.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_catalog(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            sku = (r.get("sku_id") or "").strip()
            brand = (r.get("brand_name") or "").strip()
            if not sku or not brand or brand == "-":
                continue
            rows.append({
                "id": sku,
                "b": brand,
                "m": (r.get("model_name") or "").strip(),
                "a": (r.get("article_codes") or "").strip(),
                "c": (r.get("category") or "").strip(),
            })
    return rows


def find_videos() -> list[str]:
    names = set()
    for p in REPO.rglob("*"):
        if p.suffix.lower() in (".mov", ".mp4"):
            rel = str(p.relative_to(REPO))
            if rel.startswith("var/"):
                continue
            names.add(p.name)
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog/own_products.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports/labeling/label_tool.html"))
    ap.add_argument("--videos", nargs="*", default=None,
                    help="video names to label (default: auto-detect on disk)")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    videos = args.videos if args.videos else find_videos()
    brands = sorted({row["b"] for row in catalog})

    html = TEMPLATE
    html = html.replace("__CATALOG__", json.dumps(catalog, ensure_ascii=False))
    html = html.replace("__VIDEOS__", json.dumps(videos, ensure_ascii=False))
    html = html.replace("__BRANDS__", json.dumps(brands, ensure_ascii=False))
    html = html.replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"catalog={len(catalog)} videos={len(videos)} -> {args.out}")
    print(f"open {args.out}")


TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка SKU — sku_presence.csv</title>
<style>
  :root { --bg:#0f1115; --card:#1a1d24; --line:#2a2f3a; --fg:#e8eaed; --mut:#9aa0aa;
          --acc:#2d78dc; --ok:#24b667; --warn:#f08923; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:12px 16px; border-bottom:1px solid var(--line);
           display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; }
  .muted { color:var(--mut); }
  button { background:var(--card); color:var(--fg); border:1px solid var(--line);
           border-radius:8px; padding:6px 10px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--acc); }
  button.primary { background:var(--acc); border-color:var(--acc); }
  button.ok { background:var(--ok); border-color:var(--ok); }
  .wrap { display:grid; grid-template-columns:300px 1fr; gap:0; height:calc(100vh - 58px); }
  .col { overflow:auto; padding:12px; }
  .col.left { border-right:1px solid var(--line); }
  .vid { padding:8px 10px; border:1px solid var(--line); border-radius:8px;
         margin-bottom:6px; cursor:pointer; display:flex; justify-content:space-between; }
  .vid.active { border-color:var(--acc); background:#13233b; }
  .vid .n { background:var(--card); border-radius:20px; padding:0 8px; color:var(--mut); }
  .vid .n.has { background:var(--ok); color:#fff; }
  .search { position:sticky; top:0; background:var(--bg); padding-bottom:8px; }
  input[type=text] { width:100%; padding:9px 11px; background:var(--card);
         border:1px solid var(--line); border-radius:8px; color:var(--fg); font-size:14px; }
  .brands { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
  .brands button.on { background:var(--acc); border-color:var(--acc); }
  table { width:100%; border-collapse:collapse; }
  td,th { padding:6px 8px; border-bottom:1px solid var(--line); text-align:left;
          vertical-align:top; }
  th { color:var(--mut); font-weight:600; position:sticky; top:54px; background:var(--bg); }
  .add { white-space:nowrap; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 14px; }
  .chip { background:#13233b; border:1px solid var(--acc); border-radius:20px;
          padding:4px 10px; display:flex; gap:8px; align-items:center; }
  .chip b { font-weight:600; } .chip .x { cursor:pointer; color:var(--warn); }
  .hint { background:var(--card); border:1px solid var(--line); border-radius:8px;
          padding:10px 12px; margin-bottom:12px; color:var(--mut); }
  code { background:#000; padding:1px 5px; border-radius:4px; }
  dialog { background:var(--card); color:var(--fg); border:1px solid var(--line);
           border-radius:10px; width:min(680px,92vw); }
  textarea { width:100%; height:160px; background:var(--bg); color:var(--fg);
             border:1px solid var(--line); border-radius:8px; }
</style>
</head>
<body>
<header>
  <h1>Разметка SKU</h1>
  <span class="muted">собрано __GENERATED__ · каталог <b id="catn"></b> SKU</span>
  <span style="flex:1"></span>
  <button onclick="importCsv()">Загрузить CSV</button>
  <button onclick="copyCsv()">Копировать CSV</button>
  <button class="primary" onclick="downloadCsv()">Скачать sku_presence.csv</button>
</header>

<div class="wrap">
  <div class="col left">
    <div class="hint">
      Выбери видео → справа найди товар по бренду/типу → <b>＋</b>. Если на видео
      нет наших товаров — оставь пусто и переходи дальше. Прогресс сохраняется
      автоматически.
    </div>
    <div id="videos"></div>
    <button style="margin-top:8px" onclick="addVideo()">+ другое видео…</button>
  </div>

  <div class="col right">
    <div class="search">
      <input id="q" type="text" placeholder="Поиск: бренд, модель, артикул, тип…  напр. «huter триммер» или «65/3»" oninput="render()">
      <div class="brands" id="brands"></div>
    </div>
    <h3 id="curtitle"></h3>
    <div class="chips" id="chips"></div>
    <table>
      <thead><tr><th>Бренд</th><th>Модель</th><th>Артикул</th><th>Тип</th><th></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <p class="muted" id="more"></p>
  </div>
</div>

<dialog id="impdlg">
  <h3>Вставь существующий sku_presence.csv</h3>
  <textarea id="imptext" placeholder="video,expected_sku_ids&#10;IMG_8886.MOV,HUTER_64_1_20_DY5000LX_DY6500LX"></textarea>
  <div style="margin-top:10px;text-align:right">
    <button onclick="impdlg.close()">Отмена</button>
    <button class="ok" onclick="doImport()">Загрузить</button>
  </div>
</dialog>

<script>
const CATALOG = __CATALOG__;
const VIDEOS  = __VIDEOS__;
const BRANDS  = __BRANDS__;
const KEY = "sku_label_v1";
let state = JSON.parse(localStorage.getItem(KEY) || "null") || {videos: VIDEOS.slice(), sel: {}, cur: VIDEOS[0] || ""};
let brandFilter = "";
CATALOG.forEach(r => r._s = (r.b+" "+r.m+" "+r.a+" "+r.c+" "+r.id).toLowerCase());
document.getElementById("catn").textContent = CATALOG.length;

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function selOf(v){ return state.sel[v] || (state.sel[v]=[]); }

function renderVideos(){
  const el = document.getElementById("videos"); el.innerHTML="";
  state.videos.forEach(v => {
    const n = selOf(v).length;
    const d = document.createElement("div");
    d.className = "vid" + (v===state.cur?" active":"");
    d.innerHTML = `<span>${v}</span><span class="n ${n?'has':''}">${n}</span>`;
    d.onclick = () => { state.cur=v; save(); renderAll(); };
    el.appendChild(d);
  });
}
function renderBrands(){
  const el = document.getElementById("brands"); el.innerHTML="";
  ["(все)", ...BRANDS].forEach(b => {
    const val = b==="(все)" ? "" : b;
    const btn = document.createElement("button");
    btn.textContent = b; if(val===brandFilter) btn.className="on";
    btn.onclick = () => { brandFilter=val; renderBrands(); render(); };
    el.appendChild(btn);
  });
}
function renderChips(){
  document.getElementById("curtitle").textContent = state.cur ? ("Видео: "+state.cur) : "Выбери видео";
  const el = document.getElementById("chips"); el.innerHTML="";
  selOf(state.cur).forEach(id => {
    const r = CATALOG.find(x=>x.id===id) || {b:"?",m:"",id};
    const c = document.createElement("span"); c.className="chip";
    c.innerHTML = `<b>${r.b}</b> ${r.m||r.id} <span class="x">✕</span>`;
    c.querySelector(".x").onclick = () => { state.sel[state.cur]=selOf(state.cur).filter(x=>x!==id); save(); renderAll(); };
    el.appendChild(c);
  });
}
function render(){
  const terms = document.getElementById("q").value.toLowerCase().split(/\s+/).filter(Boolean);
  const cur = selOf(state.cur);
  let hits = CATALOG.filter(r => (!brandFilter || r.b===brandFilter) && terms.every(t => r._s.includes(t)));
  const total = hits.length; hits = hits.slice(0,300);
  const tb = document.getElementById("rows"); tb.innerHTML="";
  hits.forEach(r => {
    const tr = document.createElement("tr");
    const has = cur.includes(r.id);
    tr.innerHTML = `<td>${r.b}</td><td>${r.m||"—"}</td><td>${r.a||"—"}</td><td>${r.c||"—"}</td>
      <td class="add"></td>`;
    const btn = document.createElement("button");
    btn.textContent = has ? "✓" : "＋"; if(has) btn.className="ok";
    btn.onclick = () => { if(!state.cur){alert("Сначала выбери видео слева");return;}
      if(has){ state.sel[state.cur]=cur.filter(x=>x!==r.id);} else { cur.push(r.id);} save(); renderAll(); };
    tr.querySelector(".add").appendChild(btn);
    tb.appendChild(tr);
  });
  document.getElementById("more").textContent =
    total>300 ? `показано 300 из ${total} — уточни поиск` : `${total} совпадений`;
}
function renderAll(){ renderVideos(); renderChips(); render(); }

function addVideo(){
  const v = prompt("Имя видеофайла (как в папке), напр. IMG_9001.MOV:");
  if(!v) return; if(!state.videos.includes(v)) state.videos.push(v);
  state.cur=v; save(); renderAll();
}
function buildCsv(){
  let out = "video,expected_sku_ids\n";
  state.videos.forEach(v => { out += `${v},${selOf(v).join("|")}\n`; });
  return out;
}
function downloadCsv(){
  const blob = new Blob([buildCsv()], {type:"text/csv"});
  const a = document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="sku_presence.csv"; a.click();
}
function copyCsv(){ navigator.clipboard.writeText(buildCsv()).then(()=>alert("CSV скопирован — вставь его обратно в чат или в файл data/eval/sku_presence.csv")); }
function importCsv(){ document.getElementById("impdlg").showModal(); }
function doImport(){
  const txt = document.getElementById("imptext").value.trim();
  txt.split(/\r?\n/).slice(1).forEach(line => {
    const i = line.indexOf(","); if(i<0) return;
    const v = line.slice(0,i).trim(); const ids = line.slice(i+1).replace(/^"|"$/g,"");
    if(!v) return; if(!state.videos.includes(v)) state.videos.push(v);
    state.sel[v] = ids.split(/[|;]/).map(s=>s.trim()).filter(Boolean);
  });
  save(); document.getElementById("impdlg").close(); renderAll();
}
renderBrands(); renderAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
