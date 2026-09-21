"""
Primary deliverable: the RUL model embedded in a Limble-style CMMS. -> docs/index.html
Recreates the look of a modern SaaS CMMS (left nav, top bar, asset grid, expandable
work-order panels) with the Remaining Useful Life prediction surfaced as an AI
column on the Assets view, one row per machine. Mirrors the role Case 01's ERP
dashboard plays for the defect scorer.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data_source" / "oee_predmaint.duckdb"
SNAP    = REPO / "ml" / "data" / "scoring" / "fleet_snapshot.parquet"
METRICS = REPO / "ml" / "models" / "metrics.json"
OUT     = REPO / "docs" / "index.html"

# CMMS priority palette (authentic maintenance-system colours, not BRIM report palette)
PRI = {"CRITICAL": "#e03131", "ELEVATED": "#f08c00", "MONITOR": "#1c7ed6", "OK": "#2f9e44"}
PRI_ORDER = {"CRITICAL": 0, "ELEVATED": 1, "MONITOR": 2, "OK": 3}
ACTION = {
    "CRITICAL": "Create a corrective work order and schedule within 7 days.",
    "ELEVATED": "Plan maintenance in the next 2 to 3 weeks; watch the alarm trend.",
    "MONITOR":  "Continue condition monitoring; no immediate action required.",
    "OK":       "Healthy. Proceed with the next scheduled preventive maintenance.",
}

meta = json.loads(METRICS.read_text(encoding="utf-8"))
snap = pd.read_parquet(SNAP)

con = duckdb.connect(str(DB_PATH), read_only=True)
mp = con.execute("select * from mart_oee__machine_performance").df()
pm = con.execute("select * from mart_oee__pm_compliance").df()
con.close()
mp["period_month"] = pd.to_datetime(mp["period_month"])
QUALITY = float(mp["quality_rate"].iloc[0])
months = sorted(mp["period_month"].unique())
cur_month, prior_month = months[-1], months[-2]


def machine_oee(month):
    g = mp[mp["period_month"] == month].groupby("machine_id")
    a = g["run_minutes"].sum() / g["planned_production_minutes"].sum()
    p = g.apply(lambda x: (x["performance"] * x["run_minutes"]).sum() / x["run_minutes"].sum(),
                include_groups=False)
    return a * p * QUALITY


cur_oee, prior_oee = machine_oee(cur_month), machine_oee(prior_month)
attr = mp.groupby("machine_id").agg(machine_type=("machine_type", "first"),
                                    controller_type=("controller_type", "first"),
                                    location_cell=("location_cell", "first"),
                                    machine_age_years=("machine_age_years", "first")).reset_index()
pm_idx = pm.set_index("machine_id")

df = snap.merge(attr, on="machine_id", how="left", suffixes=("", "_a"))
df["oee"] = df["machine_id"].map(cur_oee)
df["oee_prev"] = df["machine_id"].map(prior_oee)
df = df.sort_values("predicted_days_to_failure").sort_values(
    "priority", key=lambda s: s.map(PRI_ORDER), kind="stable").reset_index(drop=True)

counts = df["priority"].value_counts()
as_of = pd.Timestamp(snap["observation_date"].max()).strftime("%b %d, %Y")
open_wos = int(counts.get("CRITICAL", 0) + counts.get("ELEVATED", 0))


def oee_bar(v):
    c = "#2f9e44" if v >= 0.70 else "#f08c00" if v >= 0.50 else "#e03131"
    return (f'<div class="hbar"><div class="hbar-fill" style="width:{v*100:.0f}%;background:{c};"></div></div>'
            f'<span class="hbar-txt">{v:.0%}</span>')


def trend_icon(mid, cur, prev):
    d = (cur if pd.notna(cur) else 0) - (prev if pd.notna(prev) else 0)
    if d > 0.005:
        return '<span style="color:#2f9e44;">&#9650;</span>'
    if d < -0.005:
        return '<span style="color:#e03131;">&#9660;</span>'
    return '<span style="color:#adb5bd;">&#8594;</span>'


rows_html = ""
for i, r in df.iterrows():
    mtype = r["machine_type"]
    make = {"Fanuc": "Fanuc", "Haas": "Haas", "Mazak": "Mazak"}.get(r["controller_type"], r["controller_type"])
    asset = f"{make} {mtype}"
    pm_row = pm_idx.loc[r["machine_id"]] if r["machine_id"] in pm_idx.index else None
    pm_status = pm_row["pm_status"] if pm_row is not None else "n/a"
    pm_next = (pd.to_datetime(pm_row["next_pm_due_date"]).strftime("%m/%d/%y")
               if pm_row is not None and pd.notna(pm_row["next_pm_due_date"]) else "-")
    pm_col = {"Overdue": "#e03131", "Due Soon": "#f08c00", "On Track": "#2f9e44"}.get(pm_status, "#868e96")
    pcolor = PRI.get(r["priority"], "#868e96")
    rul = r["predicted_days_to_failure"]
    drivers = list(r["risk_drivers"]) if r["risk_drivers"] is not None else []
    expandable = r["priority"] in ("CRITICAL", "ELEVATED", "MONITOR")

    rows_html += f"""
    <tr class="asset-row" {'onclick="toggle(%d)"' % i if expandable else ''} style="{'cursor:pointer;' if expandable else ''}" id="row-{i}">
      <td><div class="asset-cell"><span class="asset-name">{asset}</span><span class="asset-id">{r['machine_id']} &middot; {int(r['machine_age_years'])} yrs</span></div></td>
      <td>{r['location_cell']}</td>
      <td class="oee-cell">{oee_bar(float(r['oee']))} {trend_icon(r['machine_id'], r['oee'], r['oee_prev'])}</td>
      <td><span class="pm-pill" style="color:{pm_col};border-color:{pm_col};">{pm_status}</span><div class="sub">due {pm_next}</div></td>
      <td class="rul-cell"><span class="rul-days" style="color:{pcolor};">{rul:.0f}</span><span class="rul-unit">days</span></td>
      <td><span class="pri-pill" style="background:{pcolor};">{r['priority']}</span>{'<span class="chev">&#9662;</span>' if expandable else ''}</td>
    </tr>"""

    if expandable:
        drv = "".join(f'<div class="rf"><span class="rf-n">{j+1}</span>{d}</div>' for j, d in enumerate(drivers)) \
            or '<div class="rf"><span class="rf-n">1</span>Elevated risk based on current condition</div>'
        rows_html += f"""
    <tr class="panel-row" id="panel-{i}" style="display:none;"><td colspan="6">
      <div class="panel" style="border-left:4px solid {pcolor};">
        <div class="panel-grid">
          <div><div class="pl">Predicted failure risk</div>
            <div class="rf-lead"><span class="pri-pill" style="background:{pcolor};">{r['priority']}</span>
              &nbsp; predicted failure in <strong>{rul:.0f} days</strong></div>
            <div class="pl" style="margin-top:12px;">Top risk drivers</div>{drv}</div>
          <div><div class="pl">Recommended action</div>
            <div class="action">{ACTION.get(r['priority'], '')}</div>
            <div class="pl" style="margin-top:12px;">Asset detail</div>
            <div class="dgrid">
              <span class="dl">Asset</span><span>{asset} ({r['machine_id']})</span>
              <span class="dl">Cell</span><span>{r['location_cell']}</span>
              <span class="dl">Age</span><span>{int(r['machine_age_years'])} years</span>
              <span class="dl">Current OEE</span><span>{float(r['oee']):.0%}</span>
              <span class="dl">PM status</span><span>{pm_status} (due {pm_next})</span>
              <span class="dl">7-day alarms</span><span>{int(r['rolling_7d_alarm_count'])}</span>
            </div></div>
          <div class="panel-actions">
            <button class="btn btn-primary">Create Work Order</button>
            <button class="btn">Assign Technician</button>
            <button class="btn">Snooze 7 days</button>
            <button class="btn">View Asset</button>
          </div>
        </div>
      </div></td></tr>"""


def tile(t):
    n = int(counts.get(t, 0))
    return (f'<div class="tile"><div class="tile-n" style="color:{PRI[t]};">{n}</div>'
            f'<div class="tile-l"><span class="dot" style="background:{PRI[t]};"></span>{t.title()}</div></div>')


NAV = [("&#9632;", "Dashboard"), ("&#9776;", "Work Orders"), ("&#9993;", "Requests"),
       ("&#9881;", "Assets", True), ("&#128197;", "PM Schedule"), ("&#128230;", "Parts"), ("&#128202;", "Reports")]
nav_html = ""
for item in NAV:
    active = len(item) > 2
    nav_html += (f'<div class="nav-item{" active" if active else ""}">'
                 f'<span class="nav-ico">{item[0]}</span>{item[1]}</div>')

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Assets &middot; Limble CMMS</title>
<style>
  *,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#f1f3f5;
    color:#212529; font-size:14px; }}
  .app {{ display:flex; min-height:100vh; }}
  /* Sidebar */
  .side {{ width:210px; background:#1b2430; color:#c4ccd6; flex-shrink:0; display:flex; flex-direction:column; }}
  .brand {{ display:flex; align-items:center; gap:9px; padding:16px 18px; border-bottom:1px solid #2b3644; }}
  .brand .logo {{ width:26px; height:26px; border-radius:6px; background:#4cae4f; display:flex; align-items:center;
    justify-content:center; color:#fff; font-weight:800; font-size:15px; }}
  .brand .name {{ font-size:16px; font-weight:700; color:#fff; }}
  .brand .name span {{ font-weight:400; color:#8b96a5; font-size:12px; }}
  .nav {{ padding:10px 0; flex:1; }}
  .nav-item {{ display:flex; align-items:center; gap:11px; padding:10px 18px; font-size:13.5px; color:#c4ccd6;
    cursor:pointer; border-left:3px solid transparent; }}
  .nav-item:hover {{ background:#232f3e; color:#fff; }}
  .nav-item.active {{ background:#232f3e; color:#fff; border-left-color:#4cae4f; font-weight:600; }}
  .nav-ico {{ width:18px; text-align:center; opacity:.85; }}
  .side-foot {{ padding:14px 18px; border-top:1px solid #2b3644; font-size:12px; color:#8b96a5; }}
  /* Main */
  .main {{ flex:1; display:flex; flex-direction:column; min-width:0; }}
  .topbar {{ background:#fff; border-bottom:1px solid #dee2e6; padding:10px 22px; display:flex; align-items:center; gap:16px; }}
  .crumbs {{ font-size:13px; color:#868e96; }}
  .crumbs b {{ color:#212529; }}
  .search {{ margin-left:auto; }}
  .search input {{ width:240px; padding:7px 11px; border:1px solid #ced4da; border-radius:6px; font-size:13px; }}
  .search input:focus {{ outline:none; border-color:#4cae4f; }}
  .avatar {{ width:30px; height:30px; border-radius:50%; background:#4cae4f; color:#fff; display:flex; align-items:center;
    justify-content:center; font-weight:700; font-size:13px; }}
  .content {{ padding:22px; }}
  .page-head {{ display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:16px; }}
  .page-head h1 {{ font-size:20px; font-weight:700; }}
  .page-head .sub {{ font-size:13px; color:#868e96; margin-top:2px; }}
  .btn-new {{ background:#4cae4f; color:#fff; border:none; padding:9px 15px; border-radius:6px; font-size:13px;
    font-weight:600; cursor:pointer; }}
  .tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px; }}
  .tile {{ background:#fff; border:1px solid #e9ecef; border-radius:10px; padding:16px 18px; }}
  .tile-n {{ font-size:30px; font-weight:800; line-height:1; }}
  .tile-l {{ font-size:12px; color:#495057; margin-top:6px; text-transform:uppercase; letter-spacing:.5px; font-weight:600; }}
  .tile-l .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
  .card {{ background:#fff; border:1px solid #e9ecef; border-radius:10px; overflow:hidden; }}
  .card-head {{ display:flex; align-items:center; gap:10px; padding:13px 18px; border-bottom:1px solid #f1f3f5; }}
  .card-head .ct {{ font-size:15px; font-weight:700; }}
  .filters {{ margin-left:auto; display:flex; gap:8px; }}
  .filters select {{ font-size:12.5px; padding:5px 8px; border:1px solid #ced4da; border-radius:6px; background:#fff; }}
  table {{ width:100%; border-collapse:collapse; }}
  thead th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:#868e96;
    font-weight:700; padding:10px 16px; background:#f8f9fa; border-bottom:1px solid #e9ecef; white-space:nowrap; }}
  th.ai-col {{ color:#2b6cb0; }}
  .ai-tag {{ font-size:9px; font-weight:800; background:#e7f0ff; color:#2b6cb0; padding:1px 5px; border-radius:3px;
    margin-left:5px; letter-spacing:.3px; }}
  .info {{ color:#adb5bd; cursor:help; margin-left:3px; }}
  .asset-row td {{ padding:12px 16px; border-bottom:1px solid #f1f3f5; vertical-align:middle; }}
  .asset-row:hover td {{ background:#f8fbf8; }}
  .asset-cell {{ display:flex; flex-direction:column; }}
  .asset-name {{ font-weight:600; }}
  .asset-id {{ font-size:12px; color:#868e96; }}
  .sub {{ font-size:11px; color:#adb5bd; margin-top:2px; }}
  .oee-cell {{ white-space:nowrap; }}
  .hbar {{ display:inline-block; width:70px; height:7px; background:#e9ecef; border-radius:4px; overflow:hidden; vertical-align:middle; }}
  .hbar-fill {{ height:100%; }}
  .hbar-txt {{ font-size:12.5px; margin-left:7px; font-variant-numeric:tabular-nums; }}
  .pm-pill {{ display:inline-block; padding:2px 9px; border:1px solid; border-radius:11px; font-size:11.5px; font-weight:600; }}
  .rul-cell {{ white-space:nowrap; }}
  .rul-days {{ font-size:20px; font-weight:800; }}
  .rul-unit {{ font-size:11px; color:#adb5bd; margin-left:3px; }}
  .pri-pill {{ display:inline-block; padding:3px 11px; border-radius:12px; color:#fff; font-size:11px; font-weight:700;
    letter-spacing:.3px; }}
  .chev {{ color:#adb5bd; font-size:11px; margin-left:8px; }}
  /* Expand panel */
  .panel-row td {{ padding:0; background:#f8f9fa; border-bottom:2px solid #e9ecef; }}
  .panel {{ padding:18px 20px; }}
  .panel-grid {{ display:grid; grid-template-columns:1.2fr 1.2fr 200px; gap:26px; }}
  .pl {{ font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.8px; color:#868e96; margin-bottom:8px; }}
  .rf-lead {{ font-size:14px; margin-bottom:6px; }}
  .rf {{ display:flex; align-items:flex-start; gap:9px; margin-bottom:7px; font-size:13.5px; line-height:1.4; }}
  .rf-n {{ width:19px; height:19px; border-radius:50%; background:#1b2430; color:#fff; font-size:11px; font-weight:700;
    display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }}
  .action {{ background:#fff8e1; border:1px solid #ffe08a; border-radius:6px; padding:9px 12px; font-size:13px; color:#5f4b00; }}
  .dgrid {{ display:grid; grid-template-columns:auto 1fr; gap:5px 12px; font-size:12.5px; }}
  .dl {{ color:#868e96; font-weight:600; }}
  .panel-actions {{ display:flex; flex-direction:column; gap:8px; }}
  .btn {{ padding:8px 12px; font-size:12.5px; border:1px solid #ced4da; border-radius:6px; background:#fff; cursor:pointer; text-align:center; }}
  .btn:hover {{ background:#f1f3f5; }}
  .btn-primary {{ background:#4cae4f; color:#fff; border-color:#4cae4f; font-weight:600; }}
  .btn-primary:hover {{ background:#409443; }}
  .statusbar {{ padding:10px 22px; font-size:12px; color:#868e96; display:flex; gap:14px; align-items:center; }}
  .statusbar .sep {{ color:#dee2e6; }}
</style></head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand"><div class="logo">L</div><div class="name">Limble<span> CMMS</span></div></div>
    <nav class="nav">{nav_html}</nav>
    <div class="side-foot">Maintenance Planner<br><span style="color:#c4ccd6;">A. Reyes</span></div>
  </aside>
  <div class="main">
    <div class="topbar">
      <div class="crumbs">Assets <span style="color:#ced4da;">&rsaquo;</span> <b>Predictive Health</b></div>
      <div class="search"><input type="text" id="q" placeholder="Search assets..."></div>
      <div class="avatar">AR</div>
    </div>
    <div class="content">
      <div class="page-head">
        <div><h1>Assets &middot; Predictive Health</h1>
          <div class="sub">Fleet ranked by predicted time to next unplanned failure &middot; updated {as_of}</div></div>
        <button class="btn-new">+ New Work Order</button>
      </div>
      <div class="tiles">{tile("CRITICAL")}{tile("ELEVATED")}{tile("MONITOR")}{tile("OK")}</div>
      <div class="card">
        <div class="card-head">
          <span class="ct">Machine Assets</span>
          <div class="filters">
            <select id="fpri"><option value="All">All priorities</option><option>CRITICAL</option><option>ELEVATED</option><option>MONITOR</option><option>OK</option></select>
          </div>
        </div>
        <table>
          <thead><tr>
            <th>Asset</th><th>Cell</th><th>Health (OEE)</th><th>Preventive Maint.</th>
            <th class="ai-col">Predicted Failure<span class="ai-tag">AI</span><span class="info" title="Predicted days to next unplanned failure. Powered by BRIM RUL Predictor v{meta['model_version']}. Click a row for detail.">&#9432;</span></th>
            <th>Priority</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
      <div class="statusbar">
        <span>{len(df)} assets</span><span class="sep">|</span>
        <span style="color:#e03131;font-weight:600;">{int(counts.get('CRITICAL',0))} critical</span><span class="sep">|</span>
        <span style="color:#f08c00;font-weight:600;">{int(counts.get('ELEVATED',0))} elevated</span><span class="sep">|</span>
        <span>{open_wos} suggested work orders</span><span class="sep">|</span>
        <span>Predictions by BRIM RUL Predictor ({meta['best_model_type']}, test MAE {meta['test']['mae']:.1f}d)</span>
      </div>
    </div>
  </div>
</div>
<script>
  function toggle(i){{ var p=document.getElementById('panel-'+i); if(!p) return;
    p.style.display = (p.style.display==='none'||!p.style.display)?'table-row':'none'; }}
  document.getElementById('fpri').addEventListener('change', function(){{
    var v=this.value;
    document.querySelectorAll('.asset-row').forEach(function(r){{
      var pill=r.querySelector('.pri-pill'); if(!pill) return;
      var show = (v==='All') || pill.textContent.trim()===v;
      r.style.display = show?'':'none';
      var pan=document.getElementById('panel-'+r.id.split('-')[1]); if(pan&&!show) pan.style.display='none';
    }});
  }});
  document.getElementById('q').addEventListener('input', function(){{
    var q=this.value.toLowerCase();
    document.querySelectorAll('.asset-row').forEach(function(r){{
      r.style.display = r.textContent.toLowerCase().includes(q)?'':'none';
    }});
  }});
</script>
</body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"CMMS asset view written to {OUT}")
print(f"  {len(df)} assets | CRITICAL {int(counts.get('CRITICAL',0))} ELEVATED {int(counts.get('ELEVATED',0))}")
