"""
FPL H2H Tracker — Streamlit app (Google Sheets backend)
Features:
- Fetch H2H league members via FPL API
- Pull per-GW official points (finished GWs) and show ranking by FPL points (descending)
- Persist data (members, gw_scores, gw_rank, gw_predictions) to **Google Sheets** (no data loss on reload)
- Monte Carlo simulation to estimate P(top1/2/3) each GW

Setup:
1) Streamlit Cloud → Settings → Secrets, paste (replace with your values):

[FYI] .streamlit/secrets.toml
FPL_LEAGUE_ID = "1007448"        # your H2H league id
INVITE_CODE = "u3dip1"           # optional, to display invite code
GSPREAD_SHEET_ID = "1AbC...xyz"  # Google Sheets spreadsheet ID

# Service Account JSON (rename the key to gcp_service_account)
[gcp_service_account]
type = "service_account"
project_id = "your-project"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "service-account@your-project.iam.gserviceaccount.com"
client_id = "..."
# (include the rest of fields exactly as in the downloaded JSON)

2) Share the Google Sheet with: service-account@your-project.iam.gserviceaccount.com (Editor)
3) Create **empty** spreadsheet, no tabs needed (the app will create tabs if missing)
4) requirements.txt additions:
requests
streamlit
numpy
pandas
gspread
google-auth
matplotlib

5) Run locally:
  pip install -r requirements.txt
  streamlit run app.py
"""

import os, json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple

# =========================
# Config from Secrets
# =========================
FPL_LEAGUE_ID = st.secrets.get("FPL_LEAGUE_ID")
INVITE_CODE   = st.secrets.get("INVITE_CODE", "")
SHEET_ID      = st.secrets.get("GSPREAD_SHEET_ID")
SVC_INFO      = st.secrets.get("gcp_service_account")

# =========================
# Streamlit Page
# =========================
st.set_page_config(page_title="FPL H2H Tracker", page_icon="⚽", layout="wide")
st.title("⚽ FPL H2H Tracker — Auto Points, Rank & Top%")
if INVITE_CODE:
    st.info(f"👉 Mã mời vào league: `{INVITE_CODE}`")

# =========================
# Google Sheets helpers (gspread)
# =========================
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_resource(show_spinner=False)
def get_gs_client():
    if not SVC_INFO:
        raise RuntimeError("Service account JSON missing in st.secrets['gcp_service_account']")
    creds = Credentials.from_service_account_info(SVC_INFO, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def get_sheet():
    client = get_gs_client()
    if not SHEET_ID:
        raise RuntimeError("GSPREAD_SHEET_ID not set in secrets")
    return client.open_by_key(SHEET_ID)

HEADER_MAP = {
    "league_members": ["entry_id","entry_name","player_name","joined_at"],
    "gw_scores": ["entry_id","gw","points","live","updated_at"],
    "gw_rank": ["gw","entry_id","rank","points"],
    "gw_predictions": ["gw","entry_id","p_top1","p_top2","p_top3","updated_at"],
}

def _get_ws(title: str):
    sh = get_sheet()
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=20)
        # write headers if known
        headers = HEADER_MAP.get(title)
        if headers:
            ws.update([headers])
    return ws

def gs_read_df(title: str) -> pd.DataFrame:
    ws = _get_ws(title)
    data = ws.get_all_records()
    if not data:
        # ensure headers
        headers = ws.row_values(1)
        if not headers:
            headers = HEADER_MAP.get(title, [])
            if headers:
                ws.update([headers])
        return pd.DataFrame(columns=headers)
    return pd.DataFrame(data)

def gs_upsert(title: str, key_cols: List[str], rows: List[dict]):
    if not rows:
        return
    df_existing = gs_read_df(title)
    headers = list(HEADER_MAP.get(title, sorted(set().union(*[set(r.keys()) for r in rows]))))
    df_new = pd.DataFrame(rows)
    # add any missing columns
    for c in headers:
        if c not in df_existing.columns:
            df_existing[c] = np.nan
        if c not in df_new.columns:
            df_new[c] = np.nan
    if df_existing.empty:
        df_out = df_new[headers]
    else:
        # Upsert by key columns
        df_existing.set_index(key_cols, inplace=True, drop=False)
        df_new.set_index(key_cols, inplace=True, drop=False)
        df_existing.update(df_new)
        df_out = pd.concat([df_existing[~df_existing.index.isin(df_new.index)], df_new], axis=0)
        df_out = df_out[headers]
        df_out.reset_index(drop=True, inplace=True)
    # Write back (overwrite sheet)
    ws = _get_ws(title)
    ws.clear()
    ws.update([headers] + df_out.astype(object).fillna("").values.tolist())

# convenience selects
def gs_select(table: str, where: Dict[str, str] = None, select: List[str] = None) -> pd.DataFrame:
    df = gs_read_df(table)
    if where:
        for k, v in where.items():
            # support ops like eq.5, lte.10
            if isinstance(v, str) and "." in v:
                op, val = v.split(".", 1)
                try:
                    val_num = float(val)
                except:
                    val_num = val
                if op == "eq":
                    df = df[df[k] == (int(val_num) if isinstance(val_num, float) and val_num.is_integer() else val_num)]
                elif op == "lte":
                    df = df[pd.to_numeric(df[k], errors="coerce") <= float(val)]
                elif op == "gte":
                    df = df[pd.to_numeric(df[k], errors="coerce") >= float(val)]
                elif op == "lt":
                    df = df[pd.to_numeric(df[k], errors="coerce") < float(val)]
                elif op == "gt":
                    df = df[pd.to_numeric(df[k], errors="coerce") > float(val)]
            else:
                df = df[df[k] == v]
    if select:
        cols = [c for c in select if c in df.columns]
        df = df[cols]
    return df.reset_index(drop=True)

# =========================
# FPL API helpers
# =========================
SESSION = requests.Session()
BASE = "https://fantasy.premierleague.com/api"

@st.cache_data(ttl=180)
def get_bootstrap():
    return SESSION.get(f"{BASE}/bootstrap-static/").json()

@st.cache_data(ttl=180)
def get_current_event():
    data = get_bootstrap()
    events = data.get("events", [])
    for e in events:
        if e.get("is_current"):
            return e["id"], e.get("finished", False)
    done = [e for e in events if e.get("finished")] or []
    if done:
        return done[-1]["id"], True
    return None, True

# H2H members (pagination)
def get_h2h_members(league_id: int, page: int = 1):
    url = f"{BASE}/leagues-h2h/{league_id}/standings/?page_standings={page}"
    r = SESSION.get(url)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=180)
def fetch_all_members(league_id: int) -> List[dict]:
    page = 1
    rows = []
    while True:
        data = get_h2h_members(league_id, page)
        standings = data.get("standings", {})
        results = standings.get("results", [])
        if not results:
            break
        for r in results:
            rows.append({
                "entry_id": r["entry"],
                "entry_name": r["entry_name"],
                "player_name": r["player_name"],
                "joined_at": pd.Timestamp.utcnow().isoformat(),
            })
        if not standings.get("has_next"):
            break
        page += 1
    return rows

# Entry history → official points per finished GW
@st.cache_data(ttl=180)
def get_entry_history(entry_id: int):
    r = SESSION.get(f"{BASE}/entry/{entry_id}/history/")
    r.raise_for_status()
    return r.json()

# =========================
# Sync routines (Google Sheets)
# =========================
def sync_members_to_db(league_id: int) -> pd.DataFrame:
    members = fetch_all_members(league_id)
    if members:
        gs_upsert("league_members", ["entry_id"], members)
    return pd.DataFrame(members)

def sync_gw_points(gw: int, finished: bool, league_id: int):
    # read members (prefer DB, fallback to API)
    dfm = gs_read_df("league_members")
    if dfm.empty:
        dfm = pd.DataFrame(fetch_all_members(int(league_id)))
    rows = []
    for _, m in dfm.iterrows():
        entry_id = int(m["entry_id"])
        h = get_entry_history(entry_id)
        current = h.get("current", [])
        row = next((r for r in current if r.get("event") == gw), None)
        pts = int(row.get("points", 0)) if row else 0
        rows.append({
            "entry_id": entry_id,
            "gw": int(gw),
            "points": pts,
            "live": (not finished),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })
    if rows:
        gs_upsert("gw_scores", ["entry_id","gw"], rows)

def recompute_rank(gw: int) -> pd.DataFrame:
    df_scores = gs_select("gw_scores", where={"gw":"eq."+str(gw)})
    if df_scores.empty:
        return pd.DataFrame()
    df = df_scores.copy()
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    df["rank"] = df["points"].rank(method="min", ascending=False).astype(int)
    # save
    gs_upsert("gw_rank", ["gw","entry_id"], df[["gw","entry_id","rank","points"]].to_dict(orient="records"))
    # join names for display
    mems = gs_select("league_members")
    out = df.merge(mems, on="entry_id", how="left").sort_values(["rank","entry_name"])\
            [["rank","entry_name","player_name","points"]]
    return out

# Monte Carlo Top probabilities
def fit_mu_sigma(entry_id: int, upto_gw: int, window: int = 5) -> Tuple[float,float]:
    rows = gs_select("gw_scores", where={"entry_id":"eq."+str(entry_id), "gw":"lte."+str(upto_gw)})
    if rows.empty:
        return 45.0, 12.0
    df = rows.sort_values("gw").tail(window)
    mu = float(df["points"].astype(float).mean()) if not df.empty else 45.0
    sd = float(df["points"].astype(float).std(ddof=1)) if len(df) >= 2 else 12.0
    sd = max(sd, 8.0); mu = min(max(mu, 25.0), 85.0)
    return mu, sd

def simulate_top_probs(gw: int, n: int = 10000) -> pd.DataFrame:
    mems = gs_select("league_members")
    if mems.empty:
        return pd.DataFrame()
    ids = mems["entry_id"].astype(int).tolist()
    names = mems["entry_name"].tolist()
    mus, sds = [], []
    for eid in ids:
        mu, sd = fit_mu_sigma(eid, upto_gw=gw-1)
        mus.append(mu); sds.append(sd)
    M = len(ids)
    draws = np.random.randn(n, M) * np.array(sds) + np.array(mus)
    ranks = (-draws).argsort(axis=1).argsort(axis=1) + 1
    p1 = (ranks==1).mean(axis=0)
    p2 = (ranks<=2).mean(axis=0)
    p3 = (ranks<=3).mean(axis=0)
    rows = []
    for i in range(M):
        rows.append({
            "gw": int(gw),
            "entry_id": int(ids[i]),
            "p_top1": round(float(p1[i]),4),
            "p_top2": round(float(p2[i]),4),
            "p_top3": round(float(p3[i]),4),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })
    if rows:
        gs_upsert("gw_predictions", ["gw","entry_id"], rows)
    df = pd.DataFrame(rows); df["entry_name"] = names
    return df.sort_values("p_top1", ascending=False)

# =========================
# UI Controls
# =========================
col1, col2, col3 = st.columns([2,1,1])
league_id = col1.text_input("H2H League ID", value=str(FPL_LEAGUE_ID or ""))
current_gw, finished = get_current_event()
col2.metric("Current GW", current_gw or "-")
col3.metric("Finished?", "Yes" if finished else "No")

c1, c2, c3 = st.columns(3)
if c1.button("Sync members"):
    if league_id:
        with st.spinner("Đang đồng bộ danh sách đội..."):
            dfm = sync_members_to_db(int(league_id))
        st.success(f"Đã lưu {len(dfm)} đội vào Google Sheets.")

if c2.button("Sync points (current GW)"):
    if current_gw:
        with st.spinner(f"Cập nhật điểm GW{current_gw}..."):
            sync_gw_points(current_gw, finished, int(league_id or 0))
        st.success("Done!")

if c3.button("Recompute rank"):
    if current_gw:
        with st.spinner("Tính BXH..."):
            pass  # will recompute inside tab
        st.success("Done!")

st.divider()

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📊 BXH vòng", "📈 Dự đoán top%", "🧰 Dữ liệu"]) 

with tab1:
    if current_gw:
        df_rank = recompute_rank(current_gw)
        st.subheader(f"BXH theo điểm FPL — GW{current_gw}")
        if df_rank is None or df_rank.empty:
            st.info("Chưa có điểm vòng này. Bấm 'Sync points'.")
        else:
            st.dataframe(df_rank, use_container_width=True)

with tab2:
    if current_gw:
        if st.button("Run Monte Carlo (10k)"):
            with st.spinner("Đang mô phỏng..."):
                dfp = simulate_top_probs(current_gw)
                st.success("Done!")
        rows = gs_select("gw_predictions", where={"gw":"eq."+str(current_gw)})
        if not rows.empty:
            mems = gs_select("league_members")
            out = rows.merge(mems, on="entry_id", how="left")
            out = out[["entry_name","p_top1","p_top2","p_top3"]].sort_values("p_top1", ascending=False)
            st.subheader(f"Xác suất Top 1/2/3 — GW{current_gw}")
            st.dataframe(out, use_container_width=True)
        else:
            st.info("Chưa có kết quả mô phỏng.")

with tab3:
    st.write("League members (Sheet):")
    st.dataframe(gs_read_df("league_members"), use_container_width=True)
    if current_gw:
        st.write(f"GW{current_gw} scores:")
        st.dataframe(gs_select("gw_scores", where={"gw":"eq."+str(current_gw)}), use_container_width=True)

# =========================
# Notes / Future work
# =========================
# - For truly live points (ongoing GW), integrate picks API:
#   /entry/{id}/event/{gw}/picks + /event/{gw}/live to account for captain, autosubs, chips
# - Add H2H 3-1-0 standings by parsing fixtures in /leagues-h2h-matches/ endpoints.
# - Add historical backfill loop to populate past GWs automatically on first run.
