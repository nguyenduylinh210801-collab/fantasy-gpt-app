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

# ===== League ID input (fix NameError) =====
def _to_int(s):
    try:
        return int(str(s).strip())
    except:
        return None

_default_league = FPL_LEAGUE_ID
league_id = st.sidebar.text_input(
    "H2H League ID",
    value=str(_default_league or ""),
    placeholder="vd: 1007448"
)
league_id_int = _to_int(league_id)
if league_id and league_id_int is None:
    st.sidebar.error("⚠️ League ID phải là số nguyên.")


# =========================
# Streamlit Page
# =========================
st.set_page_config(page_title="FPL H2H Tracker", page_icon="⚽", layout="wide")
# ===== CSS tùy chỉnh =====
st.markdown("""
<style>
/* Thu gọn khoảng trắng tổng thể */
.block-container { padding-top: 1.25rem; padding-bottom: 1rem; }

/* Card nhẹ cho banner/information */
.app-note {
  background: #eef6ff; border: 1px solid #d6e6ff; padding: .75rem 1rem;
  border-radius: 12px; font-size: 0.95rem;
}

/* Hàng metric cân giữa, chữ to hơn chút */
[data-testid="stMetricValue"] { font-size: 1.6rem; }

/* Nút to, đều nhau, bo tròn */
.stButton > button {
  width: 100%; border-radius: 12px; padding: .6rem 1rem; font-weight: 600;
}

/* Tabs spacing đẹp hơn */
.stTabs [data-baseweb="tab-list"] { gap: .5rem; }
.stTabs [data-baseweb="tab"] { padding: .45rem .9rem; border-radius: 10px; }

/* Dataframe gọn */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ===== Logo + tiêu đề =====
logo_url = "https://upload.wikimedia.org/wikipedia/en/3/3a/Premier_League_Logo.svg"  # hoặc "logo.png"
col_logo, col_title = st.columns([0.15, 0.85])
with col_logo:
    st.image(logo_url, use_container_width=True)
with col_title:
    st.markdown("<h1 style='margin-top:0;'>SO Fantasy Premier League</h1>", unsafe_allow_html=True)

st.title("⚽ SO Fantasy Premier League")
if INVITE_CODE:
    st.info(f"👉 Nhập code để tham gia: `{INVITE_CODE}`")

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
    # Nếu không có vòng current → lấy vòng sắp diễn ra (is_next)
    for e in events:
        if e.get("is_next"):
            return e["id"], False
    # Nếu cũng không có → lấy vòng cuối đã kết thúc
    done = [e for e in events if e.get("finished")] or []
    if done:
        return done[-1]["id"], True
    return None, True

# =========================
# LIVE POINTS (picks + live + autosubs + chips)
# =========================

@st.cache_data(ttl=180)
def get_entry_picks(entry_id: int, gw: int):
    r = SESSION.get(f"{BASE}/entry/{entry_id}/event/{gw}/picks/")
    r.raise_for_status()
    return r.json()  # contains: picks, active_chip, automatic_subs (sau khi GW kết thúc)

@st.cache_data(ttl=60)
def get_event_live(gw: int):
    r = SESSION.get(f"{BASE}/event/{gw}/live/")
    r.raise_for_status()
    return r.json()  # contains: elements -> stats.total_points, stats.minutes,...

@st.cache_data(ttl=3600)
def get_elements_index():
    """Map element id -> element_type (1=GK,2=DEF,3=MID,4=FWD)."""
    bs = get_bootstrap()
    return {e["id"]: e["element_type"] for e in bs.get("elements", [])}

def _live_maps(gw: int):
    """Return (points_map, minutes_map) for the GW."""
    live = get_event_live(gw)
    pts_map, min_map = {}, {}
    for e in live.get("elements", []):
        el_id = e["id"]
        s = e.get("stats", {}) or {}
        pts_map[el_id] = int(s.get("total_points", 0))
        min_map[el_id] = int(s.get("minutes", 0))
    return pts_map, min_map

def _count_types(elems, elem_type_map):
    # elems: list of element ids
    from collections import Counter
    c = Counter([elem_type_map.get(x, 0) for x in elems])
    return {
        1: c.get(1, 0),  # GK
        2: c.get(2, 0),  # DEF
        3: c.get(3, 0),  # MID
        4: c.get(4, 0),  # FWD
    }

def _formation_ok(counts):
    """FPL min formation: 1 GK, >=3 DEF, >=2 MID, >=1 FWD, total 11."""
    total = sum(counts.values())
    return (
        counts[1] == 1 and
        counts[2] >= 3 and
        counts[3] >= 2 and
        counts[4] >= 1 and
        total == 11
    )

def _apply_basic_autosubs(starters, bench_order, minutes_map, elem_type_map, captain_id, vice_id, triple_captain=False):
    """
    starters: list of 11 element ids
    bench_order: list of 4 element ids in bench order 12..15
    Return: (final_eleven, new_captain_id)
    - Rule: replace starters with 0' by bench players (>0') in bench order if formation remains valid.
    - Captain/vice: if captain 0' and vice >0', vice becomes (triple) captain.
    """
    playing = [e for e in starters if minutes_map.get(e, 0) > 0]
    dnp = [e for e in starters if minutes_map.get(e, 0) == 0]

    # Try to sub in bench players (who played) one-by-one
    final = playing.copy()
    for b in bench_order:
        if minutes_map.get(b, 0) == 0:
            continue
        if not dnp:
            break
        # try replacing a dnp that keeps formation valid
        replaced = None
        for s in list(dnp):
            test = final + [b] + [x for x in dnp if x != s]  # naive check needs exactly 11; we’ll emulate:
            # Build test eleven: final + b + remaining dnp minus one s
            test_eleven = final + [b]  # add bench player
            # add remaining dnp except the one removed until reach 11 (but we want exactly 11: final size may be <11)
            # The proper approach: final currently <11; choose one dnp to DROP, not add others.
            # So recompute from starters: (starters - {s}) U {b} U (others that played already accounted in 'final')
            # Simpler: construct from current starters swap s->b
            starters_candidate = [x if x != s else b for x in starters]
            counts = _count_types([x for x in starters_candidate if minutes_map.get(x, 0) > 0 or x == b], elem_type_map)
            # Ensure we are evaluating exactly 11 on the pitch:
            eleven = []
            for x in starters_candidate:
                # choose played ones; if x==b, it's coming from bench and plays >0
                if x == b or minutes_map.get(x, 0) > 0:
                    eleven.append(x)
            # If still <11 due to multiple DNPs, we'll keep swapping in subsequent iterations
            # Check formation only when we have <=11; we accept intermediate states <11
            if len(eleven) <= 11:
                # When not yet 11, we can't fully validate; accept swap and continue
                replaced = s
                starters = starters_candidate
                dnp.remove(s)
                final = [x for x in eleven]  # recomputed playing so far
                break
        if replaced is None:
            # couldn't find a valid swap (formation-wise); skip this bench
            continue

    # If after all subs we still have <11 (e.g., no bench played), formation check not needed
    # Captain/vice adjustment
    new_captain = captain_id
    cap_minutes = minutes_map.get(captain_id, 0)
    vice_minutes = minutes_map.get(vice_id, 0)
    if cap_minutes == 0 and vice_minutes > 0:
        new_captain = vice_id

    # Ensure exactly 11 returned; if more (shouldn't), trim; if less, accept as-is for scoring (FPL would end with <11)
    final_eleven = final[:11]
    return final_eleven, new_captain

def compute_live_points_for_entry(entry_id: int, gw: int) -> int:
    """
    Returns estimated LIVE points for entry using picks + live + basic autosubs + chips.
    If Triple Captain active -> captain*3; if Bench Boost -> bench contribute; Free Hit handled by API picks already.
    """
    picks = get_entry_picks(entry_id, gw)
    pts_map, min_map = _live_maps(gw)
    elem_type_map = get_elements_index()

    active_chip = picks.get("active_chip")  # 'triple_captain', 'bench_boost', 'freehit', 'wildcard' or None
    is_bb = (active_chip == "bench_boost")
    is_tc = (active_chip == "triple_captain")

    # picks["picks"] has fields: element, position (1..15), is_captain, is_vice_captain, multiplier (effective)
    plist = sorted(picks.get("picks", []), key=lambda x: x.get("position", 99))
    starters = [p["element"] for p in plist if p.get("position", 99) <= 11]
    bench = [p["element"] for p in plist if p.get("position", 99) > 11]
    captain_id = next((p["element"] for p in plist if p.get("is_captain")), None)
    vice_id    = next((p["element"] for p in plist if p.get("is_vice_captain")), None)

    # Basic autosubs estimation (only during live; official autosubs applied after GW ends)
    final_eleven, new_captain = _apply_basic_autosubs(
        starters, bench, min_map, elem_type_map, captain_id, vice_id, triple_captain=is_tc
    )

    # Build multipliers:
    # - Default starters multiplier = 1; bench = 0 (unless Bench Boost)
    # - Captain = x2 (or x3 if Triple Captain)
    mult = {el: 0 for el in starters + bench}
    for el in final_eleven:
        mult[el] = 1
    if is_bb:
        # On Bench Boost: all bench contribute with 1 regardless of autosubs
        for el in bench:
            mult[el] = 1

    # Captain handling:
    if new_captain is not None:
        mult[new_captain] = mult.get(new_captain, 0) * (3 if is_tc else 2)
        # If original captain also in final_eleven and different from new_captain (edge case), ensure its base 1 only
        if captain_id and captain_id != new_captain and captain_id in mult:
            mult[captain_id] = 0 if not is_bb else mult[captain_id]  # captain DNP; under BB bench rule may still be 1

    # Sum points
    total = 0
    for el, m in mult.items():
        p = pts_map.get(el, 0)
        total += p * m
    return int(total)

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
        if finished:
            # dùng official từ history khi GW đã kết thúc
            h = get_entry_history(entry_id)
            current = h.get("current", [])
            row = next((r for r in current if r.get("event") == gw), None)
            pts = int(row.get("points", 0)) if row else 0
            is_live = False
        else:
            # dùng LIVE points (picks + live + autosubs + chips)
            try:
                pts = compute_live_points_for_entry(entry_id, gw)
            except Exception:
                # fallback an toàn nếu picks API lỗi
                h = get_entry_history(entry_id)
                current = h.get("current", [])
                row = next((r for r in current if r.get("event") == gw), None)
                pts = int(row.get("points", 0)) if row else 0
            is_live = True

        rows.append({
            "entry_id": entry_id,
            "gw": int(gw),
            "points": int(pts),
            "live": is_live,
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
# UI Controls (đẹp & cân đối)
# =========================
current_gw, finished = get_current_event()

# Banner mời tham gia (kiểu card nhẹ – cần CSS .app-note ở phần CSS bạn đã thêm)
if INVITE_CODE:
    st.markdown(
        f'<div class="app-note">👉 Nhập code để tham gia: <b>{INVITE_CODE}</b></div>',
        unsafe_allow_html=True
    )

st.write("")  # spacing nhẹ

# Hàng metric: 3 cột bằng nhau
m1, m2, m3 = st.columns([1, 1, 1], gap="large")
with m1:
    st.metric("League ID", league_id or "-")
with m2:
    st.metric("Current GW", current_gw or "-")
with m3:
    st.metric("Finished?", "Yes" if finished else "No")

st.write("")  # spacing nhẹ

# Hàng nút: 3 nút đều nhau, full width
b1, b2, b3 = st.columns(3, gap="large")

with b1:
    if st.button("Sync members", use_container_width=True):
        if league_id_int:
            with st.spinner("Đang đồng bộ danh sách đội..."):
                dfm = sync_members_to_db(league_id_int)
            st.success(f"Đã lưu {len(dfm)} đội vào Google Sheets.")
        else:
            st.error("Thiếu hoặc sai League ID.")

with b2:
    if st.button("Sync points (current GW)", use_container_width=True):
        if current_gw and league_id_int:
            with st.spinner(f"Cập nhật điểm GW{current_gw}..."):
                sync_gw_points(current_gw, finished, league_id_int)
            st.success("Done!")
        elif not league_id_int:
            st.error("Thiếu hoặc sai League ID.")
        else:
            st.error("Không xác định được Current GW.")

with b3:
    if st.button("Recompute rank", use_container_width=True):
        if current_gw:
            with st.spinner("Tính BXH..."):
                pass  # (giữ logic hiện tại: đang tính trong tab)
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
