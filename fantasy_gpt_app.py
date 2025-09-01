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

# ===== Vietnamese column labels (không đổi schema) =====
VN_LABELS = {
    "h2h_table": {
        "rank": "Hạng", "entry_name": "Tên đội",
        "Pld": "Trận", "W": "Thắng", "D": "Hòa", "L": "Thua",
        "GF": "Điểm tích lũy", "GA": "Điểm thủng", "GD": "Hiệu số",
        "P": "Điểm"
    },
    "h2h_results": {
        "gw": "GW", "entry_id": "ID đội", "opp_id": "Đối thủ",
        "gf": "Điểm ghi (GW)", "ga": "Điểm thủng (GW)", "pts": "Điểm (3–1–0)"
    },
    "gw_rank": {
        "gw": "GW", "entry_id": "ID đội",
        "rank": "Hạng vòng", "points": "Điểm FPL vòng"
    },
    "gw_scores": {
        "entry_id": "ID đội", "gw": "GW", "points": "Điểm",
        "live": "Live?", "chip": "Chip", "updated_at": "Cập nhật"
    },
    "gw_predictions": {
        "gw": "GW", "entry_id": "ID đội",
        "p_top1": "P Top1", "p_top2": "P Top2", "p_top3": "P Top3",
        "updated_at": "Cập nhật"
    },
    "league_members": {
        "entry_id": "ID đội", "entry_name": "Tên đội",
        "player_name": "Tên HLV", "joined_at": "Tham gia"
    },
}

def show_vn(df, kind: str):
    """
    Đổi nhãn cột sang tiếng Việt chỉ khi hiển thị.
    kind ∈ {'h2h_table','h2h_results','gw_rank','gw_scores','gw_predictions','league_members'}
    """
    if df is None or df.empty:
        return df
    mapping = VN_LABELS.get(kind, {})
    safe_map = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=safe_map)


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

st.title("⚽ SO Fantasy Premier League")


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
    try:
        if not SVC_INFO:
            raise RuntimeError("Missing st.secrets['gcp_service_account']")
        if not SHEET_ID:
            raise RuntimeError("Missing st.secrets['GSPREAD_SHEET_ID']")
        creds = Credentials.from_service_account_info(SVC_INFO, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID)
    except gspread.exceptions.APIError as e:
        sa_email = (SVC_INFO or {}).get("client_email", "")
        st.error(
            "Không truy cập được Google Sheet.\n\n"
            "• Thường do **chưa share** đúng service account (Editor) hoặc **SHEET_ID sai**.\n"
            f"• Service account đang dùng: `{sa_email}`\n"
            f"• Sheet ID: `{SHEET_ID}`\n"
            "→ Hãy kiểm tra rồi chạy lại."
        )
        st.stop()

HEADER_MAP = {
    "league_members": ["entry_id","entry_name","player_name","joined_at"],
    "gw_scores": ["entry_id","gw","points","live","chip","updated_at"],
    "gw_rank": ["gw","entry_id","rank","points"],
    "gw_predictions": ["gw","entry_id","p_top1","p_top2","p_top3","updated_at"],
    "h2h_results": ["gw","entry_id","opp_id","gf","ga","pts"],
    "h2h_table": ["entry_id","entry_name","Pld","W","D","L","GF","GA","GD","P","rank"],
}

def _get_ws(title: str):
    sh = get_sheet()
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        try:
            ws = sh.add_worksheet(title=title, rows=1000, cols=20)
            headers = HEADER_MAP.get(title)
            if headers:
                ws.update([headers])
            return ws
        except Exception:
            st.error(
                f"Không tạo được worksheet '{title}'. "
                "Hãy kiểm tra: GSPREAD_SHEET_ID đúng chưa, và service account có quyền Editor trên Google Sheet chưa."
            )
            raise  # để gs_read_df/gs_upsert bắt và fail mềm

def gs_read_df(title: str) -> pd.DataFrame:
    """
    Đọc dữ liệu từ Google Sheets, nếu lỗi trả DF rỗng với cột từ HEADER_MAP (UI không sập).
    """
    try:
        ws = _get_ws(title)
    except Exception as e:
        st.error(f"❌ Lỗi mở worksheet '{title}': {e}")
        return pd.DataFrame(columns=HEADER_MAP.get(title, []))

    try:
        data = ws.get_all_records()
    except Exception as e:
        st.error(f"❌ Lỗi đọc worksheet '{title}': {e}")
        return pd.DataFrame(columns=HEADER_MAP.get(title, []))

    if not data:
        # đảm bảo có header
        try:
            headers = ws.row_values(1)
        except Exception:
            headers = HEADER_MAP.get(title, [])
        if not headers:
            headers = HEADER_MAP.get(title, [])
            if headers:
                try:
                    ws.update([headers])
                except Exception:
                    pass
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
    try:
        ws = _get_ws(title)
        ws.clear()
        ws.update([headers] + df_out.astype(object).fillna("").values.tolist())
    except Exception as e:
        st.error(f"❌ Lỗi ghi '{title}' lên Google Sheets: {e}")

def gs_select(table: str, where: Dict[str, str] = None, select: List[str] = None) -> pd.DataFrame:
    df = gs_read_df(table)
    if where is not None and not df.empty:
        for k, v in where.items():
            # hỗ trợ eq./lte./gte./lt./gt.
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

# Sidebar controls
st.sidebar.header("⚙️ Cài đặt")

if st.sidebar.button("Test Google Sheets"):
    try:
        sh = get_sheet()
        wss = [ws.title for ws in sh.worksheets()]
        st.sidebar.success(f"Kết nối OK. Worksheets: {wss}")
    except Exception as e:
        st.sidebar.error(f"Lỗi GS: {e}")

# === Sidebar: Admin tools ===
with st.sidebar.expander("🔧 Admin tools", expanded=True):
    sb_sync_members = st.button("Sync members", use_container_width=True)
    sb_sync_points  = st.button("Sync points (current GW)", use_container_width=True)
    sb_recompute    = st.button("Recompute rank", use_container_width=True)


# =========================
# FPL API helpers
# =========================
SESSION = requests.Session()
BASE = "https://fantasy.premierleague.com/api"

# === PATCH 1: Official detector & Active GW resolver ===
@st.cache_data(ttl=120)
def is_event_official(gw: int) -> bool:
    """Official khi BOTH finished & data_checked."""
    try:
        r = SESSION.get(f"{BASE}/event/{gw}/", timeout=15)
        if r.status_code == 200:
            d = r.json()
            return bool(d.get("finished")) and bool(d.get("data_checked"))
    except:
        pass
    return False

@st.cache_data(ttl=120)
def get_active_gw() -> tuple[int | None, bool]:
    """
    Trả về (active_gw, is_official):
    - Còn 'is_current' -> active_gw = current.id (is_official phụ thuộc data_checked).
    - Không còn 'is_current' -> active_gw = is_next.id - 1 (vòng vừa xong, thường official).
    - Cuối mùa -> GW lớn nhất.
    """
    bs = get_bootstrap()
    events = bs.get("events", []) or []
    cur = next((e for e in events if e.get("is_current")), None)
    if cur:
        gw = int(cur["id"])
        return gw, is_event_official(gw)

    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        prev_id = int(nxt["id"]) - 1
        if prev_id >= 1:
            return prev_id, is_event_official(prev_id)
        # Pre-season trước GW1
        return int(nxt["id"]), False

    if events:
        last = max(events, key=lambda x: int(x["id"]))
        gw = int(last["id"])
        return gw, is_event_official(gw)
    return None, False

@st.cache_data(ttl=180)
def is_event_official_relaxed(gw: int) -> bool:
    """
    OFFICIAL nếu:
      - gw < current_gw  (đã qua vòng hiện tại, coi như đã chốt),
      - hoặc (finished & data_checked) = True theo FPL.
    """
    bs = get_bootstrap()
    events = bs.get("events", []) or []

    cur = next((e for e in events if e.get("is_current")), None)
    cur_id = int(cur["id"]) if cur and str(cur.get("id", "")).isdigit() else None

    # Mọi GW đã qua vòng hiện tại → coi như official
    if cur_id and gw < cur_id:
        return True

    # Nếu không có current (pre/đầu mùa) thì rơi về finished + data_checked
    ev = next((e for e in events if int(e.get("id", 0)) == int(gw)), None)
    if ev:
        return bool(ev.get("finished")) and bool(ev.get("data_checked"))
    return False

@st.cache_data(ttl=180)
def get_h2h_matches_page(league_id: int, gw: int, page: int = 1):
    url = f"{BASE}/leagues-h2h-matches/league/{league_id}/?page={page}&event={gw}"
    r = SESSION.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_h2h_matches(league_id: int, gw: int):
    out, page = [], 1
    while True:
        data = get_h2h_matches_page(league_id, gw, page)
        results = (data or {}).get("results", []) or []
        if not results:
            break
        out.extend(results)
        if not data.get("has_next"):
            break
        page += 1
    return out


@st.cache_data(ttl=180)
def get_bootstrap():
    return SESSION.get(f"{BASE}/bootstrap-static/").json()



@st.cache_data(ttl=180)
def get_event_by_id(event_id: int) -> dict:
    if not event_id:
        return {}
    data = get_bootstrap()
    for e in data.get("events", []):
        if e.get("id") == event_id:
            return e
    return {}

def _fmt_time_local(iso_str: str, tz: str = "Asia/Ho_Chi_Minh") -> str:
    """
    Đổi ISO (UTC) -> giờ địa phương, luôn hiển thị nhãn ICT thay vì +07.
    """
    if not iso_str:
        return ""
    try:
        dt = pd.to_datetime(iso_str, utc=True).tz_convert(tz)
        # Ép nhãn ICT cho nhất quán
        return dt.strftime("%d %b %Y, %H:%M") + " ICT"
    except Exception:
        return iso_str or ""


@st.cache_data(ttl=180)
def get_event_times(event_id: int) -> tuple[str, str, str]:
    """
    Trả về (gw_name, start_time_local, deadline_local).
    - start_time = deadline_time của chính GW.
    - deadline_local = deadline_time của GW tiếp theo.
    """
    data = get_bootstrap()
    events = data.get("events", [])

    ev = next((e for e in events if e.get("id") == event_id), None)
    if not ev:
        return ("", "", "")

    gw_name = ev.get("name", "")
    start_local = _fmt_time_local(ev.get("deadline_time"))

    next_ev = next((e for e in events if e.get("id") == event_id + 1), None)
    deadline_local = _fmt_time_local(next_ev.get("deadline_time")) if next_ev else ""

    return (gw_name, start_local, deadline_local)

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

def _apply_basic_autosubs(
    starters, bench_order, minutes_map, elem_type_map,
    captain_id, vice_id, triple_captain=False
):
    """
    Quy tắc FPL:
      - Chỉ thay cầu thủ DNP (minutes==0).
      - GK (type=1) chỉ thay GK; outfield (2/3/4) chỉ thay outfield.
      - Duyệt bench theo đúng thứ tự 12→15; chỉ nhận cầu thủ bench có phút >0.
      - Mỗi swap phải giữ được formation tối thiểu: 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD; tổng 11 khi có đủ người.
      - Chuyển C→VC chỉ khi captain **DNP** (minutes==0). Captain 0 điểm nhưng **có phút** ⇒ KHÔNG chuyển.
    """
    def played(x): return minutes_map.get(x, 0) > 0
    def typ(x):    return elem_type_map.get(x, 0)

    cur = starters[:]
    dnp = [x for x in starters if not played(x)]

    for b in bench_order:
        if not played(b):
            continue
        bt = typ(b)

        if bt == 1:
            # GK bench chỉ thay GK DNP xuất phát
            cand = next((x for x in dnp if typ(x) == 1), None)
            if cand is None:
                continue
            cur = [b if x == cand else x for x in cur]
            dnp.remove(cand)
            continue

        # outfield bench: thử thay một outfield DNP sao cho vẫn hợp lệ
        of_dnp = [x for x in dnp if typ(x) in (2, 3, 4)]
        replaced = None
        for s in of_dnp:
            test = [b if x == s else x for x in cur]

            # On-pitch = những ai có phút >0 sau khi swap
            on_pitch = [x for x in test if played(x)]
            # b chắc chắn >0' (đã check), đảm bảo có trong on_pitch
            if b not in on_pitch:
                on_pitch.append(b)

            # Đếm formation tối thiểu
            counts = _count_types(on_pitch[:11], elem_type_map)
            ok_roles = (
                counts[1] == 1 and
                counts[2] >= 3 and
                counts[3] >= 2 and
                counts[4] >= 1
            )
            if ok_roles:
                cur = test
                dnp.remove(s)
                replaced = s
                break
        if replaced is None:
            continue

    # Chốt 11 người cuối cùng: chỉ ai có phút >0
    final_eleven = [x for x in cur if played(x)][:11]

    # Captain/Vice
    new_captain = captain_id
    cap_dnp = (minutes_map.get(captain_id, 0) == 0) if captain_id else False
    vice_ok  = (minutes_map.get(vice_id, 0) > 0) if vice_id else False
    if cap_dnp and vice_ok:
        new_captain = vice_id

    return final_eleven, new_captain



def get_final_or_live_points(entry_id: int, gw: int) -> int:
    """Ưu tiên official nếu có; nếu chưa có thì lấy live; trả về 0 nếu chưa có gì."""
    return int(get_points_map_for_gw(gw).get(entry_id, 0))



def compute_live_points_for_entry(entry_id: int, gw: int, active_chip: str | None = None) -> int:
    picks = get_entry_picks(entry_id, gw)  # chứa entry_history.event_transfers_cost
    pts_map, min_map = _live_maps(gw)
    elem_type_map = get_elements_index()

    # ✅ Ưu tiên chip từ API, rồi mới đến tham số; xử lý luôn rỗng/None
    chip_from_api = (picks.get("active_chip") or "").strip().lower()
    active_chip   = (chip_from_api or (active_chip or "")).lower()

    is_bb = (active_chip in ["bench_boost", "bboost", "bb"])
    is_tc = (active_chip in ["triple_captain", "3xc", "tc"])


    plist = sorted(picks.get("picks", []), key=lambda x: x.get("position", 99))
    starters   = [p["element"] for p in plist if p.get("position", 99) <= 11]
    bench      = [p["element"] for p in plist if p.get("position", 99) > 11]
    captain_id = next((p["element"] for p in plist if p.get("is_captain")), None)
    vice_id    = next((p["element"] for p in plist if p.get("is_vice_captain")), None)

    # ✅ Chỉ autosub & chuyển C→VC khi GW đã official
    if is_event_official(int(gw)):
        final_eleven, new_captain = _apply_basic_autosubs(
            starters, bench, min_map, elem_type_map, captain_id, vice_id,
            triple_captain=is_tc
        )
    else:
        final_eleven = starters[:]   # live: giữ nguyên 11 xuất phát
        new_captain  = captain_id    # live: không đổi captain

    # --- TÍNH ĐIỂM ---
    total = sum(pts_map.get(el, 0) for el in final_eleven)

    # Bench Boost: cộng thêm bench
    if is_bb:
        total += sum(pts_map.get(el, 0) for el in bench)

    def in_counting(x):
        if x is None:
            return False
        return (x in final_eleven) or (is_bb and x in bench)

    # Captain multiplier: cộng thêm cho captain
    if new_captain and in_counting(new_captain):
        base = pts_map.get(new_captain, 0)
        total += base if not is_tc else base * 2

    # Trừ điểm phạt chuyển nhượng (hits)
    cost = int((picks.get("entry_history") or {}).get("event_transfers_cost", 0) or 0)
    total -= cost

    return int(total)

def persist_final_gw_scores(entry_ids: list[int], gw: int):
    rows = []
    for entry_id in entry_ids:
        picks = get_entry_picks(entry_id, gw)
        active_chip = picks.get("active_chip", "")
        pts = compute_live_points_for_entry(entry_id, gw, active_chip=active_chip)
        rows.append({
            "entry_id": entry_id,
            "gw": gw,
            "points": pts,
            "live": False,
            "chip": active_chip,
            "updated_at": pd.Timestamp.utcnow().isoformat()
        })
    gs_upsert("gw_scores", ["entry_id", "gw"], rows)


# ✅ SỬA BXH:
# ❌ Sai: dùng gw_scores[entry_id]["points"]
# ✅ Đúng:

def build_rankings(entry_ids: list[int], gw: int) -> list[dict]:
    mems = gs_select("league_members")
    name_map = dict(zip(mems["entry_id"].astype(int), mems["entry_name"])) if not mems.empty else {}

    chip_map = {}
    df_gw = gs_select("gw_scores", where={"gw": "eq."+str(gw)})
    if not df_gw.empty:
        for _, r in df_gw.iterrows():
            try:
                chip_map[int(r["entry_id"])] = r.get("chip", "") or ""
            except:
                pass

    rows = []
    for entry_id in entry_ids:
        rows.append({
            "entry": entry_id,
            "entry_name": name_map.get(entry_id, str(entry_id)),
            "points": get_final_or_live_points(entry_id, gw),
            "chip": chip_map.get(entry_id, "")
        })

    rows.sort(key=lambda x: x["points"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows



# H2H members (pagination)
def get_h2h_members(league_id: int, page: int = 1):
    url = f"{BASE}/leagues-h2h/{league_id}/standings/?page_standings={page}"
    r = SESSION.get(url)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=180)
def fetch_all_members(league_id: int) -> List[dict]:
    """
    Lấy member cho H2H league.
    - Trong preseason/GW1 chưa diễn ra: standings có thể rỗng -> fallback sang new_entries.results
    """
    page = 1
    rows: List[dict] = []
    while True:
        data = get_h2h_members(league_id, page)
        standings = (data or {}).get("standings", {}) or {}
        s_results = standings.get("results", []) or []
        # Fallback: trước GW1, danh sách nằm ở new_entries.results
        if not s_results:
            new_entries = (data or {}).get("new_entries", {}) or {}
            s_results = new_entries.get("results", []) or []

        if not s_results:
            break

        for r in s_results:
            # r có thể có player_name, hoặc player_first_name/last_name
            player_name = r.get("player_name") \
                or " ".join([str(r.get("player_first_name", "")).strip(),
                             str(r.get("player_last_name", "")).strip()]).strip()
            rows.append({
                "entry_id": r.get("entry"),
                "entry_name": r.get("entry_name"),
                "player_name": player_name,
                "joined_at": pd.Timestamp.utcnow().isoformat(),
            })

        # phân trang theo standings nếu có, nếu không thì theo new_entries
        has_next = standings.get("has_next")
        if has_next:
            page += 1
            continue

        new_entries = (data or {}).get("new_entries", {}) or {}
        if new_entries.get("has_next"):
            page += 1
            continue

        break

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

# === PATCH 2: Wrapper sync theo trạng thái official của GW ===
def sync_gw_points_for(gw: int, league_id: int):
    """Nếu GW<current ⇒ coi như official; ngược lại check finished & data_checked."""
    finished = is_event_official_relaxed(int(gw))
    sync_gw_points(int(gw), finished, int(league_id))



def sync_gw_points(gw: int, finished: bool, league_id: int):
    # read members (prefer DB, fallback to API)
    dfm = gs_read_df("league_members")
    if dfm.empty:
        dfm = pd.DataFrame(fetch_all_members(int(league_id)))

    rows = []
    for _, m in dfm.iterrows():
        entry_id = int(m["entry_id"])
        chip = ""  # ✅ luôn khởi tạo

        if finished:
            # Dùng official từ history
            h = get_entry_history(entry_id)
            current = h.get("current", [])
            row = next((r for r in current if r.get("event") == gw), None)
            pts = int(row.get("points", 0)) if row else 0
            is_live = False
        else:
            # LIVE points
            try:
                picks = get_entry_picks(entry_id, gw)
                chip = picks.get("active_chip", "") or ""
                pts = compute_live_points_for_entry(entry_id, gw, active_chip=chip)
            except Exception:
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
            "chip": chip,  # ✅ giờ luôn tồn tại
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })


    if rows:
        gs_upsert("gw_scores", ["entry_id","gw"], rows)

def get_points_map_for_gw(gw: int) -> dict[int, int]:
    df = gs_select("gw_scores", where={"gw": "eq."+str(gw)})
    if df.empty:
        return {}
    df = df.copy()
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)

    # Chuẩn hoá 'live' về bool
    if "live" in df.columns:
        df["live_flag"] = (
            df["live"].astype(str).str.strip().str.lower()
              .map({"true": True, "false": False})
              .fillna(False)
        )
    else:
        df["live_flag"] = True  # thận trọng

    best = {}
    for eid, g in df.groupby("entry_id"):
        g = g.sort_values(["live_flag", "points"], ascending=[True, False])  # official trước
        row = g.iloc[0]
        best[int(eid)] = int(row["points"])
    return best

def compute_h2h_results_for_gw(league_id: int, gw: int) -> pd.DataFrame:
    pts_map = get_points_map_for_gw(gw)
    if not pts_map:
        st.warning(f"Chưa có gw_scores cho GW {gw}. Hãy bấm 'Sync points'.")
        return pd.DataFrame()

    matches = fetch_h2h_matches(int(league_id), int(gw))
    if not matches:
        st.info(f"Không tìm thấy cặp đấu H2H cho GW {gw}.")
        return pd.DataFrame()

    rows = []
    for m in matches:
        e1 = int(m["entry_1_entry"])
        e2 = int(m["entry_2_entry"])
        p1 = int(pts_map.get(e1, 0))
        p2 = int(pts_map.get(e2, 0))
        if   p1 > p2: r1, r2 = 3, 0
        elif p1 < p2: r1, r2 = 0, 3
        else:         r1, r2 = 1, 1
        rows += [
            {"gw": gw, "entry_id": e1, "opp_id": e2, "gf": p1, "ga": p2, "pts": r1},
            {"gw": gw, "entry_id": e2, "opp_id": e1, "gf": p2, "ga": p1, "pts": r2},
        ]
    df = pd.DataFrame(rows)
    if not df.empty:
        gs_upsert("h2h_results", ["gw","entry_id"], df.to_dict(orient="records"))
    return df

def build_h2h_table(upto_gw: int) -> pd.DataFrame:
    df = gs_select("h2h_results", where={"gw": "lte."+str(upto_gw)})
    if df.empty:
        return pd.DataFrame()

    df["win"]  = (df["pts"] == 3).astype(int)
    df["draw"] = (df["pts"] == 1).astype(int)
    df["loss"] = (df["pts"] == 0).astype(int)

    agg = df.groupby("entry_id").agg(
        P=("pts","sum"),
        Pld=("gw","count"),
        W=("win","sum"),
        D=("draw","sum"),
        L=("loss","sum"),
        GF=("gf","sum"),
        GA=("ga","sum"),
    ).reset_index()
    agg["GD"] = agg["GF"] - agg["GA"]

    # Join tên đội
    mems = gs_select("league_members")
    if not mems.empty:
        agg = agg.merge(mems[["entry_id","entry_name"]], on="entry_id", how="left")
    else:
        agg["entry_name"] = agg["entry_id"].astype(str)

    # Tie-breaker: P → GD → GF (KHÔNG có mini-league H2H)
    agg = agg.sort_values(["P", "GF"], ascending=[False, False]).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg)+1)

    # Lưu bảng để UI lần sau đọc nhanh
    gs_upsert("h2h_table", ["entry_id"], agg.to_dict(orient="records"))

    return agg[["rank","entry_name","P","GF","W","D","L",]]

def build_h2h_table_range(gw_from: int, gw_to: int) -> pd.DataFrame:
    df = gs_select("h2h_results", where={
        "gw": "gte." + str(gw_from)
    })  # lọc từ gw_from

    df = df[df["gw"] <= gw_to]  # lọc thêm gw_to nếu cần

    if df.empty:
        return pd.DataFrame()

    df["win"]  = (df["pts"] == 3).astype(int)
    df["draw"] = (df["pts"] == 1).astype(int)
    df["loss"] = (df["pts"] == 0).astype(int)

    agg = df.groupby("entry_id").agg(
        P=("pts","sum"),
        Pld=("gw","count"),
        W=("win","sum"),
        D=("draw","sum"),
        L=("loss","sum"),
        GF=("gf","sum"),
        GA=("ga","sum"),
    ).reset_index()
    agg["GD"] = agg["GF"] - agg["GA"]

    mems = gs_select("league_members")
    if not mems.empty:
        agg = agg.merge(mems[["entry_id","entry_name"]], on="entry_id", how="left")
    else:
        agg["entry_name"] = agg["entry_id"].astype(str)

    agg = agg.sort_values(["P","GF"], ascending=[False, False]).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg)+1)

    gs_upsert("h2h_table", ["entry_id"], agg.to_dict(orient="records"))

    return agg[["rank","entry_name","P","GF","W","D","L"]]

def build_h2h_results_view(league_id: int, gw: int) -> pd.DataFrame:
    """
    Tạo bảng 'KẾT QUẢ' cho 1 GW: mỗi dòng là một cặp (Nhóm A vs Nhóm B, điểm).
    Ưu tiên dùng official points nếu đã 'ghi điểm', nếu chưa thì dùng live.
    """
    # điểm mỗi đội trong GW
    pts_map = get_points_map_for_gw(gw)
    if not pts_map:
        return pd.DataFrame()

    # tên đội
    mems = gs_select("league_members")[["entry_id", "entry_name"]]
    name_map = dict(zip(mems["entry_id"].astype(int), mems["entry_name"]))

    # danh sách cặp đấu từ API
    matches = fetch_h2h_matches(int(league_id), int(gw))
    if not matches:
        return pd.DataFrame()

    rows = []
    for m in matches:
        a = int(m["entry_1_entry"])
        b = int(m["entry_2_entry"])
        pa = int(pts_map.get(a, 0))
        pb = int(pts_map.get(b, 0))
        rows.append({
            "Vòng": gw,
            "Nhóm A": name_map.get(a, str(a)),
            "": f"{pa}  —  {pb}",             # cột điểm ở giữa
            "Nhóm B": name_map.get(b, str(b)),
            "_pa": pa, "_pb": pb              # cột phụ để sort/hilite (không hiển thị)
        })
    df = pd.DataFrame(rows)
    # sắp xếp để trận có điểm cao nổi bật (tuỳ ý)
    return df.sort_values(["_pa", "_pb"], ascending=False).drop(columns=["_pa","_pb"]).reset_index(drop=True)


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
        mus.append(mu)
        sds.append(sd)

    M = len(ids)
    draws = np.random.randn(n, M) * np.array(sds) + np.array(mus)

    ranks = (-draws).argsort(axis=1).argsort(axis=1) + 1
    # → chuyển sang % luôn (0–100), làm tròn 2 số
    p1 = (ranks == 1).mean(axis=0) * 100
    p2 = (ranks <= 2).mean(axis=0) * 100
    p3 = (ranks <= 3).mean(axis=0) * 100

    rows = []
    for i in range(M):
        rows.append({
            "gw": int(gw),
            "entry_id": int(ids[i]),
            "p_top1": round(float(p1[i]), 2),
            "p_top2": round(float(p2[i]), 2),
            "p_top3": round(float(p3[i]), 2),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })

    if rows:
        # Lưu % vào Google Sheets để lần sau đọc lên hiển thị đúng luôn
        gs_upsert("gw_predictions", ["gw", "entry_id"], rows)

    df = pd.DataFrame(rows)
    df["entry_name"] = names
    return df.sort_values("p_top1", ascending=False)

# =========================
# =========================
# UI Controls (đẹp & cân đối)
# =========================
current_gw, finished = get_active_gw()
gw_name, gw_start, gw_deadline = get_event_times(current_gw) if current_gw else ("", "", "")

# --- Auto-run once defaults ---
if "did_first_autorun" not in st.session_state:
    st.session_state.did_first_autorun = False

# Set mặc định cho 3 input trong form (nếu chưa có)
if "gw_from" not in st.session_state:
    st.session_state.gw_from = 1
if "gw_to" not in st.session_state:
    st.session_state.gw_to = int(current_gw or 1)
if "gw_result" not in st.session_state:
    st.session_state.gw_result = int(current_gw or 1)

# (optional) chỉ auto-sync official 1 lần cho mỗi session
if "did_official_autosync" not in st.session_state:
    st.session_state.did_official_autosync = False

# === PATCH: Auto-sync tất cả GW đã official (finished & data_checked) — chạy 1 lần mỗi session
if league_id_int and not st.session_state.did_official_autosync:
    try:
        bs = get_bootstrap()
        events = bs.get("events", []) or []
        finished_official_gws = [int(e["id"]) for e in events if e.get("finished") and e.get("data_checked")]
        for gw in finished_official_gws:
            try:
                # Nếu GW đã official, hàm này sẽ ghi đè live bằng official
                sync_gw_points_for(int(gw), int(league_id_int))
            except Exception as err:
                st.sidebar.info(f"Không thể sync official cho GW{gw}: {err}")
        st.session_state.did_official_autosync = True
    except Exception as e:
        st.sidebar.info(f"Auto-sync official error: {e}")

# === FORCE REFRESH & RESYNC OFFICIAL FOR ALL PAST GWs (khi bạn thấy FPL vừa data_checked) ===
with st.sidebar.expander("♻️ Refresh official points", expanded=False):
    if st.button("Clear caches & resync 1..current"):
        try:
            # 1) clear cache để lấy trạng thái mới nhất từ FPL
            is_event_official.clear()
            get_bootstrap.clear()
            get_entry_history.clear()
            get_event_live.clear()
            get_entry_picks.clear()
            st.sidebar.success("Caches cleared.")
        except Exception as e:
            st.sidebar.info(f"Cache clear error: {e}")

        # 2) gọi sync lại cho toàn bộ 1..current_gw
        if league_id_int and current_gw:
            for g in range(1, int(current_gw) + 1):
                try:
                    sync_gw_points_for(int(g), int(league_id_int))
                except Exception as e:
                    st.sidebar.info(f"Sync GW{g} error: {e}")
            st.sidebar.success("Resynced all GWs 1..current.")

with st.sidebar.expander("♻️ Refresh official points", expanded=False):
    if st.button("Clear caches & resync 1..current"):
        try:
            # clear toàn bộ caches liên quan official
            is_event_official.clear()
            is_event_official_relaxed.clear()
            get_bootstrap.clear()
            get_entry_history.clear()
            get_event_live.clear()
            get_entry_picks.clear()
            st.sidebar.success("Caches cleared.")
        except Exception as e:
            st.sidebar.info(f"Cache clear error: {e}")

        # resync lại toàn bộ từ 1..current_gw
        if league_id_int and current_gw:
            for g in range(1, int(current_gw) + 1):
                try:
                    sync_gw_points_for(int(g), int(league_id_int))
                except Exception as e:
                    st.sidebar.info(f"Sync GW{g} error: {e}")
            st.sidebar.success("Resynced all GWs 1..current.")

# Hành động cho các nút ở sidebar
if sb_sync_members:
    if league_id_int:
        with st.spinner("Đang đồng bộ danh sách đội..."):
            dfm = sync_members_to_db(league_id_int)
        st.sidebar.success(f"Đã lưu {len(dfm)} đội vào Google Sheets.")
    else:
        st.sidebar.error("Thiếu hoặc sai League ID.")

# Nếu GW đang quan tâm đã official, đảm bảo DB có official (ghi đè live nếu cần)
if current_gw and league_id_int and is_event_official(int(current_gw)):
    try:
        sync_gw_points_for(int(current_gw), int(league_id_int))
    except Exception as e:
        st.sidebar.info(f"Không thể auto-sync official cho GW{current_gw}: {e}")

if sb_sync_points:
    if current_gw and league_id_int:
        with st.spinner(f"Cập nhật điểm GW{current_gw} (official nếu đã có)..."):
            sync_gw_points_for(int(current_gw), int(league_id_int))
        st.sidebar.success("Done!")
    elif not league_id_int:
        st.sidebar.error("Thiếu hoặc sai League ID.")
    else:
        st.sidebar.error("Không xác định được GW.")

if sb_recompute:
    if current_gw:
        with st.spinner("Tính BXH..."):
            pass
        st.sidebar.success("Done!")


# Banner mời tham gia (kiểu card nhẹ – cần CSS .app-note ở phần CSS bạn đã thêm)
if INVITE_CODE:
    st.markdown(
        f'<div class="app-note">👉 Nhập code để tham gia: <b>{INVITE_CODE}</b></div>',
        unsafe_allow_html=True
    )

st.write("")  # spacing nhẹ

# Hàng metric: 2 cột (ẩn League ID)
m_left, m_right = st.columns([2, 1], gap="large")

with m_left:
    st.metric("Current GW", f"{current_gw or '-'}")
#
#
with m_right:
    st.metric("Finished?", "Yes" if finished else "No")

st.write("")  

# =========================
# Tab layout
# =========================
tab1, tab2 = st.tabs(["🏆 Bảng xếp hạng", "📈 Dự đoán"])

with tab1:
    if not league_id_int:
        st.warning("Hãy nhập đúng H2H League ID ở sidebar.")
    else:
        # ==== Giao diện gọn trong 1 form ====
        with st.form("h2h_form", clear_on_submit=False, border=False):
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
            with col1:
                gw_from = st.number_input(
                    "Từ GW",
                    min_value=1,
                    value=int(st.session_state.gw_from),
                    step=1,
                    key="gw_from"
                )
            with col2:
                gw_to = st.number_input(
                    "Đến GW",
                    min_value=gw_from,
                    value=int(st.session_state.gw_to),
                    step=1,
                    key="gw_to"
                )
            with col3:
                gw_result = st.number_input(
                    "GW hiển thị kết quả",
                    min_value=1,
                    value=int(st.session_state.gw_result),
                    step=1,
                    key="gw_result"
                )
            with col4:
                st.markdown("### &nbsp;", unsafe_allow_html=True)
                do_both = st.form_submit_button("⚡ Cập nhật ", type="primary")

        # ✅ Chạy khi bấm nút, HOẶC tự động chạy 1 lần khi mới mở trang
        should_run_now = bool(do_both) or (not st.session_state.did_first_autorun and league_id_int)

        if should_run_now:
            # Đánh dấu đã autorun để lần sau không chạy lại
            st.session_state.did_first_autorun = True

            # 0) Đảm bảo có members
            if gs_read_df("league_members").empty and league_id_int:
                sync_members_to_db(int(league_id_int))

            # 1) Sync điểm cho dải BXH và GW kết quả (official nếu có)
            gws_need = list(range(int(gw_from), int(gw_to) + 1))
            if int(gw_result) not in gws_need:
                gws_need.append(int(gw_result))
            for g in gws_need:
                sync_gw_points_for(int(g), int(league_id_int))

            # 2) Tạo bảng kết quả & BXH
            compute_h2h_results_for_gw(int(league_id_int), int(gw_result))

            col_left, col_right = st.columns(2)

            # === BXH ===
            tbl = build_h2h_table_range(gw_from, gw_to)
            if tbl is not None and not tbl.empty:
                col_left.markdown(f"### 📊 BẢNG XẾP HẠNG ({gw_from} → {gw_to})")
                tbl_vn = show_vn(tbl, "h2h_table").reset_index(drop=True)
                col_left.dataframe(
                    tbl_vn[["Hạng", "Tên đội", "Điểm", "Điểm tích lũy", "Thắng", "Hòa", "Thua"]].set_index("Hạng"),
                    use_container_width=True
                )
            else:
                col_left.info("Chưa có dữ liệu BXH.")

            # === KẾT QUẢ ===
            df_res = build_h2h_results_view(league_id_int, gw_result)
            if df_res is not None and not df_res.empty:
                col_right.markdown(f"### 📝 KẾT QUẢ — GW {gw_result}")
                col_right.dataframe(
                    df_res.rename(columns={"": "Tỷ số"}).set_index("Vòng"),
                    use_container_width=True
                )
            else:
                col_right.info(f"Không có dữ liệu kết quả cho GW {gw_result}.")



with tab2:
    if current_gw:
        if st.button("Run Monte Carlo (10k)"):
            with st.spinner("Đang mô phỏng..."):
                dfp = simulate_top_probs(current_gw)
                st.success("Done!")
        rows = gs_select("gw_predictions", where={"gw": "eq." + str(current_gw)})
        if not rows.empty:
            mems = gs_select("league_members")
            out = rows.merge(mems, on="entry_id", how="left")

            # Ép kiểu số & sort giảm dần
            for c in ["p_top1", "p_top2", "p_top3"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.sort_values("p_top1", ascending=False)

            # Thêm ký hiệu % khi render
            show = out[["entry_name", "p_top1", "p_top2", "p_top3"]].copy()
            for c in ["p_top1", "p_top2", "p_top3"]:
                show[c] = show[c].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

            st.subheader(f"Xác suất Top 1/2/3 — GW{current_gw}")
            st.dataframe(show, use_container_width=True)
        else:
            st.info("Chưa có kết quả mô phỏng.")


