import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

LEAGUE_ID = 1007447  # Thay bằng League ID của bạn

def get_league_standings(league_id):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
    res = requests.get(url)
    data = res.json()
    return data["standings"]["results"]

def get_player_history(entry_id):
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/history/"
    res = requests.get(url)
    data = res.json()
    return data["current"]

st.set_page_config(page_title="FPL League Pro", page_icon="⚽", layout="wide")
st.title("⚽ Fantasy Premier League: League Tracker Pro")

st.info(f"Mã giải đấu của bạn: `{LEAGUE_ID}`")

with st.spinner("Đang lấy dữ liệu nhóm và điểm số từng vòng..."):
    players = get_league_standings(LEAGUE_ID)
    all_players_scores = []

    for p in players:
        entry_id = p["entry"]
        player_name = p["player_name"]
        team_name = p["entry_name"]
        rounds = get_player_history(entry_id)
        scores = [gw["points"] for gw in rounds]
        total = sum(scores)
        all_players_scores.append({
            "player_name": player_name,
            "team_name": team_name,
            "scores": scores,
            "total": total,
        })

    # Xếp hạng tổng
    all_players_scores = sorted(all_players_scores, key=lambda x: x["total"], reverse=True)

num_gws = max(len(p["scores"]) for p in all_players_scores)

# ====== Chọn vòng đấu để xem chi tiết ======
gw_selected = st.slider("Chọn vòng đấu", 1, num_gws, num_gws)

# ====== Bảng điểm từng vòng ======
gw_sorted = sorted(all_players_scores, key=lambda x: x["scores"][gw_selected-1] if len(x["scores"])>=gw_selected else 0, reverse=True)
st.subheader(f"🏆 Bảng điểm vòng {gw_selected}")
st.table({
    "Hạng": [i+1 for i in range(len(gw_sorted))],
    "Tên người chơi": [p["player_name"] for p in gw_sorted],
    "Tên đội": [p["team_name"] for p in gw_sorted],
    "Điểm vòng này": [p["scores"][gw_selected-1] if len(p["scores"])>=gw_selected else 0 for p in gw_sorted],
})

# ====== Bảng xếp hạng tổng ======
st.subheader("📊 Bảng xếp hạng tổng")
st.table({
    "Hạng": [i+1 for i in range(len(all_players_scores))],
    "Tên người chơi": [p["player_name"] for p in all_players_scores],
    "Tên đội": [p["team_name"] for p in all_players_scores],
    "Tổng điểm": [p["total"] for p in all_players_scores],
    "Chi tiết vòng": [p["scores"] for p in all_players_scores],
})

# ====== Xuất dữ liệu ra CSV ======
data = []
for p in all_players_scores:
    row = {"Tên người chơi": p["player_name"], "Tên đội": p["team_name"], "Tổng điểm": p["total"]}
    for i, score in enumerate(p["scores"], 1):
        row[f"GW{i}"] = score
    data.append(row)
df = pd.DataFrame(data)
csv = df.to_csv(index=False).encode()
st.download_button("⬇️ Tải dữ liệu CSV", data=csv, file_name="fpl_league.csv", mime="text/csv")

# ====== Biểu đồ tiến độ điểm tổng ======
st.subheader("📈 Biểu đồ tiến độ điểm tổng theo vòng")
fig, ax = plt.subplots()
for p in all_players_scores:
    # Tính điểm tích lũy mỗi vòng
    cumulative = [sum(p["scores"][:i+1]) for i in range(len(p["scores"]))]
    ax.plot(range(1, len(cumulative)+1), cumulative, marker='o', label=p["player_name"])
ax.set_xlabel("Vòng đấu")
ax.set_ylabel("Điểm tích lũy")
ax.set_title("Tiến độ tổng điểm tích lũy")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ====== So sánh điểm tuần này/tuần trước ======
if gw_selected > 1:
    st.subheader("📈 So sánh điểm tuần này với tuần trước")
    comp_table = {
        "Tên người chơi": [],
        f"GW{gw_selected-1}": [],
        f"GW{gw_selected}": [],
        "Chênh lệch": [],
    }
    for p in all_players_scores:
        last = p["scores"][gw_selected-2] if len(p["scores"])>=gw_selected else 0
        now = p["scores"][gw_selected-1] if len(p["scores"])>=gw_selected else 0
        comp_table["Tên người chơi"].append(p["player_name"])
        comp_table[f"GW{gw_selected-1}"].append(last)
        comp_table[f"GW{gw_selected}"].append(now)
        comp_table["Chênh lệch"].append(now-last)
    st.dataframe(pd.DataFrame(comp_table))

# ====== Top scorer mỗi vòng ======
st.subheader("🌟 Top scorer mỗi vòng")
top_scorers = []
for gw in range(num_gws):
    max_score = max(p["scores"][gw] for p in all_players_scores if len(p["scores"])>gw)
    names = [p["player_name"] for p in all_players_scores if len(p["scores"])>gw and p["scores"][gw]==max_score]
    top_scorers.append(", ".join(names))
df_top = pd.DataFrame({"Vòng": list(range(1,num_gws+1)), "Top scorer": top_scorers})
st.dataframe(df_top)

# ====== Đếm số lần vào top 3, top 5 ======
top3_count = defaultdict(int)
top5_count = defaultdict(int)

for gw in range(num_gws):
    scores_gw = [(p["player_name"], p["scores"][gw] if len(p["scores"])>gw else 0) for p in all_players_scores]
    top3 = sorted(scores_gw, key=lambda x: x[1], reverse=True)[:3]
    top5 = sorted(scores_gw, key=lambda x: x[1], reverse=True)[:5]
    for name, _ in top3:
        top3_count[name] += 1
    for name, _ in top5:
        top5_count[name] += 1

st.subheader("🥉 Số lần vào top 3 từng vòng")
st.write(dict(top3_count))

st.subheader("🥇 Số lần vào top 5 từng vòng")
st.write(dict(top5_count))
