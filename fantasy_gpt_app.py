import streamlit as st
import requests
import matplotlib.pyplot as plt

LEAGUE_ID = 1007448  # League của bạn

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

st.set_page_config(page_title="FPL Auto League Tracker", page_icon="⚽")
st.title("⚽ Fantasy Premier League: League Auto Tracker")

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

    # Xếp hạng theo tổng điểm
    all_players_scores = sorted(all_players_scores, key=lambda x: x["total"], reverse=True)

# ====== Bảng xếp hạng ======
st.subheader("📊 Bảng xếp hạng nhóm")
st.table({
    "Hạng": [i+1 for i in range(len(all_players_scores))],
    "Tên người chơi": [p["player_name"] for p in all_players_scores],
    "Tên đội": [p["team_name"] for p in all_players_scores],
    "Tổng điểm": [p["total"] for p in all_players_scores],
    "Chi tiết vòng": [p["scores"] for p in all_players_scores],
})

# ====== Biểu đồ tiến độ điểm ======
st.subheader("📈 Biểu đồ tiến độ điểm từng vòng")
fig, ax = plt.subplots()
for p in all_players_scores:
    ax.plot(range(1, len(p["scores"])+1), p["scores"], marker='o', label=p["player_name"])
ax.set_xlabel("Vòng đấu")
ax.set_ylabel("Điểm")
ax.set_title("Tiến độ điểm từng vòng của các người chơi")
ax.legend()
ax.grid(True)
st.pyplot(fig)
