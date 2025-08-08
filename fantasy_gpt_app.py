import streamlit as st
import os
import json
import requests
import matplotlib.pyplot as plt

# ===== Cấu trúc dữ liệu =====
class Player:
    def __init__(self, name):
        self.name = name
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def total_score(self):
        return sum(self.scores)

    def score_str(self):
        return ", ".join(str(s) for s in self.scores)

class FantasyGroup:
    def __init__(self, name, group_code):
        self.name = name
        self.code = group_code
        self.players = {}

    def add_player(self, name):
        if name not in self.players:
            self.players[name] = Player(name)

    def add_round_scores(self, scores_dict):
        for name, score in scores_dict.items():
            if name in self.players:
                self.players[name].add_score(score)

    def get_leaderboard(self):
        return sorted(self.players.values(), key=lambda p: p.total_score(), reverse=True)

# ===== Lưu & tải dữ liệu =====
def save_to_file(group):
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump({
            "group": group.name,
            "code": group.code,
            "players": {
                name: player.scores for name, player in group.players.items()
            }
        }, f, ensure_ascii=False)

def load_from_file():
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            g = FantasyGroup(data["group"], data["code"])
            for name, scores in data["players"].items():
                g.add_player(name)
                for s in scores:
                    g.players[name].add_score(s)
            return g
    except:
        return FantasyGroup("Giải Fantasy 2025", "u3dip1")

# ===== Khởi tạo trạng thái Streamlit =====
if "group" not in st.session_state:
    st.session_state.group = load_from_file()

group = st.session_state.group

# ===== Giao diện =====
st.set_page_config(page_title="Fantasy GPT", page_icon="⚽")
st.title("⚽ Fantasy League Tracker")
st.markdown(f"🔐 Mã nhóm: `{group.code}`")

with st.sidebar:
    st.header("➕ Quản lý người chơi")
    name = st.text_input("Tên người chơi mới")
    if st.button("Thêm người chơi"):
        group.add_player(name)
        st.success(f"Đã thêm {name}")

st.subheader("📝 Nhập điểm cho vòng đấu")
scores = {}
for player in group.players.values():
    score = st.number_input(f"Điểm vòng này - {player.name}", min_value=0, key=player.name)
    scores[player.name] = score

if st.button("✅ Cập nhật điểm"):
    group.add_round_scores(scores)
    save_to_file(group)
    st.success("Đã cập nhật điểm cho vòng này!")

# ===== Bảng xếp hạng =====
st.subheader("📊 Bảng xếp hạng")
leaderboard = group.get_leaderboard()

if leaderboard:
    st.table({
        "Hạng": [i + 1 for i in range(len(leaderboard))],
        "Người chơi": [p.name for p in leaderboard],
        "Tổng điểm": [p.total_score() for p in leaderboard],
        "Chi tiết vòng": [p.score_str() for p in leaderboard]
    })
else:
    st.info("Chưa có người chơi nào hoặc chưa có điểm!")

# ===== Biểu đồ tiến độ điểm =====
st.subheader("📈 Biểu đồ tiến độ điểm theo vòng")
if leaderboard:
    fig, ax = plt.subplots()
    for player in leaderboard:
        rounds = list(range(1, len(player.scores) + 1))
        ax.plot(rounds, player.scores, marker='o', label=player.name)

    ax.set_xlabel("Vòng đấu")
    ax.set_ylabel("Điểm")
    ax.set_title("Tiến độ điểm theo từng vòng")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Chưa có dữ liệu để hiển thị biểu đồ.")

# ===== Lấy điểm tự động =====
st.subheader("🌍 Lấy điểm tự động từ FPL API")
if st.button("🛰️ Lấy điểm từ FPL API"):
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        res = requests.get(url)
        data = res.json()
        name_map = {p['web_name'].lower(): p['total_points'] for p in data['elements']}

        auto_scores = {}
        for player in group.players.values():
            pname = player.name.lower()
            if pname in name_map:
                auto_scores[player.name] = name_map[pname]
            else:
                auto_scores[player.name] = 0

        group.add_round_scores(auto_scores)
        save_to_file(group)
        st.success("✅ Đã cập nhật điểm tự động!")
    except Exception as e:
        st.error(f"Lỗi khi gọi API: {e}")