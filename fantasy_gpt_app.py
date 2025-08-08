import streamlit as st
import openai
import os
from dotenv import load_dotenv
import random
import string

# ===== Load API Key =====
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# ===== Khởi tạo trạng thái Streamlit =====
if "group" not in st.session_state:
    st.session_state.group = FantasyGroup("Giải Fantasy 2025", "u3dip1")

group = st.session_state.group

# ===== Giao diện =====
st.set_page_config(page_title="Fantasy GPT", page_icon="⚽")
st.title("⚽ Fantasy League Tracker + GPT Gợi Ý Đội Hình")
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

# ===== GPT Gợi ý đội hình =====
st.subheader("🤖 GPT Gợi Ý Đội Hình Fantasy")
current_team = st.text_area("Nhập đội hình hiện tại (ví dụ: Haaland, Salah, Saka...)")

if st.button("🎯 Gợi ý thay đổi đội hình"):
    if current_team.strip() == "":
        st.warning("⛔ Vui lòng nhập đội hình.")
    else:
        prompt = (
            "Bạn là chuyên gia Fantasy Premier League. Hãy phân tích đội hình sau:\n"
            f"{current_team}\n"
            "Đưa ra 2-3 gợi ý thay đổi để tối ưu đội hình, tránh chấn thương, lựa chọn cầu thủ tiềm năng. Ngắn gọn, súc tích, dưới 150 từ."
        )

        with st.spinner("GPT đang phân tích..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Bạn là chuyên gia Fantasy Premier League."},
                        {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content
                st.success("✅ Gợi ý từ GPT:")
                st.write(reply)
            except Exception as e:
                st.error(f"Lỗi: {e}")