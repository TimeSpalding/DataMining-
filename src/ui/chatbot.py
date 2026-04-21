import re
import streamlit as st
import time
from anthropic import Anthropic
from src.ui.components import render_section_header

# =============================================
# 1. KHỞI TẠO CLIENT ANTHROPIC 
# =============================================
anthropic_client = Anthropic(api_key="")  # tự đọc env ANTHROPIC_API_KEY


# =============================================
# 2. REGEX ROUTER
# =============================================
REGEX_PATTERNS = [
    (r"(?:tìm|search|kiếm|nhạc của|bài của|ca sĩ)\s+(.+)", "SEARCH"),
    (r"(?:nhạc|bài hát|track|song)\s+(?:của|by)\s+(.+)", "SEARCH"),
    (r"(?:giống|tương tự|similar|like|gợi ý từ|playlist|mix).{0,15}(?:bài|song|track)?\s+[\"']?(.+?)[\"']?\s*$", "SIMILAR"),
    (r"(?:bài|song|track)\s+[\"'](.+?)[\"']", "SIMILAR"),      
    (r"nghe\s+(.+)\s+(?:xong|rồi|và)", "SIMILAR"),
    (r"nhạc\s+(buồn|vui|chill|sôi động|relaxing|happy|sad|energetic|lo-fi|romantic|rock|pop|jazz|classical|edm|rap|hiphop|r&b)", "VIBE"),
    (r"(?:muốn nghe|gợi ý)\s+nhạc\s+(.+)", "VIBE"),
    (r"(?:tâm trạng|mood|vibe)\s+(.+)", "VIBE"),
]

def regex_route(text: str):
    text_lower = text.lower().strip()
    for pattern, intent in REGEX_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            value = m.group(1).strip().strip("\"'")
            value = re.sub(r"^(của|by|nhạc|bài|ca sĩ|nghệ sĩ|tìm|kiếm)\s+", "", value.strip())
            return intent, value
    return None, None


# =============================================
# 3. CLAUDE HAIKU — CHỈ CLASSIFY INTENT
# =============================================
INTENT_SYSTEM_PROMPT = """Bạn là bộ phân loại intent cho hệ thống gợi ý nhạc.
Nhiệm vụ DUY NHẤS: đọc câu của người dùng và trả về ĐÚNG MỘT trong 4 nhãn sau:
SEARCH|<tên nghệ sĩ hoặc bài hát>
SIMILAR|<tên bài hát làm mẫu>
VIBE|<từ khóa tâm trạng hoặc thể loại>
CHITCHAT
"""

def claude_classify_intent(user_text: str) -> tuple[str, str]:
    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=30,           
            system=INTENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}]
        )
        raw = response.content[0].text.strip()
        if "|" in raw:
            parts = raw.split("|", 1)
            intent = parts[0].strip().upper()
            value = parts[1].strip() if len(parts) > 1 else ""
            if intent in ("SEARCH", "SIMILAR", "VIBE", "CHITCHAT"):
                return intent, value
        return "CHITCHAT", raw
    except Exception:
        return "CHITCHAT", "Error"


# =============================================
# 4. MAIN ROUTER 
# =============================================
def route_query(user_text: str) -> tuple[str, str]:
    intent, value = regex_route(user_text)
    if intent is not None:
        return intent, value
    return claude_classify_intent(user_text)


# =============================================
# 5. EXECUTE 
# =============================================
def execute_intent(intent: str, value: str, rec_sys, user_id: str, n: int):
    if intent == "SEARCH":
        ans = f"Kết quả tìm kiếm cho **'{value}'**:"
        fn = getattr(rec_sys, "search_smart", None) or getattr(rec_sys, "search_metadata")
        df = fn(value, n=n)
        return ans, df
    elif intent == "SIMILAR":
        ans = f"Playlist tương tự bài **'{value}'**:"
        df = rec_sys.generate_playlist(user_id, seed_track_names=[value], n_songs=n)
        return ans, df
    elif intent == "VIBE":
        ans = f"Nhạc theo tâm trạng **'{value}'**:"
        df = rec_sys.recommend_cold_content(text_query=value, n=n)
        return ans, df
    else:  
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="Bạn là trợ lý gợi ý âm nhạc thân thiện. Trả lời ngắn gọn bằng tiếng Việt.",
            messages=[{"role": "user", "content": value or "Xin chào"}]
        )
        return response.content[0].text.strip(), None


# =============================================
# 6. STREAMLIT UI 
# =============================================
def render_chatbot_tab(rec_sys, user_input: str, n_recs: int):
    render_section_header("AI Music Companion", 
                        subtitle="Hãy trò chuyện với tôi về âm nhạc, cảm xúc hoặc yêu cầu gợi ý bài hát.",
                        icon_name="message-circle",
                        color="#8b5cf6")
    
    st.caption("Hãy thử: 'Tìm nhạc của Sơn Tùng', 'Nhạc tương tự bài Creep', hoặc 'Nhạc chill buổi tối'.")

    if "messages_v2" not in st.session_state:
        st.session_state.messages_v2 = []

    for msg in st.session_state.messages_v2:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("df") is not None:
                st.dataframe(msg["df"], use_container_width=True)

    prompt = st.chat_input("Nhập câu hỏi về âm nhạc...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages_v2.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            t0 = time.time()
            with st.spinner("Đ Đang phân tích..."):
                intent, value = route_query(prompt)

            try:
                ans, df = execute_intent(intent, value, rec_sys, user_input, n_recs)
                elapsed = time.time() - t0
                full_ans = f"{ans}\n\n(Thực thi trong: {elapsed:.2f}s)"

                st.markdown(full_ans)
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)

                st.session_state.messages_v2.append({
                    "role": "assistant",
                    "content": full_ans,
                    "df": df
                })
            except Exception as e:
                st.error(f"Lỗi khi thực thi: {e}")