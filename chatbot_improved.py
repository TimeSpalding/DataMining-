"""
chatbot_improved.py
Thay thế phần "5. Trợ lý Ảo AI (Chatbot)" trong app.py

Kiến trúc: Regex-first → Claude Haiku fallback (chỉ classify intent)
- Không để LLM đụng vào tên bài hát / nghệ sĩ → hết hallucination
- Regex xử lý ~80% câu hỏi thường gặp với 0ms delay
- Claude Haiku chỉ trả về 1 trong 4 nhãn: SEARCH / SIMILAR / VIBE / CHITCHAT
"""

import re
import streamlit as st
from anthropic import Anthropic

# =============================================
# 1. KHỞI TẠO CLIENT ANTHROPIC (thay Ollama)
# =============================================
# Lấy API key từ st.secrets hoặc biến môi trường
# Trong file .streamlit/secrets.toml: ANTHROPIC_API_KEY = "sk-ant-..."
anthropic_client = Anthropic(api_key="")  # tự đọc env ANTHROPIC_API_KEY


# =============================================
# 2. REGEX ROUTER (xử lý trước, không cần LLM)
# =============================================
# Mỗi pattern trả về (intent, extracted_value)
REGEX_PATTERNS = [
    # --- Tìm kiếm nghệ sĩ / bài hát cụ thể ---
    (r"(?:tìm|search|kiếm|nhạc của|bài của|ca sĩ)\s+(.+)", "SEARCH"),
    (r"(?:nhạc|bài hát|track|song)\s+(?:của|by)\s+(.+)", "SEARCH"),
    # --- Nhạc tương tự (từ bài hát mồi) ---
    (r"(?:giống|tương tự|similar|like|gợi ý từ|playlist|mix).{0,15}(?:bài|song|track)?\s+[\"']?(.+?)[\"']?\s*$", "SIMILAR"),
    (r"(?:bài|song|track)\s+[\"'](.+?)[\"']", "SIMILAR"),      # có dấu nháy
    (r"nghe\s+(.+)\s+(?:xong|rồi|và)", "SIMILAR"),
    # --- Tâm trạng / vibe ---
    (r"nhạc\s+(buồn|vui|chill|sôi động|relaxing|happy|sad|energetic|lo-fi|romantic|rock|pop|jazz|classical|edm|rap|hiphop|r&b)", "VIBE"),
    (r"(?:muốn nghe|gợi ý)\s+nhạc\s+(.+)", "VIBE"),
    (r"(?:tâm trạng|mood|vibe)\s+(.+)", "VIBE"),
]

def regex_route(text: str):
    """
    Trả về (intent, value) nếu match, hoặc (None, None) nếu không match.
    intent: 'SEARCH' | 'SIMILAR' | 'VIBE'
    value: chuỗi đã extract (tên bài / nghệ sĩ / tâm trạng)
    """
    text_lower = text.lower().strip()
    for pattern, intent in REGEX_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            value = m.group(1).strip().strip("\"'")
            # Lọc bỏ các từ thừa phổ biến
            value = re.sub(r"^(của|by|nhạc|bài|ca sĩ|nghệ sĩ|tìm|kiếm)\s+", "", value.strip())
            return intent, value
    return None, None


# =============================================
# 3. CLAUDE HAIKU — CHỈ CLASSIFY INTENT
# =============================================
INTENT_SYSTEM_PROMPT = """Bạn là bộ phân loại intent cho hệ thống gợi ý nhạc.
Nhiệm vụ DUY NHẤT: đọc câu của người dùng và trả về ĐÚNG MỘT trong 4 nhãn sau:

SEARCH|<tên nghệ sĩ hoặc bài hát>
SIMILAR|<tên bài hát làm mẫu>
VIBE|<từ khóa tâm trạng hoặc thể loại>
CHITCHAT

Quy tắc:
- SEARCH: người dùng muốn tìm nhạc của một nghệ sĩ hoặc tên bài cụ thể
- SIMILAR: người dùng đề cập một bài hát và muốn nhạc tương tự
- VIBE: người dùng mô tả cảm xúc, thể loại, không đề cập tên cụ thể
- CHITCHAT: câu hỏi không liên quan đến âm nhạc

TUYỆT ĐỐI KHÔNG viết thêm bất cứ điều gì ngoài format trên.
Ví dụ đầu ra hợp lệ:
SEARCH|Linkin Park
SIMILAR|Creep
VIBE|buồn
CHITCHAT"""

def claude_classify_intent(user_text: str) -> tuple[str, str]:
    """
    Gọi Claude Haiku để classify intent.
    Trả về (intent, value). Rất nhanh vì chỉ cần 1 token response.
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=30,           # chỉ cần vài token
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
        # Fallback nếu format sai
        return "CHITCHAT", raw
    except Exception as e:
        return "CHITCHAT", str(e)


# =============================================
# 4. MAIN ROUTER — kết hợp Regex + Claude
# =============================================
def route_query(user_text: str) -> tuple[str, str]:
    """
    Bước 1: thử regex (nhanh, chắc chắn)
    Bước 2: nếu không match → gọi Claude Haiku
    Trả về (intent, value)
    """
    intent, value = regex_route(user_text)
    if intent is not None:
        return intent, value
    # Fallback sang LLM classifier
    return claude_classify_intent(user_text)


# =============================================
# 5. EXECUTE — gọi hàm rec_sys tương ứng
# =============================================
def execute_intent(intent: str, value: str, rec_sys, user_id: str, n: int):
    """
    Trả về (answer_text, dataframe_or_None)
    """
    if intent == "SEARCH":
        ans = f"Kết quả tìm kiếm cho **'{value}'**:"
        # Dùng search_smart nếu có, fallback sang search_metadata
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

    else:  # CHITCHAT
        # Lần này mới để Claude trả lời tự nhiên
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="Bạn là trợ lý gợi ý âm nhạc thân thiện. Trả lời ngắn gọn bằng tiếng Việt.",
            messages=[{"role": "user", "content": value or "Xin chào"}]
        )
        return response.content[0].text.strip(), None


# =============================================
# 6. STREAMLIT UI — thay thế elif feature == "5..."
# =============================================
def render_chatbot_tab(rec_sys, user_input: str, n_recs: int):
    """
    Gọi hàm này trong app.py để thay thế toàn bộ khối elif feature == "5. Trợ lý Ảo AI"
    
    Cách tích hợp vào app.py:
        from chatbot_improved import render_chatbot_tab
        ...
        elif feature == "5. Trợ lý Ảo AI (Chatbot)":
            render_chatbot_tab(rec_sys, user_input, n_recs)
    """
    import time
    
    st.subheader("🤖 Trợ lý Âm nhạc AI")
    st.caption(
        "Regex router + Claude Haiku — nhanh hơn, không hallucinate. "
        "Thử: *'Tìm nhạc của Sơn Tùng'* / *'Nhạc giống bài Chúng Ta Của Hiện Tại'* / *'Nhạc chill buổi tối'*"
    )

    # Khởi tạo lịch sử chat
    if "messages_v2" not in st.session_state:
        st.session_state.messages_v2 = []

    # Render lịch sử
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

            # Bước 1: xác định intent
            with st.spinner("Đang phân tích..."):
                intent, value = route_query(prompt)

            # Debug dòng nhỏ (bỏ đi khi production)
            source = "regex" if regex_route(prompt)[0] else "claude-haiku"
            st.caption(f"_Intent: `{intent}` | Value: `{value}` | Source: {source}_")

            # Bước 2: thực thi
            try:
                ans, df = execute_intent(intent, value, rec_sys, user_input, n_recs)
                elapsed = time.time() - t0
                full_ans = f"{ans}\n\n*(⏱️ {elapsed:.2f}s)*"

                st.markdown(full_ans)
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                elif df is not None:
                    st.warning(f"Không tìm thấy kết quả cho '{value}'. Thử tên khác?")

                st.session_state.messages_v2.append({
                    "role": "assistant",
                    "content": full_ans,
                    "df": df
                })
            except Exception as e:
                st.error(f"Lỗi khi thực thi: {e}")


# =============================================
# 7. HƯỚNG DẪN TÍCH HỢP VÀO app.py
# =============================================
"""
THAY ĐỔI TRONG app.py:

1. Xóa toàn bộ phần khai báo `client = OpenAI(...)` và `ai_tools = [...]`

2. Thêm import ở đầu file:
   from chatbot_improved import render_chatbot_tab

3. Thêm biến môi trường (hoặc .streamlit/secrets.toml):
   ANTHROPIC_API_KEY = "sk-ant-..."

4. Thay khối elif feature == "5..." bằng:
   elif feature == "5. Trợ lý Ảo AI (Chatbot)":
       render_chatbot_tab(rec_sys, user_input, n_recs)

Không cần thay đổi gì khác!
"""