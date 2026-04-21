import streamlit as st
import pandas as pd
import time
import sqlite3
from datetime import datetime
from src.ui.components import render_section_header

def _save_listen(db_path: str, user_id: str, track_name: str, artist_name: str, timestamp: float, rec_sys):
    """Lưu lượt nghe vào SQLite và đảm bảo ID khớp với Model"""
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False, isolation_level=None)
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION;")

        # 1. Tìm msid từ metadata
        cur.execute("SELECT msid FROM item_meta WHERE track_name = ? AND artist_name = ? LIMIT 1", (track_name, artist_name))
        row = cur.fetchone()
        if not row:
            cur.execute("SELECT msid FROM item_meta WHERE track_name = ? LIMIT 1", (track_name,))
            row = cur.fetchone()
        
        if not row:
            cur.execute("ROLLBACK;")
            return False, "Không tìm thấy bài hát"
        
        msid = row[0]
        cur.execute("INSERT OR IGNORE INTO user_raw_items (user_id, msid) VALUES (?, ?)", (user_id, msid))

        # 2. Lấy ID chuẩn từ model
        uid = rec_sys.user2idx.get(str(user_id), -1)
        iid = rec_sys.item2idx.get(msid, -1)

        if uid != -1 and iid != -1:
            cur.execute("INSERT INTO user_item_ts (uid, iid, ts) VALUES (?, ?, ?)", (int(uid), int(iid), timestamp))
        
        cur.execute("COMMIT;")
        return True, f"Thành công: {track_name}"
    except Exception as e:
        if conn: conn.execute("ROLLBACK;")
        return False, str(e)
    finally:
        if conn: conn.close()

def _get_feed(rec_sys, user_id: str, n: int = 30) -> pd.DataFrame:
    """Lấy feed thông minh: Khôi phục từ DB nếu Session bị xóa (F5)"""
    recent_msids = []
    
    # 1. Kiểm tra trong Session State trước
    if "it_history" in st.session_state and st.session_state.it_history:
        # Lấy 3 bài gần nhất từ session
        recent_tracks = [h['track'].lower() for h in st.session_state.it_history[-3:]]
        for track in recent_tracks:
            res = rec_sys.item_meta.find_by_track(track)
            if res: recent_msids.append(res[0])
            
    # 2. Nếu Session trống (do reload), khôi phục từ Database
    if not recent_msids:
        db_h = rec_sys.get_user_history(user_id, limit=3)
        if not db_h.empty:
            for _, row in db_h.iterrows():
                res = rec_sys.item_meta.find_by_track(row['Bài hát'].lower())
                if res: recent_msids.append(res[0])

    # 3. Trả về kết quả từ hàm Next In Session (Fix TypeError: remove user_id)
    if recent_msids:
        return rec_sys.recommend_next_in_session(session_msids=recent_msids, n=n)
    
    return rec_sys.recommend_hybrid(user_id, n=n)

def render_interactive_tab(rec_sys, user_input, n_recs, db_path="./mappings.db"):
    render_section_header("Interactive Session", 
                        subtitle="Hệ thống sẽ thay đổi gợi ý ngay lập tức dựa trên mỗi tương tác của bạn.",
                        icon_name="mouse-pointer",
                        color="#10b981")

    # Khởi tạo session state cho lịch sử click nếu chưa có
    if "it_history" not in st.session_state:
        st.session_state.it_history = []

    # --- PHẦN 1: HIỂN THỊ TRẠNG THÁI ---
    col_info, col_reset = st.columns([4, 1])
    
    # Logic hiển thị gợi ý đang dựa trên cái gì
    display_history = st.session_state.it_history
    if not display_history:
        # Thử lấy từ DB để hiển thị 'breadcrumb'
        db_h = rec_sys.get_user_history(user_input, limit=3)
        if not db_h.empty:
            display_history = [{"track": r['Bài hát']} for _, r in db_h.iterrows()]
            display_history.reverse()

    with col_info:
        if display_history:
            history_str = " ➔ ".join([f"**{h['track']}**" for h in display_history[-3:]])
            st.info(f"Đang nghe: {history_str}")
        else:
            st.info("Hãy chọn một bài hát để bắt đầu phiên nghe cá nhân hóa.")

    with col_reset:
        if st.button("Reset Session", use_container_width=True):
            st.session_state.it_history = []
            st.rerun()

    st.divider()

    # --- PHẦN 2: FETCH & DISPLAY FEED ---
    st.markdown("""
        <style>
        /* Ép ảnh về tỉ lệ 1:1 và bo góc */
        [data-testid="stImage"] img {
            object-fit: cover;
            height: 120px !important; 
            border-radius: 5px;
        }
        /* Thu nhỏ font tên bài hát và ép hiển thị tối đa 1 dòng */
        .song-title {
            font-size: 13px !important;
            font-weight: bold;
            margin-bottom: 2px !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Thu nhỏ font nghệ sĩ */
        .artist-name {
            font-size: 11px !important;
            color: #888;
            margin-bottom: 8px !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Thu nhỏ nút bấm cực gọn */
        .stButton>button {
            padding: 0px !important;
            font-size: 10px !important;
            height: 22px !important;
            border-radius: 4px !important;
        }
        /* Giảm khoảng cách giữa các phần tử trong card */
        [data-testid="stVerticalBlock"] {
            gap: 0.2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.spinner("Đang cập nhật luồng nhạc..."):
        feed_df = _get_feed(rec_sys, user_input, n=n_recs)

    if not feed_df.empty:
        num_cols = 5 
        cols = st.columns(num_cols)
        
        for idx, row in feed_df.iterrows():
            with cols[idx % num_cols]:
                with st.container(border=True):
                    # 1. Ảnh động (Logic Picsum seed giống các tab khác)
                    safe_seed = "".join([c for c in str(row['track_name']) if c.isalnum()])
                    img_url = f"https://picsum.photos/seed/{safe_seed}/200"
                    st.image(img_url, use_container_width=True)
                    
                    # 2. Thông tin bài hát (Dùng HTML để ép cỡ chữ nhỏ)
                    st.markdown(f'<p class="song-title">{row["track_name"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="artist-name">{row["artist_name"]}</p>', unsafe_allow_html=True)
                    
                    # 3. Nút bấm nhỏ gọn
                    button_key = f"it_btn_{row['track_name']}_{idx}"
                    if st.button("Nghe", key=button_key, use_container_width=True):
                        actual_ts = datetime.now().timestamp()
                        ok, msg = _save_listen(db_path, user_input, row['track_name'], row['artist_name'], actual_ts, rec_sys)
                        
                        if ok:
                            # ĐỒNG BỘ: Cập nhật Lịch sử âm nhạc và trình phát chính
                            from src.ui.components import handle_play_song
                            handle_play_song(row['track_name'], row['artist_name'], img_url)
                            
                            st.session_state.it_history.append({
                                "track": row['track_name'],
                                "artist": row['artist_name'],
                                "ts": actual_ts
                            })
                            st.toast(msg)
                            time.sleep(0.5)
                            st.rerun()
    else:
        st.warning("Không có gợi ý nào phù hợp hiện tại.")
