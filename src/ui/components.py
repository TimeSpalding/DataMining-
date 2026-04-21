import streamlit as st
import pandas as pd
from datetime import datetime
import textwrap
import random
import os

def _get_random_local_song():
    songs_dir = "./data/songs"
    if not os.path.exists(songs_dir):
        return None
    songs = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    if not songs:
        return None
    return os.path.join(songs_dir, random.choice(songs))

def inject_custom_css():
    css_content = textwrap.dedent("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {
            --bg-main: #f0f7f4;
            --text-main: #064e3b;
            --text-sub: #059669;
            --card-bg: rgba(255, 255, 255, 0.8);
            --spotify-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-green: #22c55e;
            --accent-light-green: #4ade80;
        }

        .stApp, .stApp [data-testid="stHeader"], .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp span {
            color: var(--text-main) !important;
        }

        .stApp {
            background: 
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(245, 158, 11, 0.1) 0px, transparent 50%),
                var(--bg-main);
            font-family: 'Outfit', sans-serif !important;
        }

        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 8rem !important;
        }

        /* Custom Card Styles */
        .song-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            padding: 16px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            gap: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .song-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            background: rgba(255, 255, 255, 0.9);
        }
        .song-card img {
            width: 100%;
            border-radius: 12px;
            aspect-ratio: 1/1;
            object-fit: cover;
        }
        .song-title {
            color: var(--text-main);
            font-weight: 700;
            font-size: 1.1rem;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .song-artist {
            color: var(--text-sub);
            font-size: 0.9rem;
            margin: 0;
        }
        
        /* Modern Section Header styling */
        .section-header-modern {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-top: 3rem;
            margin-bottom: 0.5rem;
            padding-bottom: 12px;
        }
        .section-header-modern h3 {
            color: var(--text-main);
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.025em;
        }
        .header-accent-bar {
            width: 8px;
            height: 45px;
            border-radius: 4px;
        }
        .section-subtitle-modern {
            color: var(--text-sub);
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 2rem;
            opacity: 0.8;
            padding-left: 24px;
        }

        /* Input styling for light mode - Green Accents! */
        div[data-testid="stTextInput"] input, 
        div[data-testid="stNumberInput"] input, 
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stDateInput"] input {
            background-color: white !important;
            color: var(--text-main) !important;
            border: 2px solid var(--accent-light-green) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.1) !important;
        }
        
        div[data-testid="stSelectbox"] [data-baseweb="select"], 
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background-color: white !important;
            color: var(--text-main) !important;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"]:hover,
        div[data-testid="stDateInput"] input:hover {
            border-color: var(--accent-green) !important;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
        }

        /* Fix Selectbox internal text and icons */
        div[data-testid="stSelectbox"] svg {
            fill: var(--accent-green) !important;
        }
        
        /* Checkbox styling - Green */
        div[data-testid="stCheckbox"] [role="checkbox"] {
            background-color: white !important;
            border-color: var(--accent-green) !important;
        }
        div[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
            background-color: var(--accent-green) !important;
        }
        
        /* Dropdown menu and popover styling (Baseweb Portals) */
        div[data-baseweb="popover"], div[data-baseweb="menu"], div[role="listbox"] {
            background-color: white !important;
            color: var(--text-main) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
        }
        
        div[data-baseweb="popover"] * {
            color: var(--text-main) !important;
            background-color: transparent !important;
        }
        
        div[data-baseweb="popover"] button {
            background-color: transparent !important;
        }

        li[role="option"] {
            background-color: white !important;
            color: var(--text-main) !important;
            padding: 10px 20px !important;
        }
        li[role="option"]:hover {
            background-color: rgba(139, 92, 246, 0.1) !important;
        }

        /* Specifically for the calendar days - Fix visibility */
        div[data-baseweb="calendar"] {
            background-color: white !important;
        }
        div[data-baseweb="calendar"] header, div[data-baseweb="calendar"] [role="grid"] {
            background-color: white !important;
            color: #064e3b !important;
        }
        div[data-baseweb="calendar"] * {
            color: #064e3b !important;
        }
        
        /* Ensure days are visible and not hidden */
        div[aria-roledescription="calendar day"] {
            color: #064e3b !important;
            background-color: white !important;
            font-weight: 600 !important;
        }
        
        /* Fix the squares for days outside the current month - No opacity 0 */
        div[data-baseweb="calendar"] [role="grid"] > div:not([aria-label]) {
            background-color: white !important;
            color: #d1d5db !important; /* Light grey for empty days */
        }
        
        /* Selected day highlight - Bright Green */
        div[data-baseweb="calendar"] [aria-selected="true"],
        div[data-baseweb="calendar"] [aria-selected="true"] * {
            background-color: #22c55e !important;
            color: white !important;
            border-radius: 50% !important;
        }
        
        /* Highlight Today */
        div[data-baseweb="calendar"] [aria-current="date"] {
            border: 2px solid var(--accent-green) !important;
            border-radius: 50% !important;
        }

        /* Chat UI styling */
        div[data-testid="stChatMessage"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            border-radius: 16px !important;
            color: var(--text-main) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        }
        div[data-testid="stChatMessage"] p {
            color: var(--text-main) !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: white !important;
            color: var(--text-main) !important;
            border: 2px solid var(--accent-purple) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stChatInput"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(20px);
            border-top: 1px solid rgba(139, 92, 246, 0.2);
            padding: 15px !important;
            box-shadow: 0 -10px 25px rgba(139, 92, 246, 0.1);
        }
        
        /* Remove black area at the bottom and match theme */
        footer {display: none !important;}
        [data-testid="stHeader"] {background: transparent !important;}
        div[data-testid="stBottom"] {
            background-color: transparent !important;
        }
        div[data-testid="stBottomBlockContainer"] {
            background-color: transparent !important;
            padding-bottom: 20px !important;
        }
        .stApp {
            background-color: var(--bg-main) !important;
        }
        
        /* Fixed Player Wrapper Style (Light Glassmorphism) */
        #fixed-player-hook + div {
            position: fixed !important;
            bottom: 20px !important;
            left: 20px !important;
            right: 20px !important;
            width: calc(100% - 40px) !important;
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 24px !important;
            z-index: 100000 !important;
            padding: 15px 40px !important;
            height: 90px !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Custom buttons styling */
        div[data-testid="stButton"] button {
            background-color: white !important;
            color: var(--accent-purple) !important;
            border: 1px solid var(--accent-purple) !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        div[data-testid="stButton"] button:hover {
            background-color: var(--accent-purple) !important;
            color: white !important;
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(139, 92, 246, 0.2) !important;
        }
        div[data-testid="stChatMessageContainer"] {
            padding-bottom: 140px !important;
        }
        
        /* Audio player coloring */
        audio {
            height: 35px;
            width: 100%;
            border-radius: 20px;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(0, 0, 0, 0.05);
        }
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
            color: var(--text-main) !important;
        }
        
        /* Fix Tabs Colors */
        button[data-baseweb="tab"] {
            color: var(--text-sub) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent-purple) !important;
            border-bottom-color: var(--accent-purple) !important;
        }
        </style>
        
        <!-- Load Lucide Icons -->
        <script src="https://unpkg.com/lucide@latest"></script>
        <script>
            // Initialize lucide icons
            document.addEventListener('DOMContentLoaded', (event) => {
                lucide.createIcons();
            });
            // Also run it periodically in case of streamlit re-renders
            setInterval(() => {
                lucide.createIcons();
            }, 1000);
        </script>
    """)
    st.markdown(css_content, unsafe_allow_html=True)


def render_section_header(title, subtitle=None, icon_name=None, color="#8b5cf6"):
    """
    Render a vibrant section header with a color accent and Lucide icon.
    """
    icon_html = f'<i data-lucide="{icon_name}" style="width: 32px; height: 32px; color: {color}"></i>' if icon_name else ""
    header_html = f"""
    <div class="section-header-modern">
        <div class="header-accent-bar" style="background: {color}; box-shadow: 0 0 15px {color}80;"></div>
        {icon_html}
        <h3>{title}</h3>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="section-subtitle-modern">{subtitle}</p>', unsafe_allow_html=True)


def handle_play_song(song_title, artist_name, img_url):
    audio_path = _get_random_local_song()
    # Cập nhật bài hát hiện tại
    st.session_state['current_song'] = {
        'title': song_title,
        'artist': artist_name,
        'image': img_url,
        'audio_path': audio_path
    }
    
    # Cập nhật lịch sử
    if 'history_list' not in st.session_state:
        st.session_state['history_list'] = []
        
    st.session_state['history_list'].append({
        'Bài hát': song_title,
        'Nghệ sĩ': artist_name,
        'Ngày nghe': datetime.now().strftime("%d/%m/%Y"),
        'Giờ nghe': datetime.now().strftime("%H:%M:%S"),
        'image': img_url
    })


def render_song_cards(df, key_prefix="card"):
    """
    Nhận DataFrame chứa 'track_name', 'artist_name' và render dưới dạng lưới Card.
    """
    if df is None or df.empty:
        st.warning("Không có dữ liệu bài hát.")
        return

    # Xác định các cột để dùng
    track_col = 'track_name' if 'track_name' in df.columns else 'Bài hát'
    artist_col = 'artist_name' if 'artist_name' in df.columns else 'Nghệ sĩ'
    
    # Render từng hàng ngang (ví dụ 5 bài một hàng)
    cols_per_row = 5
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(df):
                row = df.iloc[idx]
                song_title = row.get(track_col, "Unknown Track")
                artist_name = row.get(artist_col, "Unknown Artist")
                
                # Tạo link ảnh giả dựa trên tên để nhất quán
                safe_seed = "".join([c for c in str(song_title) if c.isalnum()])
                # Lấy ảnh từ cột 'image' nếu có, nếu không thì dùng placeholder
                img_url = row.get('image') if 'image' in df.columns else f"https://picsum.photos/seed/{safe_seed}/200"
                if pd.isna(img_url) or not img_url:
                    img_url = f"https://picsum.photos/seed/{safe_seed}/200"
                
                # Kiểm tra xem có cột thời gian nghe không (để hiển thị trong tab Lịch sử)
                time_info_html = ""
                if "Giờ nghe" in df.columns and "Ngày nghe" in df.columns:
                    gh = row.get("Giờ nghe", "")
                    nn = row.get("Ngày nghe", "")
                    if pd.notna(gh) and pd.notna(nn):
                        time_info_html = f'<p style="color:var(--spotify-green); font-size:0.75rem; margin-top:4px;"> {gh} - {nn}</p>'
                
                with col:
                    # HTML phần hình và chữ
                    html_card = f"""<div class="song-card"><img src="{img_url}" alt="cover"><div><p class="song-title">{song_title}</p><p class="song-artist">{artist_name}</p>{time_info_html}</div></div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
                    
                    # Nút chức năng Streamlit xử lý trạng thái
                    st.button(
                        "▶ Play", 
                        key=f"{key_prefix}_play_{idx}_{safe_seed}", 
                        on_click=handle_play_song, 
                        args=(song_title, artist_name, img_url)
                    )


def render_bottom_player():
    """
    Hiển thị thanh Music Player thật với local MP3 ở dưới cùng màn hình bằng container cố định.
    """
    current_song = st.session_state.get('current_song', None)
    
    if current_song:
        # Sử dụng Anchor ID để "móc" vào div cha của Streamlit và ép fixed
        st.markdown('<div id="fixed-player-hook"></div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Thông tin bài hát (Trái)
                html_info = f"""
                <div style="display:flex; align-items:center; gap:12px;">
                    <img src="{current_song['image']}" style="width:52px; height:52px; border-radius:4px; box-shadow:0 4px 10px rgba(0,0,0,0.3);">
                    <div style="overflow:hidden;">
                        <h4 style="color:var(--text-main); margin:0; font-size:14px; white-space:nowrap; text-overflow:ellipsis;">{current_song['title']}</h4>
                        <p style="color:var(--text-sub); margin:0; font-size:12px; white-space:nowrap; text-overflow:ellipsis;">{current_song['artist']}</p>
                    </div>
                </div>
                """
                st.markdown(html_info, unsafe_allow_html=True)
            
            with col2:
                # Trình phát nhạc thực tế (Giữa)
                if current_song.get('audio_path'):
                    st.audio(current_song['audio_path'], autoplay=True)
                else:
                    st.info("⚠️ Đang chọn file nhạc...")
