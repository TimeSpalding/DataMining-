import streamlit as st
import pandas as pd
from src.ui.components import render_section_header

def render_context_tab(rec_sys, user_input, n_recs):
    tab1, = st.tabs([
        "Tạo Playlist", 
    ])

    # --- TAB: PLAYLIST GENERATOR (Bản chọn Ca sĩ + Fix lỗi) ---
    with tab1:
        render_section_header("Playlist Generator", 
                            subtitle="Sinh danh sách phát dựa trên bài hát hoặc nghệ sĩ yêu thích.",
                            icon_name="radio",
                            color="#f43f5e")
        st.markdown("Nhập tên bài hát hoặc ca sĩ. Hệ thống sẽ liệt kê các phiên bản để bạn chọn.")
        
        seed_query = st.text_input("Nhập tên bài hoặc ca sĩ:", placeholder="Ví dụ: Joji, Do For You...", key="seed_q")

        if seed_query:
            # ... (Phần logic tìm kiếm giữ nguyên)
            search_results = rec_sys.search_metadata(seed_query, n=15)
            
            if search_results.empty:
                st.error(f"Không tìm thấy kết quả nào khớp với '{seed_query}'. Vui lòng thử từ khóa khác.")
            else:
                st.success(f"Tìm thấy các kết quả liên quan:")
                options = [f"{row['track_name']} - {row['artist_name']}" for _, row in search_results.iterrows()]
                options = list(dict.fromkeys(options))
                selected_version = st.selectbox("Chọn chính xác phiên bản bạn muốn làm mồi (Seed):", options)
                
                col_opts = st.columns(5)
                with col_opts[0]:
                    exclude_h2 = st.checkbox("Không hiện lại bài cũ", value=False, key="excl_tab2")
                with col_opts[4]:
                    refresh2 = st.button("Làm mới", key="refresh_tab2", use_container_width=True)

                if st.button("Sinh Playlist", use_container_width=True) or refresh2:
                    chosen_track = selected_version.split(" - ")[0]
                    with st.spinner(f"Đang sinh nhạc dựa trên: {selected_version}..."):
                        df_pool = rec_sys.generate_playlist(user_input, seed_track_names=[chosen_track], n_songs=n_recs * 3)
                        if not df_pool.empty:
                            df = rec_sys._apply_refresh_logic(df_pool, n=n_recs, user_id_str=user_input, exclude_history=exclude_h2)
                            st.session_state['context_playlist'] = df
                        else:
                            st.session_state['context_playlist'] = None
                            st.warning("Không tìm thấy bài hát tương đồng đủ tốt.")
                
                # Hiển thị kết quả từ session_state để duy trì khi rerun
                if st.session_state.get('context_playlist') is not None:
                    from src.ui.components import render_song_cards
                    render_song_cards(st.session_state['context_playlist'], key_prefix="ctx_playlist")

    # --- TAB 3: INTERACTIVE SESSION ---
    #with tab3:
    #    from interactive_tab import render_interactive_tab
    #    render_interactive_tab(rec_sys, user_input, n_recs, db_path="./mappings.db")