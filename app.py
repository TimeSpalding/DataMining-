"""
Streamlit Demo — Module 1 Improved: User Persona Clustering
Đọc output từ Kaggle: user_classification_improved.parquet,
dashboard_data_improved.csv, association_rules_per_cluster.csv, metadata_improved.json
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, ast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="User Persona Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid #3a3f5c;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 8px;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(99,102,241,.3); }
.metric-card .value { font-size: 2.2rem; font-weight: 700; color: #818cf8; }
.metric-card .label { font-size: .85rem; color: #94a3b8; margin-top: 4px; }

.persona-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .5px;
}

.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #3a3f5c;
}

.rule-card {
    background: #1e2130;
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: .88rem;
}
.rule-card .ant { color: #fbbf24; font-weight: 600; }
.rule-card .cons { color: #34d399; font-weight: 600; }
.rule-card .metrics { color: #94a3b8; font-size: .78rem; margin-top: 4px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #1a1f35 100%);
    border-right: 1px solid #2d3158;
}
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = [
    "#818cf8", "#34d399", "#fbbf24", "#f87171",
    "#60a5fa", "#e879f9", "#fb7185"
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "results")

#  DATA LOADING 
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data(data_path: str):
    dashboard_path = os.path.join(data_path, "dashboard_data_improved.csv")
    rules_path     = os.path.join(data_path, "association_rules_per_cluster.csv")
    meta_path      = os.path.join(data_path, "metadata_improved.json")
    parquet_path   = os.path.join(data_path, "user_classification_improved.parquet")

    # Try parquet first, fallback to dashboard csv
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(dashboard_path):
        df = pd.read_csv(dashboard_path)
    else:
        df = _generate_demo_data()

    if os.path.exists(rules_path):
        rules_df = pd.read_csv(rules_path)
        if "antecedent" in rules_df.columns:
            rules_df["antecedent"] = rules_df["antecedent"].apply(
                lambda x: x.split("|") if isinstance(x, str) else x
            )
        if "consequent" in rules_df.columns:
            rules_df["consequent"] = rules_df["consequent"].apply(
                lambda x: x.split("|") if isinstance(x, str) else x
            )
    else:
        rules_df = pd.DataFrame()

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return df, rules_df, meta


def _generate_demo_data(n=5000, seed=42):
    """Sinh dữ liệu demo khi chưa chạy Kaggle."""
    rng = np.random.default_rng(seed)
    user_types = [
        "Super Night Owl Explorer", "Active Morning Loyalist",
        "Light Evening Moderate",   "Active Afternoon Explorer",
        "Super Morning Loyalist",
    ]
    cluster_ids = list(range(len(user_types)))
    clusters = rng.integers(0, len(user_types), size=n)

    df = pd.DataFrame({
        "user_id"          : [f"user_{i:05d}" for i in range(n)],
        "cluster"          : clusters,
        "user_type"        : [user_types[c] for c in clusters],
        "total_listens"    : rng.integers(50, 8000, n),
        "active_days"      : rng.integers(1, 180, n),
        "night_ratio"      : rng.uniform(0, 1, n),
        "morning_ratio"    : rng.uniform(0, 1, n),
        "afternoon_ratio"  : rng.uniform(0, 1, n),
        "evening_ratio"    : rng.uniform(0, 1, n),
        "artist_entropy"   : rng.uniform(0, 5, n),
        "time_entropy"     : rng.uniform(0, 4, n),
        "avg_listen_hour"  : rng.uniform(0, 23, n),
        "hour_std"         : rng.uniform(0, 8, n),
    })
    # normalize time ratios
    total_ratio = df[["night_ratio","morning_ratio","afternoon_ratio","evening_ratio"]].sum(axis=1) + 1e-9
    for c in ["night_ratio","morning_ratio","afternoon_ratio","evening_ratio"]:
        df[c] = df[c] / total_ratio
    return df


#  SIDEBAR 
with st.sidebar:
    st.markdown("## 🎵 User Persona")
    st.markdown("**Module 1**")
    st.divider()

    data_path = st.text_input(
        "📁 Thư mục chứa output",
        value=DATA_DIR,
        help="Copy từ /kaggle/working về local"
    )

    st.divider()
    page = st.radio(
        "Điều hướng",
        ["📊 Tổng quan", "🗺️ Phân cụm PCA", "👤 Hồ sơ Persona",
         "🔗 Association Rules", "🔍 Tra cứu User"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Shannon Entropy · PCA · K-Means · FP-Growth")

#  LOAD 
df, rules_df, meta = load_data(data_path)

is_demo = meta == {}
if is_demo:
    st.warning("⚠️ Đang dùng **dữ liệu demo** — copy output từ Kaggle vào thư mục trên để xem kết quả thật.", icon="🗂️")

n_clusters  = df["cluster"].nunique()
cluster_ids = sorted(df["cluster"].unique())
user_types  = df.groupby("cluster")["user_type"].first().to_dict() if "user_type" in df.columns else {c: f"Cluster {c}" for c in cluster_ids}
color_map   = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cluster_ids)}

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — TỔNG QUAN
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Tổng quan":
    st.markdown("# 📊 Tổng quan Kết quả Phân Cụm")
    st.markdown("<hr style='border:1px solid #2d3158;margin-bottom:24px'>", unsafe_allow_html=True)

    # Metrics row
    sil  = meta.get("silhouette_score", df.get("silhouette", [None])[0] if "silhouette" in df.columns else None)
    pca_k = meta.get("pca_components", "—")
    pca_v = meta.get("pca_explained_variance", None)
    n_rules = meta.get("total_association_rules", len(rules_df) if not rules_df.empty else "—")

    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (len(df),    "Tổng người dùng"),
        (n_clusters, "Số cụm (K)"),
        (f"{sil:.3f}" if sil else "—", "Silhouette Score"),
        (f"{pca_k}D", "PCA chiều"),
        (f"{pca_v*100:.1f}%" if pca_v else "—", "Variance giữ lại"),
    ]
    for col, (val, lbl) in zip([col1,col2,col3,col4,col5], cards):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Cluster distribution
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown('<p class="section-header">Phân phối User theo Cluster</p>', unsafe_allow_html=True)
        dist = df.groupby(["cluster","user_type"]).size().reset_index(name="count") if "user_type" in df.columns \
               else df.groupby("cluster").size().reset_index(name="count")
        dist["label"] = dist.apply(lambda r: f"C{r['cluster']}: {r.get('user_type','')[:20] if 'user_type' in r.index else ''}", axis=1)
        fig = px.pie(
            dist, values="count", names="label",
            color_discrete_sequence=CLUSTER_COLORS,
            hole=0.52,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend=dict(font_size=11),
            margin=dict(t=10,b=10,l=10,r=10)
        )
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Phân phối Đặc trưng theo Cluster</p>', unsafe_allow_html=True)
        feat_sel = st.selectbox(
            "Chọn feature",
            ["total_listens","active_days","artist_entropy","time_entropy","avg_listen_hour"],
            key="overview_feat"
        )
        fig2 = px.box(
            df, x="cluster", y=feat_sel,
            color="cluster",
            color_discrete_sequence=CLUSTER_COLORS,
            labels={"cluster":"Cluster", feat_sel: feat_sel},
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=False,
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Shannon entropy heatmap per cluster
    st.markdown('<p class="section-header">Shannon Entropy theo Cụm</p>', unsafe_allow_html=True)
    ent_cols = [c for c in ["artist_entropy","time_entropy"] if c in df.columns]
    if ent_cols:
        ent_agg = df.groupby("cluster")[ent_cols].mean().reset_index()
        fig3 = go.Figure()
        colors_bar = ["#818cf8","#34d399"]
        for i, ec in enumerate(ent_cols):
            fig3.add_trace(go.Bar(
                x=[f"C{c}" for c in ent_agg["cluster"]],
                y=ent_agg[ec], name=ec,
                marker_color=colors_bar[i], opacity=0.85
            ))
        fig3.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend=dict(orientation="h", y=1.1),
            yaxis_title="H (bits) — Shannon Entropy",
            margin=dict(t=10,b=40,l=40,r=10), height=300
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Improvements summary
    if meta.get("improvements_v2"):
        st.markdown('<p class="section-header">3 Cải tiến </p>', unsafe_allow_html=True)
        for imp in meta["improvements_v2"]:
            st.markdown(f" {imp}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — PCA SCATTER
# ═══════════════════════════════════════════════════════════════════
elif page == "🗺️ Phân cụm PCA":
    st.markdown("# 🗺️ Phân cụm trong PCA Space")

    pca_cols_2d = [c for c in df.columns if c.startswith("pc")]
    have_pca = len(pca_cols_2d) >= 2

    if not have_pca:
        # Fake PCA từ features nếu không có cột pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        feat_for_pca = [c for c in ["total_listens","active_days","artist_entropy","time_entropy",
                                     "night_ratio","morning_ratio","afternoon_ratio","evening_ratio",
                                     "avg_listen_hour","hour_std"] if c in df.columns]
        X = df[feat_for_pca].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(3, X_scaled.shape[1]))
        comps = pca.fit_transform(X_scaled)
        for i in range(comps.shape[1]):
            df[f"pc{i+1}"] = comps[:, i]
        pca_cols_2d = [f"pc{i+1}" for i in range(comps.shape[1])]
        var_explained = pca.explained_variance_ratio_.cumsum()
        st.info(f"ℹ️ PCA tính lại tạm thời — {var_explained[-1]*100:.1f}% variance với {len(pca_cols_2d)} chiều")

    sample_n = st.slider("Số điểm hiển thị", 500, min(10000, len(df)), min(3000, len(df)), step=500)
    sample_df = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df

    label_col = "user_type" if "user_type" in sample_df.columns else "cluster"
    sample_df = sample_df.copy()
    sample_df["_color"] = sample_df["cluster"].map(color_map)
    sample_df["_label"] = sample_df[label_col].astype(str)

    col3d, col2d = st.columns([1, 1])

    with col3d:
        if len(pca_cols_2d) >= 3:
            st.markdown("#### 🌐 Scatter 3D (PC1·PC2·PC3)")
            fig3d = px.scatter_3d(
                sample_df, x="pc1", y="pc2", z="pc3",
                color="_label",
                color_discrete_sequence=CLUSTER_COLORS,
                opacity=0.65, size_max=5,
                labels={"_label": "Persona"},
            )
            fig3d.update_traces(marker=dict(size=3))
            fig3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    bgcolor="rgba(15,17,23,1)",
                    xaxis=dict(showbackground=False, color="#e2e8f0"),
                    yaxis=dict(showbackground=False, color="#e2e8f0"),
                    zaxis=dict(showbackground=False, color="#e2e8f0"),
                ),
                font_color="#e2e8f0",
                legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10,b=0,l=0,r=0), height=440
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Cần ≥3 chiều PCA để vẽ 3D")

    with col2d:
        st.markdown("#### 📐 Scatter 2D (PC1·PC2)")
        fig2d = px.scatter(
            sample_df, x="pc1", y="pc2",
            color="_label",
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.6, size_max=6,
            labels={"_label": "Persona"},
        )
        fig2d.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig2d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,34,1)",
            font_color="#e2e8f0",
            xaxis=dict(showgrid=True, gridcolor="#2d3158"),
            yaxis=dict(showgrid=True, gridcolor="#2d3158"),
            legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10,b=40,l=40,r=10), height=440
        )
        st.plotly_chart(fig2d, use_container_width=True)

    # Cluster centers
    st.markdown("#### 📍 Tâm cụm trong PCA Space")
    centers = sample_df.groupby("cluster")[pca_cols_2d[:2]].mean().reset_index()
    centers["user_type"] = centers["cluster"].map(user_types)
    st.dataframe(
        centers.rename(columns={"cluster":"Cluster","user_type":"Persona"}),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — HỒ SƠ PERSONA
# ═══════════════════════════════════════════════════════════════════
elif page == "👤 Hồ sơ Persona":
    st.markdown("# 👤 Hồ sơ Chi tiết Từng Persona")

    cluster_profile = df.groupby("cluster").agg(
        user_count       = ("user_id" if "user_id" in df.columns else "cluster", "count"),
        avg_total_listens= ("total_listens","mean"),
        avg_active_days  = ("active_days","mean"),
        avg_artist_entropy=("artist_entropy","mean"),
        avg_time_entropy = ("time_entropy","mean"),
        avg_night        = ("night_ratio","mean"),
        avg_morning      = ("morning_ratio","mean"),
        avg_afternoon    = ("afternoon_ratio","mean"),
        avg_evening      = ("evening_ratio","mean"),
        avg_hour         = ("avg_listen_hour","mean"),
    ).reset_index()
    cluster_profile["user_type"] = cluster_profile["cluster"].map(user_types)

    # Radar chart for all clusters
    st.markdown('<p class="section-header">Radar Chart — So sánh tất cả Persona</p>', unsafe_allow_html=True)
    radar_features = ["avg_total_listens","avg_active_days","avg_artist_entropy",
                      "avg_time_entropy","avg_night","avg_morning","avg_afternoon","avg_evening"]
    radar_labels   = ["Listens","Active Days","Artist Entropy","Time Entropy",
                      "Night","Morning","Afternoon","Evening"]

    # Normalize each feature 0→1
    cp_norm = cluster_profile[radar_features].copy()
    for c in radar_features:
        rng_ = cp_norm[c].max() - cp_norm[c].min()
        cp_norm[c] = (cp_norm[c] - cp_norm[c].min()) / (rng_ + 1e-9)

    fig_radar = go.Figure()
    for i, row in cluster_profile.iterrows():
        vals = cp_norm.iloc[i][radar_features].tolist()
        vals += [vals[0]]  # close the loop
        lbls  = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=lbls,
            fill="toself", opacity=0.45,
            name=f"C{row['cluster']}: {str(row.get('user_type',''))[:20]}",
            line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2),
            fillcolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(20,22,34,1)",
            radialaxis=dict(visible=True, showticklabels=False, gridcolor="#2d3158"),
            angularaxis=dict(gridcolor="#2d3158", color="#e2e8f0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20,b=20,l=40,r=40), height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Per-cluster detail cards
    st.markdown('<p class="section-header">Chi tiết Từng Cụm</p>', unsafe_allow_html=True)
    cols = st.columns(min(3, n_clusters))
    for i, row in cluster_profile.iterrows():
        c_idx = i % len(cols)
        with cols[c_idx]:
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            ut = str(row.get("user_type", f"Cluster {row['cluster']}"))
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e2130,#252840);
                        border:1px solid {color};border-radius:14px;padding:16px;margin-bottom:12px">
                <div style="font-size:1rem;font-weight:700;color:{color}">
                    Cluster {int(row['cluster'])}
                </div>
                <div style="font-size:.85rem;color:#94a3b8;margin:4px 0 12px">{ut}</div>
                <div style="display:flex;justify-content:space-between;font-size:.82rem;color:#e2e8f0">
                    <span>👥 {int(row['user_count']):,} users</span>
                    <span>📅 {row['avg_active_days']:.0f} days</span>
                </div>
                <div style="margin-top:8px;font-size:.82rem;color:#e2e8f0">
                    🎵 {row['avg_total_listens']:.0f} listens avg
                </div>
                <div style="margin-top:6px;font-size:.82rem;color:#818cf8">
                    H_artist={row['avg_artist_entropy']:.2f} bits &nbsp;|&nbsp;
                    H_time={row['avg_time_entropy']:.2f} bits
                </div>
                <div style="margin-top:8px;font-size:.78rem;color:#94a3b8">
                    🌙{row['avg_night']:.1%} ☀️{row['avg_morning']:.1%}
                    🌤️{row['avg_afternoon']:.1%} 🌆{row['avg_evening']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Time pattern line chart
    st.markdown('<p class="section-header">Phân bố Giờ Nghe theo Persona</p>', unsafe_allow_html=True)
    time_data = []
    for _, row in cluster_profile.iterrows():
        for period, val in [("Đêm 22-4h",row["avg_night"]),("Sáng 5-11h",row["avg_morning"]),
                             ("Chiều 12-17h",row["avg_afternoon"]),("Tối 18-21h",row["avg_evening"])]:
            time_data.append({"Cluster": f"C{int(row['cluster'])}: {str(row.get('user_type',''))[:16]}",
                               "Period": period, "Ratio": val})
    time_df_ = pd.DataFrame(time_data)
    fig_time = px.line(
        time_df_, x="Period", y="Ratio", color="Cluster",
        markers=True, color_discrete_sequence=CLUSTER_COLORS,
        labels={"Ratio":"Tỷ lệ thời gian nghe"},
    )
    fig_time.update_traces(line_width=2.5, marker_size=9)
    fig_time.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,34,1)",
        font_color="#e2e8f0",
        xaxis=dict(showgrid=True, gridcolor="#2d3158"),
        yaxis=dict(showgrid=True, gridcolor="#2d3158", tickformat=".0%"),
        legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10,b=40,l=60,r=10), height=360
    )
    st.plotly_chart(fig_time, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.markdown("# 🔗 FP-Growth Association Rules (Per Cluster)")

    if rules_df.empty:
        st.error("Chưa có file `association_rules_per_cluster.csv`. Chạy notebook trên Kaggle và copy output về.")
        st.stop()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><div class="value">{len(rules_df)}</div><div class="label">Tổng số luật</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><div class="value">{rules_df["cluster"].nunique()}</div><div class="label">Cụm có luật</div></div>', unsafe_allow_html=True)
    with col3:
        avg_lift = rules_df["lift"].mean() if "lift" in rules_df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="value">{avg_lift:.2f}x</div><div class="label">Lift trung bình</div></div>', unsafe_allow_html=True)
    with col4:
        max_lift = rules_df["lift"].max() if "lift" in rules_df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="value">{max_lift:.2f}x</div><div class="label">Lift cao nhất</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Filter
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_clusters = st.multiselect(
            "Lọc theo Cluster",
            options=sorted(rules_df["cluster"].unique()),
            default=list(sorted(rules_df["cluster"].unique())),
            format_func=lambda x: f"C{x}: {user_types.get(x, f'Cluster {x}')[:25]}"
        )
    with col_f2:
        min_lift = st.slider("Lift tối thiểu", 1.0, float(rules_df["lift"].max()), 1.0, 0.1)
    with col_f3:
        min_conf = st.slider("Confidence tối thiểu", 0.0, 1.0, 0.5, 0.05)

    filtered = rules_df[
        (rules_df["cluster"].isin(selected_clusters)) &
        (rules_df["lift"] >= min_lift) &
        (rules_df["confidence"] >= min_conf)
    ].sort_values("lift", ascending=False)

    st.caption(f"Hiển thị {len(filtered)} / {len(rules_df)} luật")

    # Bubble scatter — confidence vs lift
    st.markdown('<p class="section-header">Bản đồ Luật: Confidence vs Lift</p>', unsafe_allow_html=True)
    if len(filtered) > 0:
        plot_df = filtered.copy()
        plot_df["antecedent_str"] = plot_df["antecedent"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
        plot_df["consequent_str"] = plot_df["consequent"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
        plot_df["rule_str"] = plot_df["antecedent_str"] + " → " + plot_df["consequent_str"]
        plot_df["cluster_str"] = plot_df["cluster"].apply(lambda x: f"C{x}: {user_types.get(x,str(x))[:20]}")

        fig_bubble = px.scatter(
            plot_df, x="confidence", y="lift",
            size="support", color="cluster_str",
            hover_name="rule_str",
            hover_data={"confidence":":.2f","lift":":.2f","support":":.2%"},
            color_discrete_sequence=CLUSTER_COLORS,
            size_max=25,
            labels={"confidence":"Confidence","lift":"Lift","cluster_str":"Cluster"},
        )
        fig_bubble.add_hline(y=1.0, line_dash="dash", line_color="#f87171", opacity=0.7,
                             annotation_text="Lift = 1 (baseline)", annotation_position="bottom right")
        fig_bubble.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,34,1)",
            font_color="#e2e8f0",
            xaxis=dict(showgrid=True, gridcolor="#2d3158", tickformat=".0%"),
            yaxis=dict(showgrid=True, gridcolor="#2d3158"),
            legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10,b=40,l=50,r=10), height=400
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Per-cluster bar chart
    st.markdown('<p class="section-header">Top 5 Luật mỗi Cụm (Lift cao nhất)</p>', unsafe_allow_html=True)
    for c_id in selected_clusters:
        c_rules = filtered[filtered["cluster"] == c_id].head(5)
        if c_rules.empty:
            continue
        color = color_map.get(c_id, "#818cf8")
        st.markdown(f"**Cluster {c_id} — {user_types.get(c_id, '')[:35]}**")
        for _, row in c_rules.iterrows():
            ant  = ", ".join(row["antecedent"]) if isinstance(row["antecedent"], list) else str(row["antecedent"])
            cons = ", ".join(row["consequent"]) if isinstance(row["consequent"], list) else str(row["consequent"])
            st.markdown(f"""<div class="rule-card">
                <span class="ant">[{ant}]</span>
                &nbsp;→&nbsp;
                <span class="cons">[{cons}]</span>
                <div class="metrics">
                    Confidence: {row['confidence']:.0%} &nbsp;|&nbsp;
                    Lift: <b style="color:#fbbf24">{row['lift']:.2f}x</b> &nbsp;|&nbsp;
                    Support: {row['support']:.1%}
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

    # Raw table
    with st.expander("📋 Xem toàn bộ bảng luật"):
        show_df = filtered.copy()
        show_df["antecedent"] = show_df["antecedent"].apply(lambda x: " + ".join(x) if isinstance(x, list) else str(x))
        show_df["consequent"] = show_df["consequent"].apply(lambda x: " + ".join(x) if isinstance(x, list) else str(x))
        show_df["cluster"]    = show_df["cluster"].apply(lambda x: f"C{x}: {user_types.get(x,str(x))[:20]}")
        st.dataframe(
            show_df[["cluster","antecedent","consequent","confidence","lift","support"]]
              .rename(columns={"cluster":"Cluster","antecedent":"Antecedent",
                               "consequent":"Consequent","confidence":"Confidence",
                               "lift":"Lift","support":"Support"})
              .style.format({"Confidence":"{:.0%}","Lift":"{:.2f}","Support":"{:.1%}"}),
            use_container_width=True, height=400
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — TRA CỨU USER
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 Tra cứu User":
    st.markdown("# 🔍 Tra cứu Người Dùng")

    if "user_id" not in df.columns:
        st.error("Cột `user_id` không có trong dataset.")
        st.stop()

    col_s, col_rand = st.columns([3, 1])
    with col_s:
        uid_input = st.text_input("Nhập User ID", placeholder="user_00001")
    with col_rand:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Ngẫu nhiên"):
            uid_input = str(df["user_id"].sample(1).values[0])
            st.rerun()

    if uid_input:
        row = df[df["user_id"].astype(str) == uid_input.strip()]
        if row.empty:
            # partial match
            row = df[df["user_id"].astype(str).str.contains(uid_input.strip(), regex=False, na=False)]

        if row.empty:
            st.warning(f"Không tìm thấy user `{uid_input}`")
        else:
            r = row.iloc[0]
            cluster_id = int(r["cluster"])
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
            persona = str(r.get("user_type", f"Cluster {cluster_id}"))

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e2130,#252840);
                        border:2px solid {color};border-radius:18px;padding:24px;margin:16px 0">
                <div style="display:flex;align-items:center;gap:16px">
                    <div style="width:56px;height:56px;border-radius:50%;background:{color};
                                display:flex;align-items:center;justify-content:center;
                                font-size:1.6rem;flex-shrink:0">🎵</div>
                    <div>
                        <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0">{r['user_id']}</div>
                        <div style="font-size:.9rem;color:{color};font-weight:600;margin-top:2px">{persona}</div>
                        <div style="font-size:.8rem;color:#94a3b8;margin-top:2px">Cluster {cluster_id}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("🎵", "Tổng lượt nghe", f"{int(r.get('total_listens',0)):,}"),
                ("📅", "Active days",    f"{int(r.get('active_days',0))} ngày"),
                ("🎨", "Artist Entropy", f"{r.get('artist_entropy',0):.2f} bits"),
                ("⏰", "Time Entropy",   f"{r.get('time_entropy',0):.2f} bits"),
            ]
            for col_, (icon, lbl, val) in zip([c1,c2,c3,c4], metrics):
                with col_:
                    st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.4rem">{icon} {val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

            # Time distribution gauge
            st.markdown("#### ⏱️ Phân bố thời gian nghe")
            time_vals = {
                "🌙 Đêm (22-4h)"   : float(r.get("night_ratio",0)),
                "☀️ Sáng (5-11h)"  : float(r.get("morning_ratio",0)),
                "🌤️ Chiều (12-17h)": float(r.get("afternoon_ratio",0)),
                "🌆 Tối (18-21h)"  : float(r.get("evening_ratio",0)),
            }
            fig_time_u = go.Figure(go.Bar(
                x=list(time_vals.keys()), y=list(time_vals.values()),
                marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(4)],
                text=[f"{v:.1%}" for v in time_vals.values()], textposition="outside"
            ))
            fig_time_u.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,34,1)",
                font_color="#e2e8f0",
                yaxis=dict(tickformat=".0%", gridcolor="#2d3158"),
                xaxis=dict(gridcolor="#2d3158"),
                margin=dict(t=20,b=20,l=40,r=10), height=280, showlegend=False
            )
            st.plotly_chart(fig_time_u, use_container_width=True)

            # Show rules for this cluster
            if not rules_df.empty and "cluster" in rules_df.columns:
                c_rules = rules_df[rules_df["cluster"] == cluster_id].sort_values("lift", ascending=False).head(5)
                if not c_rules.empty:
                    st.markdown(f"#### 🔗 Association Rules áp dụng cho persona này")
                    for _, rl in c_rules.iterrows():
                        ant  = ", ".join(rl["antecedent"]) if isinstance(rl["antecedent"], list) else str(rl["antecedent"])
                        cons = ", ".join(rl["consequent"]) if isinstance(rl["consequent"], list) else str(rl["consequent"])
                        st.markdown(f"""<div class="rule-card">
                            <span class="ant">[{ant}]</span> → <span class="cons">[{cons}]</span>
                            <div class="metrics">Conf: {rl['confidence']:.0%} | Lift: <b style="color:#fbbf24">{rl['lift']:.2f}x</b></div>
                        </div>""", unsafe_allow_html=True)

            if len(row) > 1:
                st.info(f"Tìm thấy {len(row)} user khớp — hiển thị user đầu tiên.")
