import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLNAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "last_flag",
]

FEATURE_COLS = [
    "src_bytes", "diff_srv_rate", "same_srv_rate", "dst_bytes",
    "dst_host_diff_srv_rate", "dst_host_srv_count", "count",
    "dst_host_serror_rate", "dst_host_same_srv_rate", "serror_rate",
    "flag_SF", "dst_host_srv_serror_rate", "srv_serror_rate", "flag_S0",
    "logged_in", "dst_host_srv_diff_host_rate", "dst_host_same_src_port_rate",
    "dst_host_count", "service_http", "srv_count", "srv_diff_host_rate",
    "service_private", "dst_host_rerror_rate",
]

OHE_FEATURES = {"flag_SF", "flag_S0", "service_http", "service_private"}
NUM_FEATURES  = [f for f in FEATURE_COLS if f not in OHE_FEATURES]

ALL_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r", "other"]
_le = LabelEncoder()
_le.fit(ALL_CATEGORIES)
LABEL_MAP = {int(code): cls for code, cls in zip(_le.transform(_le.classes_), _le.classes_)}

LABEL_COLORS = {
    "normal": "#22c55e",
    "dos":    "#ef4444",
    "probe":  "#f59e0b",
    "r2l":    "#8b5cf6",
    "u2r":    "#ec4899",
    "other":  "#6b7280",
}

LABEL_THREAT = {
    "normal": ("CLEAR",    "✅", "#22c55e"),
    "dos":    ("CRITICAL", "🔴", "#ef4444"),
    "probe":  ("HIGH",     "🟠", "#f59e0b"),
    "r2l":    ("HIGH",     "🟠", "#8b5cf6"),
    "u2r":    ("CRITICAL", "🔴", "#ec4899"),
    "other":  ("MEDIUM",   "🟡", "#6b7280"),
}

LABEL_DESCRIPTIONS = {
    "normal": "Normal traffic — no intrusion detected.",
    "dos":    "Denial of Service — resource exhaustion attempt.",
    "probe":  "Probe / Surveillance — scanning for vulnerabilities.",
    "r2l":    "Remote-to-Local — unauthorised remote access attempt.",
    "u2r":    "User-to-Root — privilege escalation attempt.",
    "other":  "Unclassified / unknown attack type.",
}

MODEL_ACCURACY = {
    "Random Forest": ("75.37%", "0.88", "0.70"),
    "XGBoost":       ("76.45%", "0.90", "0.71"),
    "MLP":           ("73.03%", "0.86", "0.58"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Resource loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising models…")
def load_resources():
    with open("random_forest_model.pkl", "rb") as f: rf  = pickle.load(f)
    with open("xgboost_model.pkl",       "rb") as f: xgb = pickle.load(f)
    with open("mlp_model.pkl",           "rb") as f: mlp = pickle.load(f)
    train_df = pd.read_csv("KDDTrain+.txt", names=COLNAMES)
    scaler = StandardScaler()
    scaler.fit(train_df[NUM_FEATURES])
    return {"rf": rf, "xgb": xgb, "mlp": mlp, "scaler": scaler}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(values: dict) -> dict:
    flag    = str(values.get("flag",    "OTHER"))
    service = str(values.get("service", "other"))
    row = {col: (0.0 if col in OHE_FEATURES else float(values.get(col, 0.0)))
           for col in FEATURE_COLS}
    row["flag_SF"]         = 1.0 if flag    == "SF"      else 0.0
    row["flag_S0"]         = 1.0 if flag    == "S0"      else 0.0
    row["service_http"]    = 1.0 if service == "http"    else 0.0
    row["service_private"] = 1.0 if service == "private" else 0.0
    return row

def preprocess(rows: list[dict], scaler: StandardScaler) -> pd.DataFrame:
    df = pd.DataFrame([build_feature_row(r) for r in rows])[FEATURE_COLS]
    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])
    return df

def predict_single(df_row: pd.DataFrame, model):
    proba   = model.predict_proba(df_row)[0]
    classes = model.classes_
    label   = LABEL_MAP.get(int(classes[int(np.argmax(proba))]), "unknown")
    proba_dict = {LABEL_MAP.get(int(c), str(c)): float(p) for c, p in zip(classes, proba)}
    return label, proba_dict

def preprocess_raw_csv(df_raw: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return preprocess(df_raw.to_dict(orient="records"), scaler)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:       #0a0f1e;
  --bg2:      #0d1628;
  --card:     #111827;
  --border:   #1e3a5f;
  --cyan:     #00d4ff;
  --cyan-dim: rgba(0,212,255,0.12);
  --text:     #e2e8f0;
  --muted:    #64748b;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.block-container            { padding: 1.5rem 2rem 0 2rem; max-width: 1400px; }

[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--card); border-radius: 10px;
  padding: 4px; gap: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.9rem; border-radius: 7px;
  padding: 8px 20px; color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--cyan-dim) !important;
  color: var(--cyan) !important;
  border-bottom: none !important;
}

.stButton > button, [data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #0077a8, #00d4ff) !important;
  color: #0a0f1e !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 8px !important;
  letter-spacing: 0.05em;
}
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan) !important;
  font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.85rem !important;
}

.stat-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
}
.stat-card .label {
  font-size: 0.7rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
  font-family: 'Share Tech Mono', monospace;
}
.stat-card .value {
  font-size: 1.5rem; font-weight: 700;
  font-family: 'Share Tech Mono', monospace; color: var(--cyan);
}

.threat-badge {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 12px 24px; border-radius: 12px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 1.3rem; font-weight: 700;
  letter-spacing: 0.08em; border: 2px solid; margin-bottom: 0.5rem;
}

.count-card {
  background: var(--card); border-left: 3px solid;
  border-radius: 0 10px 10px 0; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
}
.count-card .num { font-size: 1.8rem; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
.count-card .lbl { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }

.section-header {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--cyan);
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px; margin: 1.2rem 0 0.8rem 0;
}

.app-footer {
  margin-top: 3rem; padding: 1.5rem 0;
  border-top: 1px solid var(--border);
  display: flex; justify-content: space-between;
  align-items: center; flex-wrap: wrap; gap: 0.5rem;
  font-size: 0.8rem; color: var(--muted);
  font-family: 'Share Tech Mono', monospace;
}
.app-footer a { color: var(--cyan); text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.25rem;">
  <span style="font-size:2.4rem;">🛡️</span>
  <div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:1.7rem;
                color:#e2e8f0;letter-spacing:0.04em;line-height:1.1;">
      NETWORK INTRUSION DETECTION SYSTEM
    </div>
    <div style="font-size:0.85rem;color:#64748b;margin-top:2px;">
      ML-based classification of network traffic &nbsp;·&nbsp; NSL-KDD Dataset
      &nbsp;·&nbsp; 3 models &nbsp;·&nbsp; 23 features
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-header">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Active model", ["Random Forest", "XGBoost", "MLP"],
        index=1, label_visibility="collapsed",
        help="XGBoost achieved the highest test accuracy at 76.45%.",
    )

    acc, f1_dos, f1_probe = MODEL_ACCURACY[model_choice]
    st.markdown(f"""
    <div class="stat-card">
      <div class="label">Test Accuracy</div>
      <div class="value">{acc}</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:0.75rem;">
      <div class="stat-card" style="margin-bottom:0">
        <div class="label">DoS F1</div>
        <div class="value" style="font-size:1.1rem">{f1_dos}</div>
      </div>
      <div class="stat-card" style="margin-bottom:0">
        <div class="label">Probe F1</div>
        <div class="value" style="font-size:1.1rem">{f1_probe}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Threat Levels</div>', unsafe_allow_html=True)
    for lbl, (threat, icon, color) in LABEL_THREAT.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 0;border-bottom:1px solid #1e2d40;">'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#e2e8f0">'
            f'{lbl.upper()}</span>'
            f'<span style="font-size:0.7rem;font-family:Share Tech Mono,monospace;color:{color};'
            f'border:1px solid {color};padding:1px 8px;border-radius:4px;">{threat}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.9;">
      <div><span style="color:#64748b">Dataset &nbsp;</span>NSL-KDD</div>
      <div><span style="color:#64748b">Features</span> 23 / 41 (MI &gt; 0.1)</div>
      <div><span style="color:#64748b">Classes &nbsp;</span>6 categories</div>
      <div><span style="color:#64748b">Stack &nbsp;&nbsp;&nbsp;</span>Streamlit · Plotly · sklearn</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Team</div>', unsafe_allow_html=True)
    for name, role, color in [
        ("Evans",        "Deployment & EDA",          "#00d4ff"),
        ("Akash S",      "Model Development",          "#a78bfa"),
        ("Laniya Mohan", "Preprocessing & Features",   "#34d399"),
    ]:
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid #1e2d40;">'
            f'<div style="font-weight:600;font-size:0.85rem;color:{color}">{name}</div>'
            f'<div style="font-size:0.72rem;color:#64748b">{role}</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────

res    = load_resources()
model  = {"Random Forest": res["rf"], "XGBoost": res["xgb"], "MLP": res["mlp"]}[model_choice]
scaler = res["scaler"]


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_manual, tab_batch = st.tabs(["🔍  Single Connection Analysis", "📂  Batch CSV Classification"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single connection
# ════════════════════════════════════════════════════════════════════════════

with tab_manual:
    st.markdown('<div class="section-header">Connection Parameters</div>', unsafe_allow_html=True)
    st.caption("Enter the network connection features below. Only the 23 features used during training are shown.")

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Traffic Volume**")
            src_bytes          = st.number_input("src_bytes",          min_value=0, value=0,   help="Bytes from source → dest")
            dst_bytes          = st.number_input("dst_bytes",          min_value=0, value=0,   help="Bytes from dest → source")
            count              = st.number_input("count",              min_value=0, max_value=511, value=1, help="Connections to same host (last 2s)")
            srv_count          = st.number_input("srv_count",          min_value=0, max_value=511, value=1, help="Connections to same service (last 2s)")
            dst_host_count     = st.number_input("dst_host_count",     min_value=0, max_value=255, value=1)
            dst_host_srv_count = st.number_input("dst_host_srv_count", min_value=0, max_value=255, value=1)

        with c2:
            st.markdown("**Error Rates**")
            serror_rate              = st.slider("serror_rate",              0.0, 1.0, 0.0, 0.01)
            srv_serror_rate          = st.slider("srv_serror_rate",          0.0, 1.0, 0.0, 0.01)
            dst_host_serror_rate     = st.slider("dst_host_serror_rate",     0.0, 1.0, 0.0, 0.01)
            dst_host_srv_serror_rate = st.slider("dst_host_srv_serror_rate", 0.0, 1.0, 0.0, 0.01)
            rerror_rate_val          = st.slider("dst_host_rerror_rate",     0.0, 1.0, 0.0, 0.01)
            same_srv_rate            = st.slider("same_srv_rate",            0.0, 1.0, 1.0, 0.01)
            diff_srv_rate            = st.slider("diff_srv_rate",            0.0, 1.0, 0.0, 0.01)

        with c3:
            st.markdown("**Host Rates**")
            dst_host_same_srv_rate      = st.slider("dst_host_same_srv_rate",      0.0, 1.0, 1.0, 0.01)
            dst_host_diff_srv_rate      = st.slider("dst_host_diff_srv_rate",      0.0, 1.0, 0.0, 0.01)
            dst_host_same_src_port_rate = st.slider("dst_host_same_src_port_rate", 0.0, 1.0, 0.0, 0.01)
            dst_host_srv_diff_host_rate = st.slider("dst_host_srv_diff_host_rate", 0.0, 1.0, 0.0, 0.01)
            srv_diff_host_rate          = st.slider("srv_diff_host_rate",          0.0, 1.0, 0.0, 0.01)

        st.markdown("---")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            flag = st.selectbox("TCP Flag", ["SF","S0","REJ","RSTO","SH","RSTR","S1","S2","S3","OTH"],
                                help="SF = normal close · S0 = no response (common in DoS)")
        with cc2:
            service = st.selectbox("Service", ["http","private","ftp_data","smtp",
                                               "ftp","ssh","domain_u","ecr_i","other"])
        with cc3:
            logged_in = st.selectbox("Logged In", [0, 1], help="1 = successful login")

        submitted = st.form_submit_button("⚡  RUN CLASSIFICATION", use_container_width=True, type="primary")

    if submitted:
        row = dict(
            src_bytes=src_bytes, dst_bytes=dst_bytes,
            count=count, srv_count=srv_count,
            dst_host_count=dst_host_count, dst_host_srv_count=dst_host_srv_count,
            serror_rate=serror_rate, srv_serror_rate=srv_serror_rate,
            dst_host_rerror_rate=rerror_rate_val,
            dst_host_serror_rate=dst_host_serror_rate,
            dst_host_srv_serror_rate=dst_host_srv_serror_rate,
            same_srv_rate=same_srv_rate, diff_srv_rate=diff_srv_rate,
            dst_host_same_srv_rate=dst_host_same_srv_rate,
            dst_host_diff_srv_rate=dst_host_diff_srv_rate,
            dst_host_same_src_port_rate=dst_host_same_src_port_rate,
            dst_host_srv_diff_host_rate=dst_host_srv_diff_host_rate,
            srv_diff_host_rate=srv_diff_host_rate,
            logged_in=logged_in, flag=flag, service=service,
        )
        df_input = preprocess([row], scaler)
        label, proba = predict_single(df_input, model)
        threat, icon, color = LABEL_THREAT[label]
        confidence = proba[label] * 100
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

        st.markdown("---")
        st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.2, 1.2, 2.5])
        with r1:
            st.markdown(
                f'<div class="threat-badge" style="color:{color};border-color:{color};'
                f'background:rgba({r},{g},{b},0.1)">{icon} {label.upper()}</div>'
                f'<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;'
                f'color:#64748b;margin-top:4px;">TRAFFIC CATEGORY</div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="threat-badge" style="color:{color};border-color:{color};'
                f'background:rgba({r},{g},{b},0.1)">{threat}</div>'
                f'<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;'
                f'color:#64748b;margin-top:4px;">THREAT LEVEL</div>'
                f'<div style="font-size:0.85rem;color:#94a3b8;margin-top:8px;">'
                f'Confidence: <span style="color:{color};font-weight:700">{confidence:.1f}%</span><br>'
                f'<span style="font-size:0.8rem;color:#64748b">{LABEL_DESCRIPTIONS[label]}</span></div>',
                unsafe_allow_html=True,
            )
        with r3:
            fig = go.Figure(go.Bar(
                x=list(proba.keys()),
                y=[v * 100 for v in proba.values()],
                marker_color=[LABEL_COLORS.get(l, "#6b7280") for l in proba.keys()],
                text=[f"{v*100:.1f}%" for v in proba.values()],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=11),
            ))
            fig.update_layout(
                yaxis=dict(title="Confidence (%)", range=[0,115],
                           gridcolor="#1e2d40", tickfont=dict(family="Share Tech Mono")),
                xaxis=dict(tickfont=dict(family="Share Tech Mono")),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10,b=10,l=0,r=0), height=260,
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch upload
# ════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown('<div class="section-header">Batch Classification</div>', unsafe_allow_html=True)

    with st.expander("📋 Expected CSV format"):
        st.markdown(
            "Upload a **headerless** CSV/TXT in KDD format — **41 columns** (features only) "
            "or **43 columns** (with `attack_type` and `last_flag` labels, which are ignored)."
        )
        st.code(", ".join(COLNAMES[:43]))

    uploaded = st.file_uploader("Drop your file here", type=["csv","txt"],
                                label_visibility="collapsed")

    if uploaded:
        try:
            raw = pd.read_csv(uploaded, header=None)
            if   raw.shape[1] == 43: raw.columns = COLNAMES;      raw = raw.drop(columns=["attack_type","last_flag"], errors="ignore")
            elif raw.shape[1] == 41: raw.columns = COLNAMES[:41]
            else: st.error(f"❌ Expected 41 or 43 columns — got {raw.shape[1]}."); st.stop()

            st.success(f"✅ Loaded **{len(raw):,}** connections.")

            with st.spinner("Running inference…"):
                df_proc = preprocess_raw_csv(raw, scaler)
                preds   = model.predict(df_proc)
                probas  = model.predict_proba(df_proc)

            labels = [LABEL_MAP.get(int(p), "unknown") for p in preds]
            conf   = [float(probas[i, np.argmax(probas[i])]) for i in range(len(preds))]

            results_df = raw.copy()
            results_df["predicted_category"] = labels
            results_df["confidence"]         = [f"{c*100:.1f}%" for c in conf]

            st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
            counts = pd.Series(labels).value_counts()
            cols   = st.columns(min(len(counts), 6))
            for i, (lbl, cnt) in enumerate(counts.items()):
                color = LABEL_COLORS.get(lbl, "#6b7280")
                with cols[i % len(cols)]:
                    st.markdown(
                        f'<div class="count-card" style="border-color:{color}">'
                        f'<div class="num" style="color:{color}">{cnt:,}</div>'
                        f'<div class="lbl">{lbl} · {cnt/len(labels)*100:.1f}%</div></div>',
                        unsafe_allow_html=True,
                    )

            cl, cr = st.columns(2)
            with cl:
                st.markdown('<div class="section-header">Distribution</div>', unsafe_allow_html=True)
                fig_pie = px.pie(values=counts.values, names=counts.index,
                                 color=counts.index, color_discrete_map=LABEL_COLORS, hole=0.5)
                fig_pie.update_traces(textfont=dict(family="Share Tech Mono"))
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(t=10,b=10), height=300,
                                      legend=dict(font=dict(family="Share Tech Mono",color="#e2e8f0")),
                                      font=dict(color="#e2e8f0"))
                st.plotly_chart(fig_pie, use_container_width=True)

            with cr:
                st.markdown('<div class="section-header">Count by Category</div>', unsafe_allow_html=True)
                fig_bar = go.Figure(go.Bar(
                    x=counts.index.tolist(), y=counts.values.tolist(),
                    marker_color=[LABEL_COLORS.get(l,"#6b7280") for l in counts.index],
                    text=[str(v) for v in counts.values], textposition="outside",
                    textfont=dict(family="Share Tech Mono"),
                ))
                fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                      yaxis=dict(gridcolor="#1e2d40",tickfont=dict(family="Share Tech Mono")),
                                      xaxis=dict(tickfont=dict(family="Share Tech Mono")),
                                      margin=dict(t=10,b=10), height=300, font=dict(color="#e2e8f0"))
                st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("📄 Full results table"):
                st.dataframe(results_df[["src_bytes","dst_bytes","flag","service",
                                         "predicted_category","confidence"]],
                             use_container_width=True)

            st.download_button("⬇️  Download predictions as CSV",
                               data=results_df.to_csv(index=False),
                               file_name="nids_predictions.csv", mime="text/csv",
                               use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
  <div>
    🛡️ &nbsp;NIDS · Network Intrusion Detection System &nbsp;·&nbsp;
    Trained on <a href="https://www.unb.ca/cic/datasets/nsl.html" target="_blank">NSL-KDD</a>
  </div>
</div>
""", unsafe_allow_html=True)