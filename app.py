"""DiaGuard — premium diabetes risk screening app."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predict import (
    FEATURE_META,
    PRESETS,
    artifacts_exist,
    dataset_stats,
    feature_status,
    load_artifacts,
    predict,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiaGuard · Diabetes Risk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

      :root {
        --bg-deep: #050508;
        --bg-surface: #0c0c12;
        --card: rgba(255,255,255,0.04);
        --card-hover: rgba(255,255,255,0.06);
        --ink: #f1f5f9;
        --muted: #94a3b8;
        --line: rgba(255,255,255,0.08);
        --line-strong: rgba(255,255,255,0.14);
        --brand: #10b981;
        --brand-glow: rgba(16,185,129,0.35);
        --accent: #c9a227;
        --accent-soft: rgba(201,162,39,0.15);
      }

      .stApp {
        background:
          radial-gradient(ellipse 80% 50% at 50% -20%, rgba(16,185,129,0.12), transparent),
          radial-gradient(ellipse 60% 40% at 100% 0%, rgba(99,102,241,0.08), transparent),
          radial-gradient(ellipse 50% 30% at 0% 100%, rgba(201,162,39,0.05), transparent),
          linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-surface) 100%);
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--ink);
      }

      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 1.25rem; max-width: 1200px; padding-bottom: 3rem;}

      h1, h2, h3, h4, .outfit {font-family: 'Space Grotesk', sans-serif; color: var(--ink);}
      p, li, span {color: var(--ink);}
      .stMarkdown p {color: #cbd5e1;}

      /* Nav */
      .topbar {
        display: flex; align-items: center; justify-content: space-between;
        background: rgba(12,12,18,0.75); backdrop-filter: blur(20px);
        border: 1px solid var(--line); border-radius: 20px;
        padding: 1rem 1.5rem; margin-bottom: 1.75rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
      }
      .logo {
        font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.5rem;
        color:var(--ink); letter-spacing:-0.02em;
      }
      .logo span {
        background: linear-gradient(135deg, var(--brand), #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      .logo-sub {font-size:0.78rem; color:var(--muted); font-weight:500; margin-top:0.15rem;}
      .status-pill {
        display:inline-flex; align-items:center; gap:0.4rem;
        background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25);
        color:#34d399; font-size:0.72rem; font-weight:600; text-transform:uppercase;
        letter-spacing:0.06em; padding:0.35rem 0.75rem; border-radius:999px;
      }
      .status-dot {
        width:6px; height:6px; border-radius:50%; background:#34d399;
        box-shadow:0 0 8px #34d399; animation:pulse 2s ease-in-out infinite;
      }
      @keyframes pulse {0%,100%{opacity:1;}50%{opacity:0.5;}}

      /* Cards */
      .card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 20px; padding: 1.5rem 1.65rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
      }
      .card-title {
        font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.1rem;
        margin: 0 0 0.25rem 0; color: var(--ink); letter-spacing:-0.01em;
      }
      .card-sub {color:var(--muted); font-size:0.85rem; margin:0 0 1rem 0;}

      /* Stat tiles */
      .stat-grid {display:grid; grid-template-columns:repeat(4,1fr); gap:0.85rem; margin-bottom:1.5rem;}
      @media(max-width:900px){.stat-grid{grid-template-columns:repeat(2,1fr);}}
      .stat {
        background:var(--card); border:1px solid var(--line); border-radius:16px;
        padding:1.1rem 1.25rem; text-align:left;
        transition: border-color 0.2s, background 0.2s;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
      }
      .stat:hover {border-color:var(--line-strong); background:var(--card-hover);}
      .stat-label {font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); font-weight:600;}
      .stat-value {font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:var(--ink); margin-top:0.2rem; letter-spacing:-0.02em;}
      .stat-accent {
        background: linear-gradient(135deg, var(--brand), #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
      }

      /* Verdict */
      .verdict {
        border-radius:20px; padding:1.5rem 1.5rem; text-align:center;
        border:1px solid; margin-bottom:1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
      }
      .verdict-pct {
        font-family:'Space Grotesk',sans-serif; font-size:3.25rem; font-weight:700; line-height:1;
        letter-spacing:-0.03em;
      }
      .verdict-label {font-size:1.05rem; font-weight:600; margin-top:0.4rem; letter-spacing:0.02em;}
      .verdict-sub {font-size:0.88rem; color:var(--muted); margin-top:0.55rem; line-height:1.55;}

      /* Tips */
      .tip {
        background:rgba(255,255,255,0.03); border-left:3px solid var(--brand);
        padding:0.75rem 1rem; border-radius:0 12px 12px 0;
        margin-bottom:0.55rem; font-size:0.86rem; color:#cbd5e1; line-height:1.5;
        border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line);
      }

      /* Feature group label */
      .group-label {
        font-size:0.68rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.1em; color:var(--brand); margin:1.1rem 0 0.55rem 0;
      }

      /* Footer */
      .app-footer {
        color:#64748b; font-size:0.78rem; text-align:center; padding:1.5rem 1rem;
        border-top:1px solid var(--line); margin-top:1.5rem;
      }

      /* Streamlit overrides */
      .stSlider label, .stNumberInput label {
        font-weight:600 !important; font-size:0.86rem !important; color:#e2e8f0 !important;
      }
      .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {color:var(--brand) !important;}
      div[data-testid="stSliderThumb"] {background:var(--brand) !important; box-shadow:0 0 12px var(--brand-glow) !important;}
      div[data-testid="stSliderTrack"] > div {background:var(--brand) !important;}

      .stButton > button {
        border-radius: 12px !important; font-weight: 600 !important;
        transition: all 0.2s ease !important; font-family: 'Inter', sans-serif !important;
      }
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        border: 1px solid rgba(52,211,153,0.3) !important;
        color: #fff !important;
        box-shadow: 0 4px 20px var(--brand-glow), inset 0 1px 0 rgba(255,255,255,0.15) !important;
      }
      .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px var(--brand-glow) !important;
        transform: translateY(-1px);
      }
      .stButton > button[kind="secondary"] {
        border-radius: 12px !important;
        border: 1px solid var(--line-strong) !important;
        background: rgba(255,255,255,0.04) !important;
        color: #e2e8f0 !important;
      }
      .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.2) !important;
      }

      div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 0.85rem 1.1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
      }
      div[data-testid="stMetric"] label {color: var(--muted) !important; font-size:0.72rem !important; text-transform:uppercase; letter-spacing:0.05em;}
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family:'Space Grotesk',sans-serif !important; color: var(--ink) !important; font-weight:700 !important;
      }

      .stTabs [data-baseweb="tab-list"] {
        gap:8px; background:rgba(255,255,255,0.03); border:1px solid var(--line);
        border-radius:14px; padding:6px; margin-bottom:1.25rem;
      }
      .stTabs [data-baseweb="tab"] {
        border-radius:10px; font-weight:600; padding:10px 22px;
        background:transparent; border:none; color:var(--muted);
        font-family:'Inter',sans-serif;
      }
      .stTabs [data-baseweb="tab"]:hover {color:var(--ink); background:rgba(255,255,255,0.04);}
      .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.08)) !important;
        color: #34d399 !important;
        border: 1px solid rgba(52,211,153,0.3) !important;
        box-shadow: 0 2px 12px rgba(16,185,129,0.15) !important;
      }
      .stTabs [data-baseweb="tab-highlight"] {display:none;}
      .stTabs [data-baseweb="tab-border"] {display:none;}

      .stCaption {color: var(--muted) !important;}
      div[data-testid="stExpander"] {background:var(--card); border:1px solid var(--line); border-radius:14px;}
      .stCode, code {background:rgba(0,0,0,0.4) !important; color:#34d399 !important; border:1px solid var(--line);}
      div[data-testid="stAlert"] {border-radius:12px;}

      hr {border-color:var(--line) !important; margin:1.5rem 0 !important;}

      /* Plotly chart container */
      .js-plotly-plot .plotly .modebar {display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    model, preprocessing, metrics, backend = load_artifacts()
    return model, preprocessing, metrics, backend


def init_session():
    if "inputs" not in st.session_state:
        st.session_state.inputs = {k: v["default"] for k, v in FEATURE_META.items()}
    if "input_version" not in st.session_state:
        st.session_state.input_version = 0
    if "last_preset" not in st.session_state:
        st.session_state.last_preset = "Custom"


def apply_preset(name: str):
    if name in PRESETS and name != st.session_state.last_preset:
        st.session_state.inputs = PRESETS[name].copy()
        st.session_state.last_preset = name
        st.session_state.input_version += 1


def render_header():
    st.markdown(
        """
        <div class="topbar">
          <div>
            <div class="logo">Dia<span>Guard</span></div>
            <div class="logo-sub">AI Diabetes Risk Screening · Pima Indians Dataset</div>
          </div>
          <div class="status-pill"><span class="status-dot"></span> Model Live</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tiles(metrics: dict, live_risk: str | None = None):
    stats = dataset_stats()
    tiles = [
        ("Model Accuracy", f"{metrics.get('test_accuracy', 0) * 100:.0f}%", False),
        ("ROC-AUC Score", f"{metrics.get('test_auc', 0):.2f}", False),
        ("Dataset Size", f"{stats.get('total_patients', 768):,}", False),
        ("Live Risk", live_risk or "—", True),
    ]
    html = '<div class="stat-grid">'
    for label, value, accent in tiles:
        cls = "stat-value stat-accent" if accent else "stat-value"
        html += f'<div class="stat"><div class="stat-label">{label}</div><div class="{cls}">{value}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def make_risk_donut(probability: float, color: str) -> go.Figure:
    pct = probability * 100
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            values=[pct, 100 - pct],
            hole=0.72,
            marker={"colors": [color, "rgba(255,255,255,0.06)"]},
            textinfo="none",
            hoverinfo="skip",
            direction="clockwise",
            sort=False,
        )
    )
    fig.update_layout(
        annotations=[
            {
                "text": f"<b>{pct:.0f}%</b><br><span style='font-size:12px;color:#94a3b8'>risk score</span>",
                "x": 0.5, "y": 0.5, "font_size": 28, "showarrow": False,
                "font": {"family": "Space Grotesk", "color": "#f1f5f9"},
            }
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def make_impact_chart(impacts: list) -> go.Figure:
    labels = [i["label"] for i in impacts[:6]][::-1]
    values = [i["impact"] * 100 for i in impacts[:6]][::-1]
    colors = ["#10b981" if v >= 0 else "#475569" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
        textfont={"size": 11, "color": "#94a3b8"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis={
            "title": {"text": "Impact on risk (%)", "font": {"color": "#64748b", "size": 11}},
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.12)",
            "tickfont": {"color": "#64748b"},
        },
        yaxis={"tickfont": {"size": 12, "color": "#cbd5e1"}},
        font={"family": "Inter", "color": "#94a3b8"},
    )
    return fig


def make_comparison_chart(values: dict) -> go.Figure:
    stats = dataset_stats()
    features = ["Glucose", "BMI", "Age", "BloodPressure"]
    patient = [values[f] for f in features]
    median = [stats.get(f, {}).get("median", 0) for f in features]
    labels = [FEATURE_META[f]["label"] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="You", x=labels, y=patient,
        marker_color="#10b981", marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name="Dataset median", x=labels, y=median,
        marker_color="rgba(99,102,241,0.5)", marker_line_width=0,
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        legend={
            "orientation": "h", "y": 1.12, "x": 0,
            "font": {"color": "#94a3b8", "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        yaxis={"gridcolor": "rgba(255,255,255,0.06)", "tickfont": {"color": "#64748b"}},
        xaxis={"tickfont": {"color": "#cbd5e1"}},
        font={"family": "Inter", "size": 12, "color": "#94a3b8"},
    )
    return fig


def render_feature_input(feature: str, version: int):
    meta = FEATURE_META[feature]
    val = float(st.session_state.inputs[feature])
    status, color = feature_status(feature, val)

    col1, col2 = st.columns([3, 1])
    with col1:
        if feature == "DiabetesPedigreeFunction":
            step = 0.001
        elif feature == "BMI":
            step = 0.1
        else:
            step = 1.0
        new_val = st.slider(
            meta["label"],
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=val,
            step=step,
            help=meta["help"],
            key=f"sl_{feature}_{version}",
        )
    with col2:
        st.markdown(
            f"<div style='text-align:right;padding-top:1.6rem;'>"
            f"<span style='background:{color}18;color:{color};"
            f"padding:4px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;'>"
            f"{status}</span><br>"
            f"<span style='color:#64748b;font-size:0.78rem;'>{meta['unit']}</span></div>",
            unsafe_allow_html=True,
        )
    st.session_state.inputs[feature] = new_val


def page_screening(model, preprocessing, result: dict):
    v = st.session_state.input_version

    # Preset quick-load
    st.markdown('<p class="card-sub" style="margin-bottom:0.5rem;">Quick-load a patient profile</p>', unsafe_allow_html=True)
    pcols = st.columns(len(PRESETS) + 1)
    with pcols[0]:
        if st.button("Custom", use_container_width=True, type="secondary" if st.session_state.last_preset != "Custom" else "primary"):
            st.session_state.last_preset = "Custom"
            st.session_state.input_version += 1
            st.rerun()
    for i, (name, _) in enumerate(PRESETS.items()):
        with pcols[i + 1]:
            if st.button(name, use_container_width=True, type="primary" if st.session_state.last_preset == name else "secondary"):
                apply_preset(name)
                st.rerun()

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Clinical Inputs</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Adjust values — risk updates automatically</p>', unsafe_allow_html=True)

        groups = {}
        for feat, meta in FEATURE_META.items():
            groups.setdefault(meta["group"], []).append(feat)

        for group, features in groups.items():
            st.markdown(f'<p class="group-label">{group}</p>', unsafe_allow_html=True)
            gcols = st.columns(2)
            for idx, feat in enumerate(features):
                with gcols[idx % 2]:
                    render_feature_input(feat, v)

        if st.button("Reset all to defaults", type="secondary"):
            st.session_state.inputs = {k: m["default"] for k, m in FEATURE_META.items()}
            st.session_state.last_preset = "Custom"
            st.session_state.input_version += 1
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        style = result["style"]
        st.markdown(
            f"""
            <div class="verdict" style="background:{style['bg']};border-color:{style['border']};">
              <div class="verdict-pct" style="color:{style['color']};">{result['risk_percent']:.0f}%</div>
              <div class="verdict-label" style="color:{style['color']};">{style['label']}</div>
              <div class="verdict-sub">{style['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(make_risk_donut(result["probability"], style["color"]), use_container_width=True)

        outcome = "Positive screening" if result["is_diabetic"] else "Negative screening"
        ocolor = style["color"]
        st.markdown(
            f"<div style='text-align:center;margin-bottom:1rem;'>"
            f"<span style='background:{ocolor}15;color:{ocolor};padding:6px 16px;"
            f"border-radius:20px;font-weight:700;font-size:0.9rem;'>{outcome}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">What drives your score?</p>', unsafe_allow_html=True)
        st.plotly_chart(make_impact_chart(result["impacts"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Health insights</p>', unsafe_allow_html=True)
        for tip in result["tips"]:
            st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">You vs. population median</p>', unsafe_allow_html=True)
    st.plotly_chart(make_comparison_chart(st.session_state.inputs), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='app-footer'>"
        "For educational use only · Not a medical diagnosis · "
        "Always consult a healthcare professional</div>",
        unsafe_allow_html=True,
    )


def page_model(metrics: dict, backend: str):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Model Performance</p>', unsafe_allow_html=True)
    backend_label = "TensorFlow Keras ANN" if backend == "tensorflow" else "scikit-learn MLP"
    st.caption(f"Backend: {backend_label}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('test_accuracy', 0) * 100:.1f}%")
    c2.metric("ROC-AUC", f"{metrics.get('test_auc', 0):.3f}")
    c3.metric("Threshold", f"{metrics.get('threshold', 0.5):.0%}")
    c4.metric("Features", len(FEATURE_META))
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Architecture</p>', unsafe_allow_html=True)
        layers = [
            ("Input layer", "8 clinical features"),
            ("Hidden 1", "64 neurons · ReLU · BatchNorm · Dropout 15%"),
            ("Hidden 2", "32 neurons · ReLU · BatchNorm · Dropout 10%"),
            ("Hidden 3", "16 neurons · ReLU"),
            ("Output", "1 neuron · Sigmoid"),
        ]
        for name, detail in layers:
            st.markdown(f"**{name}** — {detail}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        if metrics.get("confusion_matrix"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="card-title">Confusion Matrix</p>', unsafe_allow_html=True)
            cm = np.array(metrics["confusion_matrix"])
            fig = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                colorscale=[[0, "rgba(16,185,129,0.15)"], [0.5, "rgba(16,185,129,0.45)"], [1, "#10b981"]],
                text=cm, texttemplate="%{text}",
                textfont={"size": 20, "color": "#f1f5f9"},
                showscale=False,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=280,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis={"tickfont": {"color": "#94a3b8"}},
                yaxis={"tickfont": {"color": "#94a3b8"}},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Preprocessing Pipeline</p>', unsafe_allow_html=True)
    for i, step in enumerate([
        "Zero values in Glucose, BP, SkinThickness, Insulin, BMI → treated as missing",
        "Median imputation on training set",
        "StandardScaler normalization",
        "Sigmoid output · 0.5 classification threshold",
    ], 1):
        st.markdown(f"{i}. {step}")
    st.markdown("</div>", unsafe_allow_html=True)


def page_about():
    stats = dataset_stats()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">About DiaGuard</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        DiaGuard is an AI-powered diabetes screening tool built on the
        **Pima Indians Diabetes Dataset** ({stats.get('total_patients', 768)} patients,
        {stats.get('diabetes_rate', 0.35) * 100:.0f}% positive rate).

        The neural network analyzes eight standard clinical measurements and returns
        a probability score with feature-level explanations — helping you understand
        *why* a risk score is high or low.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Feature glossary</p>', unsafe_allow_html=True)
    for feat, meta in FEATURE_META.items():
        st.markdown(f"**{meta['label']}** ({meta['unit']}) — {meta['help']}")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    if not artifacts_exist():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.error("Model not found. Train it first:")
        st.code("python train_model.py", language="bash")
        if st.button("Train now"):
            with st.spinner("Training…"):
                from train_model import train
                train()
            st.cache_resource.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    model, preprocessing, metrics, backend = get_model_bundle()

    render_header()

    tab1, tab2, tab3 = st.tabs(["Screening", "Model", "About"])
    with tab1:
        init_session()
        result = predict(st.session_state.inputs, model, preprocessing, backend)
        stat_tiles(metrics, f"{result['risk_percent']:.0f}%")
        page_screening(model, preprocessing, result)
    with tab2:
        page_model(metrics, backend)
    with tab3:
        page_about()


if __name__ == "__main__":
    main()
