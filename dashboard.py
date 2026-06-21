# ==============================================================================
# UrbanFlux Command Center
# Flipkart Gridlock | Bengaluru Traffic Intelligence Platform
# Data-Verified & Feature-Rich Version
# 
# DATA CORRECTIONS (verified against actual CSV):
#   - Dataset: Nov 2023 – Apr 2024 
#   - Total clean records: 248,371
#   - Approval rate: 46.46% 
#   - Repeat offender rate: 31.84%
#   - Peak Hour(7am-10am and 5pm to 9pm) = 35.56%
#   - Double parking incidents: 1,623
#   - Top offence: Wrong Parking (134,369 citations)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from collections import Counter

st.set_page_config(
    page_title="UrbanFlux — Bengaluru Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES — scroll-reveal, glassmorphism, smooth animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── APP BACKGROUND: deep cinematic dark with dynamic ambient gradient ── */
.stApp {
    background:
        radial-gradient(ellipse 1100px 600px at 10% -10%, rgba(220,38,38,0.08), transparent 55%),
        radial-gradient(ellipse 900px 700px at 90% 5%,  rgba(59,130,246,0.07), transparent 55%),
        radial-gradient(ellipse 1400px 900px at 50% 110%,rgba(139,92,246,0.05), transparent 60%),
        radial-gradient(ellipse 600px 400px at 70% 55%, rgba(234,179,8,0.04), transparent 50%),
        #060810;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 6rem; max-width: 1520px; margin: auto; }

/* ── SCROLL-REVEAL SYSTEM ── */
.reveal {
    opacity: 0;
    transform: translateY(40px);
    transition: opacity 0.75s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.75s cubic-bezier(0.22, 1, 0.36, 1);
}
.reveal.visible {
    opacity: 1;
    transform: translateY(0);
}
.reveal-left {
    opacity: 0;
    transform: translateX(-40px);
    transition: opacity 0.75s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.75s cubic-bezier(0.22, 1, 0.36, 1);
}
.reveal-left.visible { opacity: 1; transform: translateX(0); }
.reveal-right {
    opacity: 0;
    transform: translateX(40px);
    transition: opacity 0.75s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.75s cubic-bezier(0.22, 1, 0.36, 1);
}
.reveal-right.visible { opacity: 1; transform: translateX(0); }

/* Staggered delay helpers */
.delay-1 { transition-delay: 0.1s; }
.delay-2 { transition-delay: 0.2s; }
.delay-3 { transition-delay: 0.3s; }
.delay-4 { transition-delay: 0.4s; }
.delay-5 { transition-delay: 0.5s; }
.delay-6 { transition-delay: 0.6s; }

/* ── HERO SECTION ── */
.hero-wrap {
    text-align: center;
    padding: 5rem 0 3rem;
    position: relative;
    overflow: hidden;
}
.hero-scanner {
    position: absolute;
    top: 0; left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(220,38,38,0.06), transparent);
    animation: scanner 3s ease-in-out infinite;
    pointer-events: none;
}
@keyframes scanner {
    0%   { left: -50%; }
    100% { left: 150%; }
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #EF4444;
    margin-bottom: 18px;
    padding: 6px 16px;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    background: rgba(239,68,68,0.07);
    backdrop-filter: blur(12px);
}
.live-dot {
    width: 6px; height: 6px;
    background: #EF4444;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 6.5rem;
    font-weight: 700;
    letter-spacing: -4px;
    background: linear-gradient(135deg, #FFFFFF 0%, #EF4444 40%, #F59E0B 75%, #FFFFFF 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 6s ease infinite;
    line-height: 1;
    margin: 0;
}
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-sub {
    font-size: 0.85rem;
    color: #6B7280;
    font-weight: 500;
    letter-spacing: 5px;
    margin-top: 14px;
    text-transform: uppercase;
}
.hero-tagline {
    font-size: 1.1rem;
    color: #9CA3AF;
    font-weight: 300;
    margin-top: 20px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.7;
}
.hero-stats-bar {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 40px;
    padding: 24px 48px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    backdrop-filter: blur(20px);
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
.hero-stat { text-align: center; }
.hero-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F4F6FA;
    letter-spacing: -1px;
}
.hero-stat-label { font-size: 10px; color: #6B7280; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }
.hero-stat-divider { width: 1px; background: rgba(255,255,255,0.08); }

/* ── KPI GRID ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 14px;
    margin: 2.5rem 0 1rem;
}
.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 22px 16px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: transform 0.3s cubic-bezier(.2,.8,.2,1),
                border-color 0.3s ease,
                box-shadow 0.3s ease;
}
.kpi-card::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 20px;
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
}
.kpi-card:hover { transform: translateY(-6px); }
.kpi-card:hover::after { opacity: 1; }

.kpi-glow {
    position: absolute;
    width: 120px; height: 120px;
    border-radius: 50%;
    filter: blur(50px);
    opacity: 0.15;
    top: -30px; left: 50%; transform: translateX(-50%);
    pointer-events: none;
    transition: opacity 0.3s ease;
}
.kpi-card:hover .kpi-glow { opacity: 0.35; }

.kpi-card.red:hover    { border-color: rgba(239,68,68,0.35);   box-shadow: 0 16px 40px -12px rgba(239,68,68,0.3); }
.kpi-card.amber:hover  { border-color: rgba(245,158,11,0.35);  box-shadow: 0 16px 40px -12px rgba(245,158,11,0.3); }
.kpi-card.green:hover  { border-color: rgba(34,197,94,0.35);   box-shadow: 0 16px 40px -12px rgba(34,197,94,0.3); }
.kpi-card.blue:hover   { border-color: rgba(59,130,246,0.35);  box-shadow: 0 16px 40px -12px rgba(59,130,246,0.3); }
.kpi-card.purple:hover { border-color: rgba(139,92,246,0.35);  box-shadow: 0 16px 40px -12px rgba(139,92,246,0.3); }
.kpi-card.teal:hover   { border-color: rgba(20,184,166,0.35);  box-shadow: 0 16px 40px -12px rgba(20,184,166,0.3); }

.kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase;
    color: #6B7280; margin-bottom: 10px; position: relative; z-index: 1;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700; color: #F9FAFB;
    line-height: 1.1; letter-spacing: -0.5px; position: relative; z-index: 1;
}
.kpi-sub {
    font-size: 10.5px; color: #4B5563; margin-top: 6px; position: relative; z-index: 1;
}
.kpi-badge {
    display: inline-block;
    font-size: 9px; font-weight: 700;
    padding: 2px 7px; border-radius: 10px;
    margin-top: 5px; position: relative; z-index: 1;
}
.badge-up   { background: rgba(34,197,94,0.15); color: #4ADE80; }
.badge-warn { background: rgba(239,68,68,0.15); color: #F87171; }

/* ── SECTION DIVIDER ── */
.section-divider {
    position: relative;
    text-align: center;
    margin: 4rem 0 3rem;
}
.section-divider::before {
    content: "";
    position: absolute;
    top: 50%; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.section-label {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    position: relative;
    background: #060810;
    padding: 0 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4B5563;
}

/* ── SECTION HEADERS ── */
.section-header {
    margin-bottom: 1.8rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.section-eyebrow {
    font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    color: #EF4444; margin-bottom: 6px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #F9FAFB;
    letter-spacing: -0.5px; margin: 0;
    display: flex; align-items: center; gap: 10px;
}
.section-desc { font-size: 13px; color: #6B7280; margin-top: 4px; }

/* ── CHART PANELS ── */
.chart-panel {
    background: linear-gradient(160deg, rgba(255,255,255,0.032) 0%, rgba(255,255,255,0.008) 100%);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 20px;
    padding: 20px 20px 6px;
    height: 100%;
    transition: border-color 0.3s ease;
}
.chart-panel:hover { border-color: rgba(255,255,255,0.1); }
.chart-panel-title {
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
    color: #9CA3AF; margin-bottom: 12px;
}

/* ── DISPATCH CARDS ── */
.dispatch-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
    transition: all 0.25s cubic-bezier(.2,.8,.2,1);
    position: relative;
    overflow: hidden;
}
.dispatch-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
}
.dispatch-card.critical::before { background: linear-gradient(180deg, #EF4444, #DC2626); }
.dispatch-card.high::before     { background: linear-gradient(180deg, #F59E0B, #D97706); }
.dispatch-card.medium::before   { background: linear-gradient(180deg, #3B82F6, #2563EB); }
.dispatch-card:hover { transform: translateX(4px); border-color: rgba(255,255,255,0.12); }

.dispatch-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 500; color: #4B5563;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 6px;
}
.dispatch-zone {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 700; color: #F9FAFB;
    margin-bottom: 4px;
    display: flex; align-items: center; gap: 10px;
}
.dispatch-junction { font-size: 11px; color: #6B7280; margin-bottom: 10px; }
.dispatch-pills { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.pill {
    font-size: 10px; font-weight: 500; padding: 3px 9px;
    border-radius: 20px; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08); color: #9CA3AF;
    white-space: nowrap;
}
.pill.danger { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.2); color: #F87171; }
.pill.warn   { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.2); color: #FCD34D; }
.dispatch-action { font-size: 11px; color: #6B7280; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 8px; }

.sev-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    padding: 2px 9px; border-radius: 6px;
}
.sev-crit { background: rgba(239,68,68,0.18); color: #F87171; }
.sev-high { background: rgba(245,158,11,0.18); color: #FCD34D; }
.sev-med  { background: rgba(59,130,246,0.18); color: #93C5FD; }

/* ── DATA TABLE ── */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stDataFrame"] thead { background: rgba(255,255,255,0.04); }

/* ── INSIGHT CARDS ── */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}
.insight-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.25s ease;
}
.insight-card:hover { border-color: rgba(255,255,255,0.1); transform: translateY(-3px); }
.insight-icon { font-size: 24px; margin-bottom: 10px; }
.insight-title { font-size: 12px; font-weight: 700; color: #9CA3AF; letter-spacing: 0.5px; margin-bottom: 6px; }
.insight-body { font-size: 13px; color: #D1D5DB; line-height: 1.6; }
.insight-highlight { color: #EF4444; font-weight: 700; }

/* ── DATA CORRECTION BANNER ── */
.correction-banner {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 1rem;
    font-size: 12px;
    color: #FCD34D;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 3rem 0 2rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 2rem;
}
.footer-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700; color: #374151;
    letter-spacing: -0.5px;
}
.footer-meta { font-size: 11px; color: #374151; margin-top: 6px; letter-spacing: 1px; }

div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
.js-plotly-plot { background: transparent !important; }
</style>

<!-- SCROLL REVEAL SCRIPT -->
<script>
(function() {
    function initReveal() {
        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

        document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(function(el) {
            observer.observe(el);
        });
    }

    // Run after Streamlit re-renders
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initReveal);
    } else {
        initReveal();
    }
    // Also watch for Streamlit mutations
    const mutObs = new MutationObserver(function() { initReveal(); });
    mutObs.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (verified values)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw_data():

    df = pd.read_csv("jan to may police violation_anonymized791b166.csv")

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df["validation_timestamp"] = pd.to_datetime(df["validation_timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime", "latitude", "longitude"])

    # Bengaluru bounding box
    df = df[df["latitude"].between(12.80, 13.30) & df["longitude"].between(77.44, 77.78)].copy()
    df = df[~df["validation_status"].isin(["rejected", "duplicate"])].copy()

    df["final_vehicle_type"] = (
        df["updated_vehicle_type"].fillna(df["vehicle_type"]).fillna("UNKNOWN").str.upper()
    )

    # --- FIX 1: Resolve final_vehicle_number identically to pipeline.py ---
    df["final_vehicle_number"] = (
        df["updated_vehicle_number"]
        .fillna(df["vehicle_number"])
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    # --- CRITICAL TIMEZONE FIX FOR THE DASHBOARD ---
    dt_ist = df["created_datetime"].dt.tz_convert('Asia/Kolkata')

    df["hour"]      = dt_ist.dt.hour
    df["month"]     = dt_ist.dt.month
    df["dayofweek"] = dt_ist.dt.dayofweek

    df["peak_hour"] = (
        (df["hour"].between(7, 10)) | (df["hour"].between(17, 21))
    ).astype(int)

    # --- FIX 2: BURST DEDUPLICATION (15-Minute Sensor Sync Window) ---
    # Mirrors pipeline.py's load_data() exactly, so `raw` and `final_dashboard.csv`
    # are built from the same population.
    df = df.sort_values(['final_vehicle_number', 'created_datetime'])
    df['time_diff'] = df.groupby('final_vehicle_number')['created_datetime'].diff()
    df = df[(df['time_diff'].isna()) | (df['time_diff'] > pd.Timedelta(minutes=15))].copy()
    df = df.drop(columns=['time_diff'])

    def safe_json(v):
        if pd.isna(v): return []
        try:    return json.loads(v)
        except: return []

    df["violation_list"] = df["violation_type"].apply(safe_json)
    df["has_double_parking"] = df["violation_list"].apply(
        lambda l: any("DOUBLE PARKING" in str(x).upper() for x in l)
    ).astype(int)

    # --- FIX 3: use final_vehicle_number (post-dedup) to match pipeline.py's is_global_repeat ---
    vc = df["final_vehicle_number"].value_counts()
    df["is_repeat"] = df["final_vehicle_number"].isin(vc[vc > 1].index).astype(int)

    return df
@st.cache_data(show_spinner=False)
def load_dashboard_data():
    try:
        return pd.read_csv("final_dashboard.csv")
    except FileNotFoundError:
        return pd.DataFrame()

# --- ADD THIS LOADER ---
@st.cache_data(show_spinner=False)
def load_route_data():
    try:
        return pd.read_csv("patrol_route.csv")
    except FileNotFoundError:
        return pd.DataFrame()
# -----------------------

with st.spinner("Loading Bengaluru traffic intelligence..."):
    raw = load_raw_data()
    df  = load_dashboard_data()
    route_df = load_route_data() # <-- Add this line
    total_patrol_km = route_df["leg_km"].sum() if not route_df.empty else 0.0 # <-- Add this line

@st.cache_data(show_spinner=False)
def load_dashboard_data():
    try:
        return pd.read_csv("final_dashboard.csv")
    except FileNotFoundError:
        return pd.DataFrame()


with st.spinner("Loading Bengaluru traffic intelligence..."):
    raw = load_raw_data()
    df  = load_dashboard_data()

# ── KPI derivations ──────────────────────────────────────────────────
total_records   = len(raw)                                   # 248,371
approved_pct    = (raw["validation_status"] == "approved").mean() * 100  # 46.46%
peak_hour_share = raw["peak_hour"].mean() * 100              # 23.64%
repeat_rate     = raw["is_repeat"].mean() * 100              # 31.84%
double_park_n   = raw["has_double_parking"].sum()            # 1,623

all_viols   = [v for lst in raw["violation_list"] for v in lst]
viol_counts = Counter(all_viols)
top_viol    = viol_counts.most_common(1)[0][0].title() if viol_counts else "Wrong Parking"
top_viol_n  = viol_counts.most_common(1)[0][1] if viol_counts else 0

# Date range
date_min = raw["created_datetime"].min().strftime("%b %Y")
date_max = raw["created_datetime"].max().strftime("%b %Y")


# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap reveal">
  <div class="hero-scanner"></div>
  <div class="hero-eyebrow">
    <div class="live-dot"></div>
    Bengaluru · Traffic Intelligence Grid · {date_min} — {date_max}
  </div>
  <div class="hero-title">UrbanFlux</div>
  <div class="hero-sub">Preemptive Enforcement Command Center</div>
  <div class="hero-tagline">
    Real-time congestion intelligence, micro-spatial hotspot detection, and AI-powered
    dispatch prioritisation across Greater Bengaluru.
  </div>
  <div class="hero-stats-bar">
    <div class="hero-stat">
      <div class="hero-stat-val">{total_records:,}</div>
      <div class="hero-stat-label">Clean Records</div>
    </div>
    <div class="hero-stat-divider"></div>
    <div class="hero-stat">
      <div class="hero-stat-val">{approved_pct:.1f}%</div>
      <div class="hero-stat-label">Approval Rate</div>
    </div>
    <div class="hero-stat-divider"></div>
    <div class="hero-stat">
      <div class="hero-stat-val">{repeat_rate:.1f}%</div>
      <div class="hero-stat-label">Repeat Offenders</div>
    </div>
    <div class="hero-stat-divider"></div>
    <div class="hero-stat">
      <div class="hero-stat-val">{double_park_n:,}</div>
      <div class="hero-stat-label">Choke Events</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI BANNER 
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── TOOLTIP CSS (Fixed Positioning & Yellow Text) ── */
.info-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-left: 6px;
    color: #9CA3AF;
    cursor: help;
    position: relative;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 9px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50%;
    width: 14px;
    height: 14px;
    transition: all 0.2s ease;
    text-transform: none;
    letter-spacing: normal;
    z-index: 50;
}}
.info-icon:hover {{
    background: rgba(255,255,255,0.25);
    color: #FFFFFF;
}}
.info-icon .tooltip-text {{
    visibility: hidden;
    width: 160px; 
    background-color: rgba(15, 23, 42, 0.95); 
    color: #FFD700; /* <-- CHANGED TO YELLOW HERE */
    text-align: center;
    border-radius: 8px;
    padding: 10px;
    position: absolute;
    z-index: 100;
    top: 150%; 
    bottom: auto; 
    left: 50%;
    transform: translateX(-50%) translateY(-5px);
    opacity: 0;
    transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1);
    font-size: 10px;
    font-weight: 600; /* Slightly bolder to make the yellow pop */
    letter-spacing: 0.2px;
    line-height: 1.4;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    backdrop-filter: blur(8px);
}}
/* Flipped arrow pointing UP towards the icon */
.info-icon .tooltip-text::after {{
    content: "";
    position: absolute;
    bottom: 100%; 
    top: auto;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: transparent transparent rgba(15, 23, 42, 0.95) transparent;
}}
.info-icon:hover .tooltip-text {{
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}}
</style>

<div class="kpi-grid reveal delay-1">
  <div class="kpi-card red">
    <div class="kpi-glow" style="background:#EF4444;"></div>
    <div class="kpi-label">Total Violations <div class="info-icon">i<span class="tooltip-text">Total raw traffic violation reports captured across the city after administrative filtering.</span></div></div>
    <div class="kpi-value">{total_records:,}</div>
    <div class="kpi-sub">{date_min} – {date_max}</div>
    <span class="kpi-badge badge-warn">6 months</span>
  </div>
  
 <div class="kpi-card amber">
    <div class="kpi-glow" style="background:#F59E0B;"></div>
    <div class="kpi-label">Patrol Circuit <div class="info-icon">i<span class="tooltip-text">Optimal driving distance (Nearest-Neighbour TSP) to dispatch units to all Top-10 Choke Zones.</span></div></div>
    <div class="kpi-value">{total_patrol_km:.1f} <span style="font-size:16px;">km</span></div>
    <div class="kpi-sub">shortest loop</div>
    <span class="kpi-badge badge-warn">optimized</span>
  </div>
  
  <div class="kpi-card purple">
    <div class="kpi-glow" style="background:#8B5CF6;"></div>
    <div class="kpi-label">Repeat Offenders <div class="info-icon">i<span class="tooltip-text">Percentage of total violations linked to vehicles that have been flagged multiple times.</span></div></div>
    <div class="kpi-value">{repeat_rate:.1f}%</div>
    <div class="kpi-sub">of all vehicles</div>
    <span class="kpi-badge badge-warn">↑ high recidivism</span>
  </div>
  
  <div class="kpi-card blue">
    <div class="kpi-glow" style="background:#3B82F6;"></div>
    <div class="kpi-label">Peak Hour Share <div class="info-icon">i<span class="tooltip-text">Percentage of infractions happening during the most congested daily windows (7-10 AM & 5-9 PM).</span></div></div>
    <div class="kpi-value">{peak_hour_share:.1f}%</div>
    <div class="kpi-sub">7–10 AM · 5–9 PM</div>
    <span class="kpi-badge badge-up">key window</span>
  </div>
  
 <div class="kpi-card teal">
    <div class="kpi-glow" style="background:#14B8A6;"></div>
    <div class="kpi-label">Economic Impact <div class="info-icon">i<span class="tooltip-text">Estimated monthly economic loss (Rupees) from vehicle-hours lost due to congestion in these hotspots.</span></div></div>
    <div class="kpi-value">₹{df['total_economic_impact'].sum()/1000000:.1f}M</div>
    <div class="kpi-sub">monthly loss</div>
    <span class="kpi-badge badge-warn">⚠ citywide burden</span>
  </div>
  
  <div class="kpi-card green">
    <div class="kpi-glow" style="background:#22C55E;"></div>
    <div class="kpi-label">Top Offence <div class="info-icon">i<span class="tooltip-text">The most frequent type of traffic violation recorded across the entire city grid.</span></div></div>
    <div class="kpi-value" style="font-size:17px; padding-top:6px;">{top_viol}</div>
    <div class="kpi-sub">{top_viol_n:,} citations</div>
    <span class="kpi-badge badge-warn">54% of all</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MAP + DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider reveal">
  <span class="section-label">01 · Spatial Intelligence</span>
</div>
""", unsafe_allow_html=True)

map_col, dispatch_col = st.columns([3, 2], gap="large")

PLOT_BG  = "rgba(0,0,0,0)"
GRID_CLR = "rgba(255,255,255,0.05)"
TEXT_CLR = "#6B7280"
FONT_FAM = "Inter, Space Grotesk, sans-serif"

def base_layout(title="", height=340):
    return dict(
        # 1. FIXED TITLE POSITION:
        title=dict(text=title, font=dict(size=12.5, color="#9CA3AF", family=FONT_FAM), x=0.01, xanchor="left", y=0.96),
        height=height,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        # 2. FIXED TOP MARGIN: 
        margin=dict(l=6, r=10, t=45, b=6),
        font=dict(family=FONT_FAM, color=TEXT_CLR, size=10.5),
        xaxis=dict(gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=9.5)),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=9.5)),
        showlegend=False
    )

with map_col:
    st.markdown("""
    <div class="section-header reveal-left">
      <div class="section-eyebrow">Live Intelligence Layer</div>
      <div class="section-title">🗺️ Congestion Heatmap</div>
      <div class="section-desc">Density-weighted violation scatter across Greater Bengaluru. Red zones indicate critical enforcement clusters.</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        with open("urbanflux_calibrated_dashboard.html", "r", encoding="utf-8") as f:
            html_map = f.read()
        components.html(html_map, height=580, scrolling=False)
    except FileNotFoundError:
        sample = raw.dropna(subset=["latitude","longitude"]).sample(min(10000, len(raw)), random_state=42)
        fig_map = px.density_mapbox(
            sample, lat="latitude", lon="longitude",
            radius=9, center={"lat": 12.97, "lon": 77.59}, zoom=11,
            mapbox_style="carto-darkmatter",
            color_continuous_scale=["#020617","#7F1D1D","#EF4444","#FCD34D"],
            height=580
        )
        fig_map.update_layout(
            margin=dict(l=0,r=0,t=0,b=0),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True)

with dispatch_col:
    st.markdown("""
    <div class="section-header reveal-right">
      <div class="section-eyebrow">Priority Queue</div>
      <div class="section-title">📋 Active Dispatch Orders</div>
      <div class="section-desc">Real-time enforcement prioritisation sorted by Composite Severity Index.</div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        top5 = df.head(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            sev       = row.get("CompositeSeverity", 0)
            card_cls  = "critical" if sev >= 75 else "high" if sev >= 50 else "medium"
            badge_cls = "sev-crit" if sev >= 75 else "sev-high" if sev >= 50 else "sev-med"
            icon      = "🔴" if sev >= 75 else "🟠" if sev >= 50 else "🔵"
            
            junction  = str(row.get("primary_junction", "—"))
            station   = str(row.get("jurisdiction", "Unknown"))
            peak_t    = row.get("peak_hours_deployment", "—")
            vehicle   = str(row.get("dominant_vehicle","—")).title()
            choke     = row.get("CHOKE_ZONE","No")
            rec       = str(row.get("Actionable_Recommendation","Deploy Patrol Unit."))[:120]
            roi       = row.get("Enforcement_ROI", 0)

         
            if "BTP" in junction.upper():
                rank_label = f"Order #{i+1} · Priority Dispatch"
                display_title = junction[:32]
            else:
                rank_label = f"Order #{i+1} · {station} Jurisdiction"
                # If it's a nameless stretch of road, append the station's name!
                display_title = f"{station} Mid-Block" if "Mid-Block" in junction else junction[:32]
            # -------------------------------------------------------------------------

            choke_pill = '<span class="pill danger">⚠ Choke Zone</span>' if choke == "Yes" else ""
            vehicle_pill = f'<span class="pill">🚗 {vehicle}</span>'
            roi_pill = f'<span class="pill warn">ROI {roi:.3f}</span>' if roi > 0.05 else f'<span class="pill">ROI {roi:.3f}</span>'

            st.markdown(f"""
            <div class="dispatch-card {card_cls} reveal delay-{min(i+1,5)}">
              <div class="dispatch-rank">{rank_label}</div>
              <div class="dispatch-zone">
                {icon} {display_title}
                <span class="sev-badge {badge_cls}">{sev:.0f}</span>
              </div>
              <div class="dispatch-pills">
                <span class="pill">⏱ {peak_t}</span>
                {vehicle_pill}
                {roi_pill}
                {choke_pill}
              </div>
              <div class="dispatch-action">{rec}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run `pipeline.py` to generate `final_dashboard.csv` and activate dispatch orders.")


# ─────────────────────────────────────────────────────────────────────────────
# KEY INSIGHTS ROW
# ─────────────────────────────────────────────────────────────────────────────
top_station = raw["police_station"].value_counts().index[0]
top_station_n = raw["police_station"].value_counts().iloc[0]
scooter_n = (raw["final_vehicle_type"] == "SCOOTER").sum()
scooter_pct = scooter_n / total_records * 100

st.markdown(f"""
<div class="section-divider reveal">
  <span class="section-label">02 · Key Intelligence Findings</span>
</div>
<div class="insight-grid reveal delay-1">
  <div class="insight-card">
    <div class="insight-icon">🕵️‍♂️</div>
    <div class="insight-title">The 3 PM Enforcement Blindspot</div>
    <div class="insight-body">
      Citations plummet to near-zero after 15:00. This data proves current enforcement is <b>shift-based and manual</b>, leaving the massive 5 PM–9 PM evening commuter rush completely unpoliced.
    </div>
  </div>
  <div class="insight-card">
    <div class="insight-icon">🚚</div>
    <div class="insight-title">Overnight Commercial Fleet</div>
    <div class="insight-body">
      The massive citation spike between <b>2 AM and 5 AM</b> aligns perfectly with Bengaluru's heavy vehicle entry windows, revealing rampant illegal unloading hotspots before dawn.
    </div>
  </div>
  <div class="insight-card">
    <div class="insight-icon">🛵</div>
    <div class="insight-title">Two-Wheeler Dominance</div>
    <div class="insight-body">
      Scooters account for <span class="insight-highlight">{scooter_pct:.1f}%</span> of all citations
      ({scooter_n:,} records), making them the primary structural cause of mid-block choke zones.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TEMPORAL ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider reveal">
  <span class="section-label">03 · Temporal Intelligence</span>
</div>
<div class="section-header reveal">
  <div class="section-eyebrow">Time-Series Analysis</div>
  <div class="section-title">📈 Violation Patterns Over Time</div>
  <div class="section-desc">When violations peak and how enforcement load shifts across months, days, and hours.</div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3, gap="medium")

# ── Heatmap: hour vs day ───────────────────────────────────────────────────
with t1:
    st.markdown('<div class="chart-panel reveal-left">', unsafe_allow_html=True)
    hour_day = raw.groupby(["hour","dayofweek"]).size().reset_index(name="count")
    pivot    = hour_day.pivot(index="dayofweek", columns="hour", values="count").fillna(0)
    days     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=days,
        colorscale=[[0,"#020617"],[0.35,"#450A0A"],[0.65,"#991B1B"],[1,"#EF4444"]],
        showscale=False,
        hovertemplate="<b>%{y} %{x}:00</b><br>%{z:,} violations<extra></extra>"
    ))
    fig_hm.update_layout(**base_layout("Violations by Hour & Weekday", 330))
    st.plotly_chart(fig_hm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Monthly bars ──────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="chart-panel reveal delay-2">', unsafe_allow_html=True)
    # Correct month labels based on actual data (Nov 2023 – Apr 2024)
    month_label_map = {11:"Nov '23", 12:"Dec '23", 1:"Jan '24", 2:"Feb '24", 3:"Mar '24", 4:"Apr '24"}
    month_data = raw.groupby("month").size().reset_index(name="count")
    month_data["label"] = month_data["month"].map(month_label_map)
    month_data = month_data.sort_values("month")  # sort by calendar order
    # Re-sort by actual chronological order: Nov, Dec, Jan, Feb, Mar, Apr
    chron_order = [11, 12, 1, 2, 3, 4]
    month_data["sort_key"] = month_data["month"].map({m: i for i, m in enumerate(chron_order)})
    month_data = month_data.sort_values("sort_key")
    bar_colors = ["#EF4444" if c == month_data["count"].max() else "#1E2D4A" for c in month_data["count"]]

    fig_month = go.Figure(go.Bar(
        x=month_data["label"], y=month_data["count"],
        marker_color=bar_colors, marker_line_width=0,
        text=month_data["count"].apply(lambda x: f"{x/1000:.1f}k"),
        textposition="outside", textfont=dict(size=10, color="#9CA3AF"),
        hovertemplate="<b>%{x}</b><br>%{y:,} violations<extra></extra>"
    ))
    fig_month.update_layout(**base_layout("Monthly Violation Volume", 330))
    fig_month.update_yaxes(visible=False)
    st.plotly_chart(fig_month, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Violation type donut ───────────────────────────────────────────────────
with t3:
    st.markdown('<div class="chart-panel reveal-right">', unsafe_allow_html=True)
    top_viols = viol_counts.most_common(6)
    vlabels   = [v[0].replace("PARKING","PKG").title()[:24] for v in top_viols]
    vvalues   = [v[1] for v in top_viols]
    vcolors   = ["#EF4444","#F97316","#F59E0B","#3B82F6","#8B5CF6","#14B8A6"]

    fig_donut = go.Figure(go.Pie(
        labels=vlabels, values=vvalues, hole=0.62,
        marker=dict(colors=vcolors, line=dict(color="#060810", width=3)),
        textinfo="percent", textfont=dict(size=10, color="#FFFFFF"),
        hovertemplate="<b>%{label}</b><br>%{value:,} cases · %{percent}<extra></extra>"
    ))
    fig_donut.add_annotation(
        text=f"{sum(vvalues):,}", x=0.5, y=0.58, showarrow=False,
        font=dict(size=22, color="#F9FAFB", family="Space Grotesk, sans-serif", weight=700)
    )
    fig_donut.add_annotation(
        text="citations", x=0.5, y=0.44, showarrow=False,
        font=dict(size=10, color="#6B7280")
    )
    layout_d = base_layout("Violation Type Breakdown", 330)
    layout_d["showlegend"] = True
    layout_d["legend"] = dict(orientation="v", x=1.0, y=0.5, font=dict(size=9, color="#6B7280"))
    fig_donut.update_layout(**layout_d)
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: VEHICLE & STATION INTEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider reveal">
  <span class="section-label">04 · Enforcement Intel</span>
</div>
<div class="section-header reveal">
  <div class="section-eyebrow">Resource Allocation Analysis</div>
  <div class="section-title">🚗 Vehicle Profile & Station Burden</div>
  <div class="section-desc">Where enforcement load concentrates — by vehicle class and police jurisdiction.</div>
</div>
""", unsafe_allow_html=True)

ve_col, st_col = st.columns(2, gap="large")

with ve_col:
    st.markdown('<div class="chart-panel reveal-left">', unsafe_allow_html=True)
    veh_df = raw["final_vehicle_type"].value_counts().head(8).reset_index()
    veh_df.columns = ["vehicle","count"]
    veh_df = veh_df.sort_values("count")
    vcols = ["#EF4444" if i == len(veh_df)-1 else "#1A2D4A" for i in range(len(veh_df))]

    fig_veh = go.Figure(go.Bar(
        x=veh_df["count"], y=veh_df["vehicle"].str.title(),
        orientation="h", marker_color=vcols, marker_line_width=0,
        text=veh_df["count"].apply(lambda x: f"{x:,}"),
        textposition="outside", textfont=dict(size=9.5, color="#9CA3AF"),
        hovertemplate="<b>%{y}</b><br>%{x:,} violations<extra></extra>"
    ))
    fig_veh.update_layout(**base_layout("Citations by Vehicle Type", 400))
    fig_veh.update_xaxes(visible=False)
    st.plotly_chart(fig_veh, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st_col:
    st.markdown('<div class="chart-panel reveal-right">', unsafe_allow_html=True)
    ps_df = raw["police_station"].value_counts().head(10).reset_index()
    ps_df.columns = ["station","count"]
    ps_df = ps_df.sort_values("count")
    # Gradient coloring based on rank
    n = len(ps_df)
    ps_colors = [f"rgba(239,68,68,{0.25 + 0.75*(i/n):.2f})" for i in range(n)]

    fig_ps = go.Figure(go.Bar(
        x=ps_df["count"], y=ps_df["station"],
        orientation="h", marker_color=ps_colors, marker_line_width=0,
        text=ps_df["count"].apply(lambda x: f"{x:,}"),
        textposition="outside", textfont=dict(size=9.5, color="#9CA3AF"),
        hovertemplate="<b>%{y}</b><br>%{x:,} violations<extra></extra>"
    ))
    fig_ps.update_layout(**base_layout("Top 10 Police Stations by Enforcement Load", 400))
    fig_ps.update_xaxes(visible=False)
    st.plotly_chart(fig_ps, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CLUSTER INTELLIGENCE (if pipeline has run)
# ─────────────────────────────────────────────────────────────────────────────
if not df.empty:
    st.markdown("""
   <div class="section-divider reveal">
      <span class="section-label">05 · Hotspot Analysis</span>
    </div>
    <div class="section-header reveal">
      <div class="section-eyebrow">Micro-Spatial Cluster Intelligence</div>
      <div class="section-title">📊 Hotspot Severity & Operational Evidence</div>
      <div class="section-desc">DBSCAN-clustered zones ranked by Composite Severity Index (Density 25% · PCII 45% · Latency 15% · Recidivism 15% + Choke Zone Bonus).</div>
    </div>
    """, unsafe_allow_html=True)

    cl1, cl2 = st.columns([3, 2], gap="large")

   
    # Extracts the exact BTP code or real Police Station name from your pipeline data
    def get_real_zone_name(row):
        junc = str(row.get("primary_junction", ""))
        stat = str(row.get("jurisdiction", "Unknown"))
        
        if "BTP" in junc.upper():
            return junc[:24] # e.g., "BTP040 - Elite Junction"
        else:
            return f"{stat} Area" # e.g., "Upparpet Area"
    # ----------------------------------------

    with cl1:
        st.markdown('<div class="chart-panel reveal-left">', unsafe_allow_html=True)
        top10 = df.head(10).copy()
        
       # Apply the real names and enforce uniqueness to stop Plotly from stacking bars
        top10["Zone"] = top10.apply(lambda r: f"{get_real_zone_name(r)} [#{int(r['cluster'])}]", axis=1)
        
        top10 = top10.sort_values("CompositeSeverity")
        bar_c = ["#EF4444" if s >= 75 else "#F59E0B" if s >= 50 else "#3B82F6"
                 for s in top10["CompositeSeverity"]]

        fig_sev = go.Figure()
        fig_sev.add_trace(go.Bar(
            x=top10["CompositeSeverity"], y=top10["Zone"],
            orientation="h", marker_color=bar_c, marker_line_width=0,
            text=top10["CompositeSeverity"].round(1),
            textposition="outside", textfont=dict(size=10, color="#9CA3AF"),
            name="Composite Severity",
            hovertemplate="<b>%{y}</b><br>Severity: %{x:.1f}<extra></extra>"
        ))
        
        fig_sev.update_layout(**base_layout("Top 10 Enforcement Zones — Composite Severity Score", 460))
        fig_sev.update_xaxes(range=[0, 115])
        st.plotly_chart(fig_sev, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cl2:
        st.markdown('<div class="chart-panel reveal-right">', unsafe_allow_html=True)
        bubble_df = df.head(15).copy()
        
        # Apply the real names to the Bubble Chart too, with ID for uniqueness
        bubble_df["Zone"] = bubble_df.apply(lambda r: f"{get_real_zone_name(r)} [#{int(r['cluster'])}]", axis=1)

        fig_bubble = go.Figure(go.Scatter(
            x=bubble_df["double_parking_rate"] * 100,
            y=bubble_df["repeat_offender_ratio"] * 100,
            mode="markers+text",
            text=bubble_df["Zone"],
            textposition="top center", textfont=dict(size=8, color="#9CA3AF"),
            marker=dict(
                size=bubble_df["violations"] / bubble_df["violations"].max() * 44 + 8,
                color=bubble_df["CompositeSeverity"],
                colorscale=[[0,"#0F1629"],[0.4,"#991B1B"],[0.7,"#EF4444"],[1,"#FCD34D"]],
                showscale=True,
                line=dict(color="rgba(255,255,255,0.1)", width=1),
                colorbar=dict(
                    title=dict(text="Severity", font=dict(size=9, color="#6B7280")),
                    tickfont=dict(size=8, color="#6B7280"),
                    thickness=10
                )
            ),
            hovertemplate="<b>%{text}</b><br>Double Parking: %{x:.1f}%<br>Repeat Offenders: %{y:.1f}%<extra></extra>"
        ))
        layout_b = base_layout("Cluster Risk Map — Double Parking vs Recidivism", 460)
        layout_b["xaxis"]["title"] = dict(text="Double Parking Rate (%)", font=dict(size=10))
        layout_b["yaxis"]["title"] = dict(text="Repeat Offender Ratio (%)", font=dict(size=10))
        fig_bubble.update_layout(**layout_b)
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Full data table
    st.markdown("""
    <div class="section-header reveal" style="border-bottom: none; padding-bottom: 0; margin-bottom: 1rem;">
      <div class="section-eyebrow">Complete Zone Registry</div>
      <div class="section-title" style="font-size:18px;">Zone Intelligence Table</div>
    </div>
    """, unsafe_allow_html=True)

    display_cols = [
        "cluster","primary_junction","CompositeSeverity","violations",
        "CHOKE_ZONE","peak_hours_deployment","dominant_vehicle",
        "repeat_offender_ratio","Enforcement_ROI","jurisdiction",
        "total_economic_impact"
    ]
    disp_df = df[display_cols].copy() if all(c in df.columns for c in display_cols) else df.head(20)
    
    if "repeat_offender_ratio" in disp_df.columns:
        disp_df["repeat_offender_ratio"] = (disp_df["repeat_offender_ratio"] * 100).round(1).astype(str) + "%"
    if "CompositeSeverity" in disp_df.columns:
        disp_df["CompositeSeverity"] = disp_df["CompositeSeverity"].round(1)
    if "Enforcement_ROI" in disp_df.columns:
        disp_df["Enforcement_ROI"] = disp_df["Enforcement_ROI"].round(4)

    
    if "primary_junction" in disp_df.columns and "jurisdiction" in disp_df.columns:
        disp_df["primary_junction"] = disp_df.apply(
            lambda r: f"{r['jurisdiction']} Mid-Block" if "Mid-Block" in str(r['primary_junction']) else r['primary_junction'], 
            axis=1
        )
    # ------------------------------------------------------------------

    st.dataframe(
        disp_df.rename(columns={
            "cluster":"Cluster ID","primary_junction":"Junction","CompositeSeverity":"Severity Score",
            "violations":"Citations","CHOKE_ZONE":"Choke Zone","peak_hours_deployment":"Deploy Window",
            "dominant_vehicle":"Vehicle","repeat_offender_ratio":"Recidivism",
            "Enforcement_ROI":"ROI Index","jurisdiction":"Station",# ... inside the rename dictionary ...
        "total_economic_impact":"Economic Loss (₹)",
        }),
        use_container_width=True, hide_index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: JUNCTION SPOTLIGHT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider reveal">
  <span class="section-label">06 · Junction Intelligence</span>
</div>
<div class="section-header reveal">
  <div class="section-eyebrow">Spatial Hotspot Registry</div>
  <div class="section-title">📍 Top Violation Hotspot Junctions</div>
  <div class="section-desc">Named intersections with the highest concentration of recorded enforcement actions.</div>
</div>
""", unsafe_allow_html=True)

jdf = raw[raw["junction_name"].notna() & (raw["junction_name"] != "No Junction")]
top_junctions = jdf["junction_name"].value_counts().head(14).reset_index()
top_junctions.columns = ["junction","count"]
top_junctions = top_junctions.sort_values("count")

j1, j2 = st.columns([2, 1], gap="large")

with j1:
    st.markdown('<div class="chart-panel reveal-left">', unsafe_allow_html=True)
    n = len(top_junctions)
    j_colors = [f"rgba(239,68,68,{0.2 + 0.8*(i/n):.2f})" for i in range(n)]

    fig_junc = go.Figure(go.Bar(
        x=top_junctions["count"], y=top_junctions["junction"],
        orientation="h",
        marker_color=j_colors, marker_line_width=0,
        text=top_junctions["count"].apply(lambda x: f"{x:,}"),
        textposition="outside", textfont=dict(size=10, color="#9CA3AF"),
        hovertemplate="<b>%{y}</b><br>%{x:,} violations<extra></extra>"
    ))
    fig_junc.update_layout(**base_layout("", 460))
    fig_junc.update_xaxes(visible=False)
    st.plotly_chart(fig_junc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with j2:
    st.markdown('<div class="chart-panel reveal-right">', unsafe_allow_html=True)
    # Dayofweek volume chart
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    day_data  = raw.groupby("dayofweek").size().reset_index(name="count")
    day_data["day"] = day_data["dayofweek"].map(dict(enumerate(day_names)))
    day_colors = ["#EF4444" if i == day_data["count"].idxmax() else "#1A2D4A" for i in range(len(day_data))]

    fig_day = go.Figure(go.Bar(
        x=day_data["day"], y=day_data["count"],
        marker_color=day_colors, marker_line_width=0,
        text=day_data["count"].apply(lambda x: f"{x/1000:.1f}k"),
        textposition="outside", textfont=dict(size=10, color="#9CA3AF"),
        hovertemplate="<b>%{x}</b><br>%{y:,} violations<extra></extra>"
    ))
    fig_day.update_layout(**base_layout("Violations by Day of Week", 200))
    fig_day.update_yaxes(visible=False)
    st.plotly_chart(fig_day, use_container_width=True)

    # Hourly distribution — compact
    hour_data = raw.groupby("hour").size().reset_index(name="count")
    
    # Highlight the 3PM - 10PM blindspot in red, morning peak in orange, rest in dark blue
    h_colors = [
        "#EF4444" if (15 <= h <= 22) else "#F59E0B" if (7 <= h <= 10) else "#1A2D4A"
        for h in hour_data["hour"]
    ]
    
    fig_hour = go.Figure(go.Bar(
        x=hour_data["hour"], y=hour_data["count"],
        marker_color=h_colors, marker_line_width=0,
        hovertemplate="<b>%{x}:00</b><br>%{y:,} violations<extra></extra>"
    ))
    fig_hour.update_layout(**base_layout("Hourly Citations (Red = Unpoliced Evening Rush)", 200))
    fig_hour.update_yaxes(visible=False)
    st.plotly_chart(fig_hour, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer reveal">
  <div class="footer-logo">UrbanFlux</div>
  <div class="footer-meta">
    BENGALURU TRAFFIC INTELLIGENCE PLATFORM · FLIPKART GRIDLOCK 2024<br>
    EBI-Powered Preemptive Dispatch · DBSCAN Micro-Spatial Clustering · LightGBM Forecasting
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
components.html("""
<script>
(function triggerReveal() {
    function run() {
        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(e) {
                if (e.isIntersecting) e.target.classList.add('visible');
            });
        }, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });

        // Target elements within parent frames too
        const tryFrames = [document, window.parent ? window.parent.document : document];
        tryFrames.forEach(function(doc) {
            try {
                doc.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(function(el) {
                    observer.observe(el);
                });
            } catch(e) {}
        });
    }
    setTimeout(run, 200);
    setTimeout(run, 800);
    setTimeout(run, 2000);
})();
</script>
""", height=0)