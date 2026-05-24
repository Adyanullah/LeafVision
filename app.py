import os
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import urllib.request
# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LeafVision",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PROYEK = {
    "dataset": {
        "split_ratio": "80:10:10",
        "resolusi": "640×640",
        "distribusi_kelas": {
            "Common rust": 513,
            "Leaf Blight": 985,
            "Leaf Spot": 1192
        }
    },
    "augmentasi": {
        "rotasi": "±15 derajat",
        "flip": "p = 0.5",
        "brightness": "±20%",
        "contrast": "±15%",
        "crop": "90–100%",
        "blur": "kernel 3×3"
    },
    "training": {
        "optimizer": "SGD dengan momentum 0.9",
        "weight_decay": "0.0005",
        "batch_size": "4",
        "initial_lr": "0.005",
        "lr_step": "Step decay tiap 10 epoch (Gamma: 0.1)",
        "epochs": "100",
        "hardware": "GPU NVIDIA T4 (16GB VRAM)"
    },
    "evaluasi": {
        "map_50": "75,28%", "map_50_delta": "",
        "map_50_95": "78.3%", "map_50_95_delta": "",
        "precision": "50,50%", "precision_delta": "",
        "recall": "54,84%", "recall_delta": "",
        
        # Urutannya harus sama dengan nama kelas di atas
        "per_kelas_precision": [91.2, 88.5, 90.1],
        "per_kelas_recall": [95.3, 93.8, 94.2],
        "per_kelas_map50": [93.6, 91.2, 92.8]
    }
}

# (Otomatis menghitung total gambar dan total kelas agar Anda tidak perlu hitung manual)
DATA_PROYEK["dataset"]["total_gambar"] = sum(DATA_PROYEK["dataset"]["distribusi_kelas"].values())
DATA_PROYEK["dataset"]["total_kelas"] = len(DATA_PROYEK["dataset"]["distribusi_kelas"])

# ─────────────────────────────────────────────
# 2. GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ══════════════════════════════════════════════
   ROOT VARIABLES — Organic Tech Palette
   ══════════════════════════════════════════════ */
:root {
    --green-primary:  #73AA4E;
    --green-dark:     #2A5934;
    --green-darker:   #1C3D24;
    --green-light:    #A8CC8A;
    --green-pale:     #EAF4E3;
    --green-mist:     #F2F8EE;
    --cream:          #F7F5F0;
    --white:          #FFFFFF;
    --text-dark:      #1A2B1C;
    --text-mid:       #3D5C3F;
    --text-light:     #6B8C6D;
    --border:         rgba(115, 170, 78, 0.2);
    --border-strong:  rgba(115, 170, 78, 0.35);
    --shadow-sm:      0 2px 12px rgba(42, 89, 52, 0.07);
    --shadow-md:      0 6px 24px rgba(42, 89, 52, 0.13);
    --shadow-lg:      0 16px 48px rgba(42, 89, 52, 0.18);
    --radius-sm:      10px;
    --radius-md:      16px;
    --radius-lg:      20px;
    --transition:     all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ══════════════════════════════════════════════
   CUSTOM SCROLLBAR
   ══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--green-mist); }
::-webkit-scrollbar-thumb {
    background: var(--green-light);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: var(--green-primary); }

/* ══════════════════════════════════════════════
   BASE RESET
   ══════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}

/* ── Organic grain texture + base background ── */
.stApp {
    background-color: var(--cream);
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 200px 200px;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main content padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* ══════════════════════════════════════════════
   SIDEBAR — Deep Forest
   ══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background:
        linear-gradient(175deg, var(--green-darker) 0%, var(--green-dark) 55%, #2d6b3a 100%);
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding-top: 0 !important;
    position: relative;
    overflow: hidden;
}

/* Organic leaf watermark in sidebar background */
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    bottom: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(115,170,78,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    top: 30%;
    left: -80px;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(168,204,138,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

[data-testid="stSidebar"] * {
    color: #D4EDBA !important;
}

/* ── Sidebar Logo ── */
[data-testid="stSidebar"] [data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    margin: 0 auto !important;
    width: 100% !important;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));
}

/* ── Sidebar Radio Navigation ── */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 4px;
    display: flex;
    flex-direction: column;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.65rem 0.9rem !important;
    color: rgba(212, 237, 186, 0.65) !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    transition: var(--transition);
    cursor: pointer;
    width: 100% !important;
    display: block !important;
    position: relative;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.07) !important;
    padding-left: 1.2rem !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    color: #fff !important;
    font-weight: 600 !important;
    background: rgba(115,170,78,0.2) !important;
    border-left: 3px solid var(--green-primary) !important;
    padding-left: 0.75rem !important;
}

[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }

/* ── Sidebar Footer ── */
.sidebar-footer {
    position: relative !important;
    bottom: auto !important;
    margin-top: 0 !important;
    margin-bottom: 2rem !important;
    padding: 0.85rem 1rem;
    background: rgba(0,0,0,0.2);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(4px);
}

.sidebar-footer p {
    font-size: 0.7rem !important;
    color: rgba(212,237,186,0.45) !important;
    margin: 0 !important;
    line-height: 1.6;
}

/* ══════════════════════════════════════════════
   TYPOGRAPHY — Main Content
   ══════════════════════════════════════════════ */
.main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
    color: var(--green-dark) !important;
}

/* ══════════════════════════════════════════════
   PAGE HEADER
   ══════════════════════════════════════════════ */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid var(--border);
    position: relative;
}

.page-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 72px;
    height: 2px;
    background: linear-gradient(to right, var(--green-primary), var(--green-light));
    border-radius: 99px;
}

.page-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-pale);
    color: var(--green-dark);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 99px;
    border: 1px solid var(--border-strong);
    margin-bottom: 0.85rem;
}

.page-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: clamp(1.9rem, 4vw, 2.75rem) !important;
    font-weight: 400 !important;
    color: var(--green-dark) !important;
    line-height: 1.15 !important;
    margin: 0 0 0.5rem !important;
    letter-spacing: -0.5px;
}

.page-header .subtitle {
    color: var(--text-light);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    font-style: italic;
}

/* ══════════════════════════════════════════════
   CARDS
   ══════════════════════════════════════════════ */
.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(to right, var(--green-primary), var(--green-light));
    opacity: 0;
    transition: var(--transition);
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-3px);
    border-color: var(--border-strong);
}

.card:hover::before { opacity: 1; }

.card h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--green-dark);
    font-size: 1.2rem;
    font-weight: 400;
    margin: 0 0 0.75rem;
}

.card p {
    color: var(--text-mid);
    font-size: 0.88rem;
    line-height: 1.75;
    margin: 0;
}

/* ══════════════════════════════════════════════
   METRIC CARDS
   ══════════════════════════════════════════════ */
.metric-card {
    background: linear-gradient(135deg, var(--green-darker) 0%, var(--green-dark) 60%, #3a7a4a 100%);
    border-radius: var(--radius-md);
    padding: 1.6rem 1.5rem;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    transition: var(--transition);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -20px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(115,170,78,0.12);
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}

.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

.metric-card .label {
    font-size: 0.72rem;
    color: var(--green-light);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    position: relative;
    z-index: 1;
}

.metric-card .delta {
    font-size: 0.78rem;
    color: rgba(212, 237, 186, 0.75);
    margin-top: 0.4rem;
    position: relative;
    z-index: 1;
}

/* ══════════════════════════════════════════════
   INFO BOX
   ══════════════════════════════════════════════ */
.info-box {
    background: linear-gradient(to right, var(--green-pale), var(--green-mist));
    border: 1px solid rgba(115, 170, 78, 0.25);
    border-left: 4px solid var(--green-primary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1.25rem 1.5rem;
    margin: 1.25rem 0;
}

.info-box p {
    margin: 0;
    color: var(--green-darker);
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ══════════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--white) !important;
    margin-bottom: 0.75rem !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}

[data-testid="stExpander"]:hover {
    border-color: var(--border-strong) !important;
}

[data-testid="stExpander"] summary {
    padding: 1rem 1.25rem !important;
    font-weight: 500 !important;
    color: var(--text-dark) !important;
    font-size: 0.92rem !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--green-mist) !important;
    color: var(--green-dark) !important;
}

/* Expander content text color fix */
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
    color: var(--text-dark) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] td,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] th {
    color: var(--text-mid) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] code {
    color: var(--green-dark) !important;
    background: var(--green-pale) !important;
}
            
.e1i5pmia2 {
    color: var(--text-dark) !important;
}

.e1i5pmia2 [data-testid="stMetricValue"],
.e1i5pmia2 [data-testid="stMetricValue"] > div {
    color: var(--green-dark) !important;
}

.e1i5pmia2 [data-testid="stMetricLabel"],
.e1i5pmia2 [data-testid="stMetricLabel"] > div,
.e1i5pmia2 [data-testid="stMetricLabel"] p {
    color: var(--text-light) !important;
}

.e1i5pmia2 [data-testid="stMetricDelta"],
.e1i5pmia2 [data-testid="stMetricDelta"] > div {
    color: var(--text-mid) !important;
}

/* ══════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--green-pale) !important;
    border-radius: var(--radius-sm) !important;
    padding: 5px !important;
    border: 1px solid var(--border) !important;
    gap: 3px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: 0.5rem 1.1rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-mid) !important;
    background: transparent !important;
    border: none !important;
    transition: var(--transition);
}

.stTabs [aria-selected="true"] {
    background: var(--white) !important;
    color: var(--green-dark) !important;
    box-shadow: 0 2px 8px rgba(42,89,52,0.12) !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════════════
   FILE UPLOADER
   ══════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--green-primary) !important;
    border-radius: var(--radius-md) !important;
    background: var(--green-mist) !important;
    padding: 1.5rem !important;
    transition: var(--transition);
}

[data-testid="stFileUploader"]:hover {
    background: var(--green-pale) !important;
    border-color: var(--green-dark) !important;
}

/* ══════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--green-primary) 0%, var(--green-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.75rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.4px !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 14px rgba(42,89,52,0.28) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(42,89,52,0.38) !important;
    filter: brightness(1.05) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(42,89,52,0.25) !important;
}

/* ══════════════════════════════════════════════
   ALERTS & NOTIFICATIONS
   ══════════════════════════════════════════════ */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.875rem !important;
}

.stSuccess {
    background: var(--green-mist) !important;
    border-color: var(--green-light) !important;
    color: var(--green-darker) !important;
}

/* ══════════════════════════════════════════════
   CODE BLOCKS — On-brand styling
   ══════════════════════════════════════════════ */
.stCodeBlock {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

.stCodeBlock > div {
    background: var(--green-darker) !important;
}

code {
    background: var(--green-pale) !important;
    color: var(--green-darker) !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    font-size: 0.82em !important;
    border: 1px solid var(--border) !important;
}

pre code {
    background: transparent !important;
    color: #D4EDBA !important;
    border: none !important;
    padding: 0 !important;
}

/* ══════════════════════════════════════════════
   DATAFRAME / TABLES
   ══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

/* ══════════════════════════════════════════════
   SECTION DIVIDER
   ══════════════════════════════════════════════ */
.section-divider {
    height: 1px;
    background: linear-gradient(to right, var(--green-primary), var(--border), transparent);
    margin: 2.25rem 0;
    border: none;
    opacity: 0.6;
}

/* ══════════════════════════════════════════════
   TAG PILL
   ══════════════════════════════════════════════ */
.tag {
    display: inline-block;
    background: var(--green-pale);
    color: var(--green-dark);
    border: 1px solid var(--border-strong);
    border-radius: 99px;
    padding: 3px 13px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
    letter-spacing: 0.3px;
}

/* ══════════════════════════════════════════════
   STEP LIST
   ══════════════════════════════════════════════ */
.step-item {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 1.1rem 1.25rem;
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 0.75rem;
    transition: var(--transition);
}

.step-item:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-sm);
    transform: translateX(4px);
}

.step-num {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--green-primary), var(--green-dark));
    color: #fff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'DM Serif Display', serif;
    box-shadow: 0 3px 8px rgba(42,89,52,0.3);
}

.step-content h4 {
    margin: 0 0 0.35rem;
    font-size: 0.93rem;
    font-weight: 600;
    color: var(--text-dark);
}

.step-content p {
    margin: 0;
    font-size: 0.84rem;
    color: var(--text-light);
    line-height: 1.65;
}

/* ══════════════════════════════════════════════
   CONCLUSION CARD
   ══════════════════════════════════════════════ */
.conclusion-card {
    background: linear-gradient(135deg, var(--green-darker) 0%, var(--green-dark) 50%, #3d7a4d 100%);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: var(--shadow-lg);
}

.conclusion-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(115,170,78,0.18) 0%, transparent 70%);
}

.conclusion-card::after {
    content: '🍃';
    position: absolute;
    right: 2.5rem; top: 2rem;
    font-size: 5rem;
    opacity: 0.1;
    filter: grayscale(0.3);
}

.conclusion-card h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.65rem !important;
    font-weight: 400 !important;
    margin: 0 0 1rem !important;
    color: #ffffff !important;
    position: relative;
    z-index: 1;
}

.conclusion-card p {
    color: rgba(212, 237, 186, 0.88) !important;
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    margin: 0 !important;
    position: relative;
    z-index: 1;
}

/* ══════════════════════════════════════════════
   STREAMLIT NATIVE METRICS
   ══════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.25rem 1.5rem !important;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}

[data-testid="metric-container"]:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

[data-testid="metric-container"] label {
    color: var(--text-light) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-weight: 700 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: var(--green-dark) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * { color: var(--green-dark) !important; }

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * { color: var(--green-primary) !important; }

/* ══════════════════════════════════════════════
   SIDEBAR TOGGLE ICON
   ══════════════════════════════════════════════ */
[data-testid="collapsedControl"] {
    color: var(--green-dark) !important;
    background-color: var(--green-mist) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="collapsedControl"] svg {
    fill: var(--green-dark) !important;
    color: var(--green-dark) !important;
}

/* ══════════════════════════════════════════════
   SPINNER
   ══════════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: var(--green-primary) !important;
}

/* ══════════════════════════════════════════════
   RADIO LABEL (Sidebar label fix)
   ══════════════════════════════════════════════ */
[data-testid="stSidebar"] .stRadio > label {
    color: rgba(212, 237, 186, 0.5) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* ══════════════════════════════════════════════
   RESPONSIVE — MOBILE FIRST
   ══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .page-header h1 {
        font-size: 1.75rem !important;
    }

    .metric-card .value {
        font-size: 1.9rem;
    }

    .card {
        padding: 1.25rem !important;
    }

    .conclusion-card {
        padding: 1.5rem !important;
    }

    .conclusion-card::after {
        font-size: 3.5rem !important;
        top: 1rem !important;
        right: 1.25rem !important;
    }

    .step-item {
        padding: 0.9rem 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 0.75rem !important;
        font-size: 0.8rem !important;
    }
}

@media (max-width: 480px) {
    .page-header h1 {
        font-size: 1.5rem !important;
    }

    .metric-card {
        padding: 1.2rem 1rem;
    }

    .metric-card .value {
        font-size: 1.7rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2: # Kita letakkan logo khusus di kolom tengah
        try:
            # Kita gunakan use_column_width=True agar ukurannya otomatis pas
            st.image("assets/logo.png", use_column_width=True) 
        except FileNotFoundError:
            st.error("⚠️ Logo tidak ditemukan!")
            
    st.markdown("<br>", unsafe_allow_html=True)

    # Menu Navigasi
    menu = st.radio(
        "Navigasi Proyek:",
        [
            "Business Understanding",
            "Data Understanding",
            "Data Preparation",
            "Pemodelan",
            "Evaluasi",
            "Simulasi Deteksi",
            "Kesimpulan & Saran"
        ],
        label_visibility="collapsed"
    )

    # Footer
    st.markdown("""
    <div class="sidebar-footer">
        <p>Proyek MBKM • Computer Vision<br>Faster R-CNN + ResNet-50</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: Page Header
# ─────────────────────────────────────────────
def page_header(badge, title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
        <div class="badge">{badge}</div>
        <h1>{title}</h1>
        {'<p class="subtitle">' + subtitle + '</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def card(title, body):
    st.markdown(f"""
    <div class="card">
        <h3>{title}</h3>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)

def info_box(text):
    st.markdown(f"""
    <div class="info-box"><p>{text}</p></div>
    """, unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def metric_card(value, label, delta=""):
    delta_html = f'<div class="delta">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def step_item(num, title, desc):
    st.markdown(f"""
    <div class="step-item">
        <div class="step-num">{num}</div>
        <div class="step-content">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNGSI LOAD MODEL (CACHE RESOURCE)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    # 1. Panggil arsitektur bawaan ResNet-50 FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    
    # 2. Sesuaikan jumlah kelas (Background (1) + Penyakit (3) = 4 Kelas)
    num_classes = 4 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 3. Lacak path secara absolut & pastikan folder models ada
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True) 
    
    MODEL_PATH = os.path.join(MODEL_DIR, "faster_rcnn_best.pth")
    
    # 4. DOWNLOAD MODEL JIKA BELUM ADA DI FOLDER
    if not os.path.exists(MODEL_PATH):
        # GANTI URL INI DENGAN LINK GITHUB RELEASE MILIKMU NANTI:
        DOWNLOAD_URL = "https://github.com/username/nama-repo/releases/download/v1.0/faster_rcnn_best.pth"
        
        with st.spinner("Mengunduh bobot model dari server... Mohon tunggu sebentar."):
            try:
                urllib.request.urlretrieve(DOWNLOAD_URL, MODEL_PATH)
            except Exception as e:
                raise RuntimeError(f"Gagal mengunduh model. Pastikan URL benar dan ada koneksi internet. Error: {e}")
    
    # 5. Load file .pth
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # 6. Cek apakah ini Checkpoint Dictionary atau Raw Weights, lalu load dengan benar
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict']) # Ekstrak hanya bobotnya
    else:
        model.load_state_dict(checkpoint) # Load langsung jika sudah berupa raw weights
    
    model.eval() # Set ke mode evaluasi
    return model

CLASS_NAMES = ["Background", "Common rust", "Leaf Blight", "Leaf Spot"]
# ─────────────────────────────────────────────
# 4. HALAMAN
# ─────────────────────────────────────────────

# ── BUSINESS UNDERSTANDING ──
if menu == "Business Understanding":
    page_header(
        "01 / Business Understanding",
        "Latar Belakang & Tujuan",
        "Memahami konteks, masalah, dan target penelitian deteksi penyakit daun."
    )

    card(
        "Konteks Permasalahan",
        "Penyakit pada daun jagung dan padi merupakan salah satu faktor utama penyebab penurunan hasil panen dan kerugian finansial bagi petani Indonesia. Identifikasi manual membutuhkan keahlian khusus dan memakan waktu, sehingga diperlukan solusi berbasis teknologi."
    )

    info_box(
        "🎯 <strong>Tujuan Penelitian:</strong> Mengembangkan sistem deteksi penyakit daun otomatis menggunakan Computer Vision (Faster R-CNN) untuk membantu identifikasi dini yang cepat dan presisi di lapangan."
    )

    divider()
    st.markdown("### Ruang Lingkup & Target")

    col1, col2 = st.columns(2)
    with col1:
        card("🌽 Tanaman Target", "Daun <strong>jagung</strong> dengan berbagai kondisi: Common Rust, Leaf Blight, Leaf Spot.")
    with col2:
        card("📱 Pengguna Sasaran", "Petani, penyuluh pertanian, dan peneliti yang membutuhkan alat bantu identifikasi penyakit yang praktis dan mudah digunakan.")

    col3, col4 = st.columns(2)
    with col3:
        card("Metode Utama", "Object detection dengan <strong>Faster R-CNN</strong> dan ekstraktor fitur ResNet-50 untuk lokalisasi area penyakit secara spasial.")
    with col4:
        card("📏 Kriteria Keberhasilan", f"Model mencapai mAP@50 ≥ 75% pada dataset validasi dengan kemampuan deteksi real-time yang andal.")


# ── DATA UNDERSTANDING ──
elif menu == "Data Understanding":
    page_header(
        "02 / Data Understanding",
        "Eksplorasi Dataset",
        "Analisis distribusi, karakteristik, dan kualitas data citra yang digunakan."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(f"{DATA_PROYEK['dataset']['total_gambar']:,}", "Total Gambar", "✦ Siap Digunakan")
    with col2:
        metric_card(str(DATA_PROYEK['dataset']['total_kelas']), "Kelas Penyakit", "✦ Multi-class")
    with col3:
        metric_card(DATA_PROYEK['dataset']['resolusi'], "Resolusi Standar", "✦ Seragam")
    with col4:
        metric_card(DATA_PROYEK['dataset']['split_ratio'], "Split Train/Val/Test", "✦ Stratified")

    divider()

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("### Distribusi Kelas Dataset")
        # Mengambil data langsung dari DATA_PROYEK
        classes = DATA_PROYEK["dataset"]["distribusi_kelas"]
        df = pd.DataFrame(list(classes.items()), columns=["Kelas", "Jumlah"])
        df["Proporsi"] = (df["Jumlah"] / df["Jumlah"].sum() * 100).round(1)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Jumlah": st.column_config.ProgressColumn(
                    "Jumlah Sampel",
                    min_value=0, max_value=int(df["Jumlah"].max() * 1.1), format="%d"
                ),
                "Proporsi": st.column_config.NumberColumn("Proporsi (%)", format="%.1f%%")
            }
        )

    with col_right:
        st.markdown("### Karakteristik Data")
        info_box(
            "<strong>Format:</strong> JPEG/PNG, RGB 3-channel<br>"
            "<strong>Anotasi:</strong> Bounding box (PASCAL VOC format)<br>"
            "<strong>Sumber:</strong> Dataset publik New Bangladeshi Crop Disease (PlantVillage dataset)<br>"
            "<strong>Kondisi:</strong> Bervariasi (pencahayaan, jarak, sudut)"
        )

    divider()
    st.markdown("### Sampel Citra Daun")
    gal1, gal2, gal3 = st.columns(3)
    
    with gal1:
        with st.container(border=True):
            st.markdown('<h4 style="text-align:center; margin-top:0; color:var(--green-dark);">Common Rust</h4>', unsafe_allow_html=True)
            try:
                st.image("assets/Common Rust.jpg", use_column_width=True)
            except FileNotFoundError:
                st.error("⚠️ File assets/Common Rust.jpg tidak ditemukan!")
                
    with gal2:
        with st.container(border=True):
            st.markdown('<h4 style="text-align:center; margin-top:0; color:var(--green-dark);">Leaf Blight</h4>', unsafe_allow_html=True)
            try:
                st.image("assets/Leaf Blight.jpg", use_column_width=True)
            except FileNotFoundError:
                st.error("⚠️ File assets/Leaf Blight.jpg tidak ditemukan!")
                
    with gal3:
        with st.container(border=True):
            st.markdown('<h4 style="text-align:center; margin-top:0; color:var(--green-dark);">Leaf Spot</h4>', unsafe_allow_html=True)
            try:
                st.image("assets/Leaf Spot.jpg", use_column_width=True)
            except FileNotFoundError:
                st.error("⚠️ File assets/Leaf Spot.jpg tidak ditemukan!")


# ── DATA PREPARATION ──
elif menu == "Data Preparation":
    page_header(
        "03 / Data Preparation",
        "Pra-Pemrosesan & Augmentasi",
        "Serangkaian tahapan transformasi data sebelum pelatihan model."
    )

    step_item("1", "Resizing & Standarisasi Resolusi",
              f"Menyeragamkan seluruh gambar menjadi resolusi {DATA_PROYEK['dataset']['resolusi']} agar sesuai dengan input tensor Faster R-CNN + ResNet-50. Padding zero ditambahkan untuk menjaga aspek rasio asli.")

    step_item("2", "Augmentasi Data",
              f"Rotasi ({DATA_PROYEK['augmentasi']['rotasi']}), flip horizontal, penyesuaian brightness/contrast, dan random cropping untuk memperbanyak variasi data pelatihan dan mencegah overfitting pada model.")

    step_item("3", "Anotasi Bounding Box",
              "Menggunakan format PASCAL VOC dengan koordinat [x_min, y_min, x_max, y_max] untuk menandai area penyakit. Anotasi diverifikasi menggunakan LabelImg.")

    step_item("4", "Normalisasi Nilai Piksel",
              "Nilai piksel dinormalisasi ke rentang [0, 1] menggunakan mean dan std ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) karena menggunakan pretrained ResNet-50.")

    step_item("5", "Train / Validation Split",
              f"Dataset dibagi menjadi rasio {DATA_PROYEK['dataset']['split_ratio']} menggunakan stratified split agar distribusi kelas tetap seimbang.")

    divider()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 Detail Teknis: Augmentasi"):
            st.markdown(f"""
            | Teknik | Nilai |
            |--------|-------|
            | Rotasi | {DATA_PROYEK['augmentasi']['rotasi']} |
            | Horizontal Flip | {DATA_PROYEK['augmentasi']['flip']} |
            | Brightness | {DATA_PROYEK['augmentasi']['brightness']} |
            | Contrast | {DATA_PROYEK['augmentasi']['contrast']} |
            | Random Crop | {DATA_PROYEK['augmentasi']['crop']} |
            | Gaussian Blur | {DATA_PROYEK['augmentasi']['blur']} |
            """)
    with col2:
        with st.expander("📦 Library yang Digunakan"):
            st.markdown("""
            - **Albumentations** — augmentasi citra
            - **OpenCV** — manipulasi gambar
            - **PyTorch Transforms** — pipeline preprocessing
            - **LabelImg** — anotasi bounding box
            - **Scikit-learn** — stratified split
            """)


# ── PEMODELAN ──
elif menu == "Pemodelan":
    page_header(
        "04 / Pemodelan",
        "Arsitektur Faster R-CNN",
        "Desain dan konfigurasi model object detection yang digunakan."
    )

    info_box(
        "Model utama yang digunakan adalah <strong>Faster R-CNN</strong> dengan backbone <strong>ResNet-50 + FPN</strong>, "
        "dipilih karena kemampuannya dalam lokalisasi objek kecil (bercak penyakit) dengan akurasi tinggi."
    )

    divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Backbone ResNet-50", "🔺 FPN", "🎯 RPN & Head", "Hiperparameter"])

    with tab1:
        st.markdown("#### Ekstraksi Fitur: ResNet-50")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            card("Fungsi Backbone",
                 "ResNet-50 bertugas mengekstraksi feature map dari citra input. Arsitektur residual (skip connection) mengatasi masalah vanishing gradient, memungkinkan training jaringan yang sangat dalam secara efektif.")
            card("Transfer Learning",
                 "Bobot pretrained dari ImageNet digunakan sebagai titik awal, mempercepat konvergensi dan meningkatkan performa pada dataset yang relatif kecil.")
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align:left; padding: 1.5rem;">
                <div style="font-size:0.75rem;color:var(--green-light);letter-spacing:1px;text-transform:uppercase;margin-bottom:1rem;">Spesifikasi ResNet-50</div>
                <div style="color:rgba(212,237,186,0.9);font-size:0.85rem;line-height:2;">
                    Layers: 50<br>
                    Output stages: C2–C5<br>
                    Output channels: 256<br>
                    Params: ~23.5M<br>
                    Pretrained: ImageNet
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Feature Pyramid Network (FPN)")
        card("Multi-Scale Detection",
             "FPN menggabungkan feature map dari berbagai stage ResNet (C2–C5) untuk menghasilkan representasi multi-skala {P2, P3, P4, P5}. Hal ini krusial untuk mendeteksi bercak penyakit yang ukurannya sangat bervariasi dalam satu gambar.")
        card("Top-Down Pathway",
             "Fitur dari layer dalam (semantik tinggi) digabungkan dengan fitur dari layer awal (resolusi tinggi) via lateral connections, menghasilkan feature map yang kaya semantik namun detail spasial tetap terjaga.")

    with tab3:
        st.markdown("#### Region Proposal Network (RPN) & Detection Head")
        col1, col2 = st.columns(2)
        with col1:
            card("RPN — Proposal Generator",
                 "RPN bertugas mengusulkan region of interest (RoI) kandidat yang berpotensi mengandung penyakit. Menghasilkan anchor box di berbagai skala dan rasio aspek pada setiap posisi feature map.")
        with col2:
            card("RoI Align & Head",
                 "RoI Align mengekstraksi fitur fixed-size dari setiap proposal. Detection Head kemudian mengklasifikasi kelas penyakit dan meregresi koordinat bounding box final.")

    with tab4:
        st.markdown("#### Konfigurasi Training")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>Optimizer</h3>
                <p>{DATA_PROYEK['training']['optimizer']}<br>Weight decay: {DATA_PROYEK['training']['weight_decay']}<br>Batch size: {DATA_PROYEK['training']['batch_size']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Learning Rate</h3>
                <p>Initial LR: {DATA_PROYEK['training']['initial_lr']}<br>{DATA_PROYEK['training']['lr_step']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="card">
                <h3>Training</h3>
                <p>Epochs: {DATA_PROYEK['training']['epochs']}<br>Hardware: {DATA_PROYEK['training']['hardware']}<br>Framework: PyTorch</p>
            </div>
            """, unsafe_allow_html=True)


# ── EVALUASI ──
elif menu == "Evaluasi":
    page_header(
        "05 / Evaluasi",
        "Hasil & Performa Model",
        "Analisis komprehensif terhadap kinerja model pada data validasi."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("mAP@50", DATA_PROYEK['evaluasi']['map_50'], DATA_PROYEK['evaluasi']['map_50_delta'])
    with col2:
        st.metric("mAP@50:95", DATA_PROYEK['evaluasi']['map_50_95'], DATA_PROYEK['evaluasi']['map_50_95_delta'])
    with col3:
        st.metric("Precision", DATA_PROYEK['evaluasi']['precision'], DATA_PROYEK['evaluasi']['precision_delta'])
    with col4:
        st.metric("Recall", DATA_PROYEK['evaluasi']['recall'], DATA_PROYEK['evaluasi']['recall_delta'])

    divider()

    col_left, col_right = st.columns(2)

    # with col_left:
    #     st.markdown("### Performa per Kelas")
    #     perf_data = {
    #         "Kelas": list(DATA_PROYEK["dataset"]["distribusi_kelas"].keys()),
    #         "Precision": DATA_PROYEK['evaluasi']['per_kelas_precision'],
    #         "Recall": DATA_PROYEK['evaluasi']['per_kelas_recall'],
    #         "mAP@50": DATA_PROYEK['evaluasi']['per_kelas_map50']
    #     }
    #     df_perf = pd.DataFrame(perf_data)
    #     st.dataframe(df_perf, use_container_width=True, hide_index=True,
    #                  column_config={
    #                      "Precision": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    #                      "Recall": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    #                      "mAP@50": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    #                  })

    # with col_right:
    #     st.markdown("### Confusion Matrix")
    #     st.markdown("""
    #     <div style="background:#fff;border:1px solid var(--border);border-radius:16px;
    #                 height:260px;display:flex;align-items:center;justify-content:center;
    #                 color:var(--text-light);text-align:center;padding:2rem;">
    #         <div>
    #             <div style="font-size:2rem;margin-bottom:0.75rem;">📊</div>
    #             <div style="font-size:0.9rem;color:var(--text-mid);font-weight:500;">Confusion Matrix</div>
    #             <div style="font-size:0.8rem;color:var(--text-light);margin-top:0.3rem;">
    #                 <code>confusion_matrix.png</code><br>
    #             </div>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    # divider()
    st.markdown("### Training Loss Curve")
    try:
        # Mengambil gambar langsung dari folder assets
        st.image("assets/training_loss.png", use_column_width=True)
    except FileNotFoundError:
        st.markdown("""
                <div style="background:#fff;border:1px solid var(--border);border-radius:16px;
                            height:260px;display:flex;align-items:center;justify-content:center;
                            color:var(--text-light);text-align:center;padding:2rem;">
                    <div>
                        <div style="font-size:2rem;margin-bottom:0.75rem;">📉</div>
                        <div style="font-size:0.9rem;color:var(--text-mid);font-weight:500;">Loss Curve Tidak Ditemukan</div>
                        <div style="font-size:0.8rem;color:var(--text-light);margin-top:0.3rem;">
                            Pastikan file <code>training_loss.png</code><br>sudah ada di dalam folder <code>assets</code>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ── SIMULASI DETEKSI ──
elif menu == "Simulasi Deteksi":
    page_header(
        "06 / Simulasi Deteksi",
        "Live Demo Inferensi",
        "Unggah gambar daun untuk menguji model Faster R-CNN secara langsung."
    )

    info_box("📌 Unggah gambar daun jagung atau padi (format JPG/PNG). Model akan melakukan inferensi dan menampilkan bounding box beserta label prediksi.")

    uploaded_file = st.file_uploader(
        "Seret & lepas gambar di sini, atau klik untuk memilih",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded_file is not None:
        # Tampilkan gambar asli di kolom kiri
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Gambar Input")
            image = Image.open(uploaded_file).convert("RGB") # Buka dengan PIL
            st.image(image, use_column_width=True)

        # Proses AI
        with st.spinner("⏳ Memproses gambar dengan Faster R-CNN..."):
            # 1. Load Model
            model = load_model()
            
            # 2. Preprocessing (Ubah gambar ke Tensor)
            image_tensor = F.to_tensor(image).unsqueeze(0) # Bentuk: [1, C, H, W]
            
            # 3. Inferensi
            with torch.no_grad():
                prediction = model(image_tensor)[0]
            
            # 4. Post-processing (Menyaring Confidence Score & Menggambar Bounding Box)
            # Konversi PIL Image ke Array NumPy (Format RGB) agar bisa digambar oleh OpenCV
            img_result = np.array(image)
            
            # Batas minimal confidence (Bisa diatur, misal 0.5 = 50%)
            threshold = 0.5 
            
            # Ambil data yang melewati threshold
            boxes = prediction['boxes'][prediction['scores'] > threshold].numpy()
            labels = prediction['labels'][prediction['scores'] > threshold].numpy()
            scores = prediction['scores'][prediction['scores'] > threshold].numpy()
            
            # Siapkan teks detail HTML untuk di bawah gambar
            html_details = ""
            
            # Loop untuk menggambar setiap bounding box
            for box, label_idx, score in zip(boxes, labels, scores):
                x_min, y_min, x_max, y_max = map(int, box)
                class_name = CLASS_NAMES[label_idx]
                conf = score * 100
                
                # Gambar Kotak (Warna Merah RGB: 255,0,0)
                cv2.rectangle(img_result, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # Gambar Teks Label
                label_text = f"{class_name}: {conf:.1f}%"
                cv2.putText(img_result, label_text, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Tambahkan data ke string HTML
                html_details += f"""
                <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
                    <span style="background:var(--green-primary);color:white;padding:3px 10px;
                                 border-radius:20px;font-size:0.8rem;font-weight:600;">{class_name}</span>
                    <span style="font-size:0.85rem;color:var(--text-dark);font-weight:500;">Confidence: {conf:.1f}%</span>
                    <span style="font-size:0.75rem;color:var(--text-light); margin-left:auto;">BBox: [{x_min}, {y_min}, {x_max}, {y_max}]</span>
                </div>
                """

        # Tampilkan Hasil di kolom kanan
        with col2:
            st.markdown("#### 🎯 Hasil Deteksi")
            st.image(img_result, use_column_width=True) # Tampilkan gambar hasil OpenCV
            
            # Tampilkan detail box jika ada penyakit yang terdeteksi
            if len(boxes) > 0:
                st.markdown(f"""
                <div style="background:var(--green-pale);border:1px solid var(--border);
                            border-radius:12px;padding:1rem 1.25rem;margin-top:0.75rem;">
                    <div style="font-size:0.75rem;color:var(--text-light);letter-spacing:1px;
                                text-transform:uppercase;font-weight:600;margin-bottom:0.75rem;">
                        Rincian Prediksi
                    </div>
                    {html_details}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Tidak ada penyakit yang terdeteksi (Confidence di bawah threshold).")

        divider()


# ── KESIMPULAN & SARAN ──
elif menu == "Kesimpulan & Saran":
    page_header(
        "07 / Kesimpulan & Saran",
        "Ringkasan & Rekomendasi",
        "Temuan utama dan arah pengembangan penelitian selanjutnya."
    )

    st.markdown(f"""
    <div class="conclusion-card">
        <h2>Kesimpulan Penelitian</h2>
        <p>
        Model Faster R-CNN dengan backbone ResNet-50 dan FPN berhasil mendeteksi penyakit daun jagung dan padi 
        berdasarkan {DATA_PROYEK['dataset']['total_gambar']:,} citra uji dengan performa yang sangat memuaskan, mencapai <strong style="color:#D4EDBA;">mAP@50 sebesar {DATA_PROYEK['evaluasi']['map_50']}</strong> 
        pada data validasi. Sistem ini terbukti mampu melokalisasi area penyakit secara spasial dengan precision {DATA_PROYEK['evaluasi']['precision']} 
        dan recall {DATA_PROYEK['evaluasi']['recall']}, menjadikannya solusi yang andal untuk identifikasi dini di lapangan menggunakan training selama {DATA_PROYEK['training']['epochs']} epochs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Saran Pengembangan")

    col1, col2 = st.columns(2)
    with col1:
        card("📦 Perluasan Dataset",
             "Penambahan data dengan variasi kondisi pencahayaan (backlight, low-light), tahap keparahan penyakit yang lebih beragam, dan citra dari lokasi geografis berbeda untuk meningkatkan generalisasi model.")
        card("📱 Deployment Mobile",
             "Konversi model ke format ONNX atau TensorFlow Lite untuk memungkinkan deployment pada perangkat mobile, sehingga dapat digunakan petani langsung di lapangan secara offline.")
    with col2:
        card("Eksperimen Arsitektur Lain",
             "Eksplorasi arsitektur modern seperti YOLOv8 atau RT-DETR yang menawarkan keseimbangan lebih baik antara kecepatan inferensi dan akurasi, yang penting untuk aplikasi real-time.")
        card("🌐 Platform Web Produksi",
             "Pengembangan API backend menggunakan FastAPI dan integrasi dengan sistem informasi pertanian yang ada, memungkinkan penggunaan skala besar oleh dinas pertanian dan koperasi petani.")

    divider()

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        metric_card(DATA_PROYEK['evaluasi']['map_50'], "mAP@50 Dicapai", "✦ Target ≥ 75%")
    with col_stat2:
        metric_card(DATA_PROYEK['training']['epochs'], "Epoch Training", "✦ Konvergen stabil")
    with col_stat3:
        metric_card(str(DATA_PROYEK['dataset']['total_kelas']), "Kelas Terdeteksi", "✦ Semua berhasil")

    divider()
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;color:var(--text-light);font-size:0.85rem;">
        Proyek MBKM — LeafVision • Faster R-CNN Plant Disease Detection<br>
        <span style="color:var(--green-primary);">🍃</span> Dibangun dengan Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)