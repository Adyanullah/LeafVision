import os
import urllib.request
import numpy as np
import cv2
import streamlit as st
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from ultralytics import YOLO
    from ultralytics.nn import tasks as ultralytics_tasks

    # ─────────────────────────────────────────────
    # TRIPLET ATTENTION MODULE
    # Paper: "Rotate to Attend: Convolutional Triplet Attention Module" (WACV 2021)
    # HARUS identik dengan definisi yang dipakai saat training, agar:
    #   1. Unpickling checkpoint .pt berhasil (class harus ditemukan & sama strukturnya)
    #   2. Bobot state_dict cocok dengan arsitektur layer yang dibangun ulang di sini
    # ─────────────────────────────────────────────
    class ZPool(nn.Module):
        """Concatenate max-pool and avg-pool along channel dim."""
        def forward(self, x):
            return torch.cat([
                x.max(dim=1, keepdim=True)[0],
                x.mean(dim=1, keepdim=True)
            ], dim=1)

    class AttentionGate(nn.Module):
        """Single branch: spatial attention along one pair of axes."""
        def __init__(self, kernel_size=7):
            super().__init__()
            pad = kernel_size // 2
            self.compress = ZPool()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
            self.bn = nn.BatchNorm2d(1)

        def forward(self, x):
            attn = torch.sigmoid(self.bn(self.conv(self.compress(x))))
            return x * attn

    class TripletAttention(nn.Module):
        """
        Triplet Attention: three rotated attention branches
          Branch 1: H x W  (standard spatial, C-W-H rotated)
          Branch 2: C x W  (channel-width)
          Branch 3: C x H  (channel-height)
        """
        def __init__(self, kernel_size=7):
            super().__init__()
            self.branch1 = AttentionGate(kernel_size)  # H x W
            self.branch2 = AttentionGate(kernel_size)  # C x W
            self.branch3 = AttentionGate(kernel_size)  # C x H

        def forward(self, x):
            # Branch 1: normal H-W attention
            b1 = self.branch1(x)
            # Branch 2: rotate to C-W plane
            x2 = x.permute(0, 2, 1, 3)
            b2 = self.branch2(x2).permute(0, 2, 1, 3)
            # Branch 3: rotate to C-H plane
            x3 = x.permute(0, 3, 2, 1)
            b3 = self.branch3(x3).permute(0, 3, 2, 1)
            return (b1 + b2 + b3) / 3.0

    # Daftarkan class ke parser Ultralytics agar terbaca saat memuat
    # checkpoint/arsitektur YAML yang memuat layer TripletAttention
    ultralytics_tasks.TripletAttention = TripletAttention

    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PulmoVision | Deteksi Tumor Paru",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. KONFIGURASI MODEL — SESUAIKAN BAGIAN INI
#    Tidak perlu mendefinisikan nama kelas secara manual:
#    nama kelas otomatis diambil dari file model YOLOv8 (model.names)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIG = {
    # Path lokal tempat bobot model akan dicari/disimpan
    "local_path": os.path.join(BASE_DIR, "models", "yolov8_lung_tumor_best.pt"),
    # Ganti dengan URL GitHub Release / cloud storage tempat bobot .pt Anda disimpan
    "download_url": "https://github.com/USERNAME/REPO/releases/download/v1.0/yolov8_lung_tumor_best.pt",
    "default_conf": 0.25,
    "default_iou": 0.45,
}

# Palet warna untuk bounding box, dipilih berputar berdasarkan index kelas
BOX_PALETTE = [
    (220, 38, 38),   # merah
    (37, 99, 235),   # biru
    (217, 119, 6),   # oranye
    (5, 150, 105),   # hijau
    (124, 58, 237),  # ungu
]

# ─────────────────────────────────────────────
# 3. GLOBAL CSS — Tema "Clinical Tech" (biru-teal)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

:root {
    --blue-primary:  #2E86AB;
    --blue-dark:     #1B4F72;
    --blue-darker:   #112E40;
    --blue-light:    #8FC1DD;
    --blue-pale:     #E8F2F8;
    --blue-mist:     #F3F8FB;
    --cream:         #F7F8FA;
    --white:         #FFFFFF;
    --text-dark:     #16242B;
    --text-mid:      #3D5A66;
    --text-light:    #6B8694;
    --border:        rgba(46, 134, 171, 0.2);
    --border-strong: rgba(46, 134, 171, 0.35);
    --shadow-sm:      0 2px 12px rgba(27, 79, 114, 0.07);
    --shadow-md:      0 6px 24px rgba(27, 79, 114, 0.13);
    --shadow-lg:      0 16px 48px rgba(27, 79, 114, 0.18);
    --radius-sm:      10px;
    --radius-md:      16px;
    --radius-lg:      20px;
    --transition:     all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="stAppViewContainer"]::-webkit-scrollbar,
[data-testid="stSidebar"]::-webkit-scrollbar,
::-webkit-scrollbar { width: 6px; height: 6px; }

[data-testid="stAppViewContainer"]::-webkit-scrollbar-track,
[data-testid="stSidebar"]::-webkit-scrollbar-track,
::-webkit-scrollbar-track { background: var(--blue-mist); }

[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb,
[data-testid="stSidebar"]::-webkit-scrollbar-thumb,
::-webkit-scrollbar-thumb {
    background: var(--blue-light);
    border-radius: 99px;
    border: 2px solid var(--blue-mist);
}

[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb:hover,
[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover,
::-webkit-scrollbar-thumb:hover { background: var(--blue-primary); }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}

.stApp {
    background-color: var(--cream);
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 200px 200px;
}

#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stAppDeployButton"] { display: none !important; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, var(--blue-darker) 0%, var(--blue-dark) 55%, #1f5f86 100%);
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    position: relative;
    overflow: hidden;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(46,134,171,0.15) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

[data-testid="stSidebar"] * { color: #D6E9F4 !important; }

[data-testid="stSidebar"] [data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    margin: 0 auto !important;
}

[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background-color: var(--blue-light) !important;
}

[data-testid="stSidebar"] .sidebar-section-title {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    color: rgba(214, 233, 244, 0.55) !important;
    margin: 1.4rem 0 0.5rem !important;
}

.sidebar-footer {
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    padding: 0.85rem 1rem;
    background: rgba(0,0,0,0.2);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(255,255,255,0.07);
}

.sidebar-footer p {
    font-size: 0.7rem !important;
    color: rgba(214,233,244,0.5) !important;
    margin: 0 !important;
    line-height: 1.6;
}

/* TYPOGRAPHY */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 { color: var(--blue-dark) !important; }

/* PAGE HEADER */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid var(--border);
    position: relative;
}

.page-header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 72px; height: 2px;
    background: linear-gradient(to right, var(--blue-primary), var(--blue-light));
    border-radius: 99px;
}

.page-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--blue-pale);
    color: var(--blue-dark);
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
    color: var(--blue-dark) !important;
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

/* CARDS */
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
    background: linear-gradient(to right, var(--blue-primary), var(--blue-light));
    opacity: 0;
    transition: var(--transition);
}

.card:hover { box-shadow: var(--shadow-md); transform: translateY(-3px); border-color: var(--border-strong); }
.card:hover::before { opacity: 1; }

.card h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--blue-dark);
    font-size: 1.2rem;
    font-weight: 400;
    margin: 0 0 0.75rem;
}

.card p { color: var(--text-mid); font-size: 0.88rem; line-height: 1.75; margin: 0; }

/* METRIC CARDS */
.metric-card {
    background: linear-gradient(135deg, var(--blue-darker) 0%, var(--blue-dark) 60%, #2b7095 100%);
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

.metric-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-lg); }

.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

.metric-card .label {
    font-size: 0.72rem;
    color: var(--blue-light);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    position: relative;
    z-index: 1;
}

.metric-card .delta {
    font-size: 0.78rem;
    color: rgba(214, 233, 244, 0.75);
    margin-top: 0.4rem;
    position: relative;
    z-index: 1;
}

/* INFO BOX */
.info-box {
    background: linear-gradient(to right, var(--blue-pale), var(--blue-mist));
    border: 1px solid rgba(46, 134, 171, 0.25);
    border-left: 4px solid var(--blue-primary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1.25rem 1.5rem;
    margin: 1.25rem 0;
}

.info-box p { margin: 0; color: var(--blue-darker); font-size: 0.88rem; line-height: 1.7; }

/* EXPANDER */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--white) !important;
    margin-bottom: 0.75rem !important;
    box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] summary { padding: 1rem 1.25rem !important; font-weight: 500 !important; color: var(--text-dark) !important; font-size: 0.92rem !important; }
[data-testid="stExpander"] summary:hover { background: var(--blue-mist) !important; color: var(--blue-dark) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * { color: var(--text-dark) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] td,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] th { color: var(--text-mid) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] code { color: var(--blue-dark) !important; background: var(--blue-pale) !important; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--blue-primary) !important;
    border-radius: var(--radius-md) !important;
    background: var(--blue-mist) !important;
    padding: 1.5rem !important;
    transition: var(--transition);
}
[data-testid="stFileUploader"]:hover { background: var(--blue-pale) !important; border-color: var(--blue-dark) !important; }

/* BUTTONS */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--blue-primary) 0%, var(--blue-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.75rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.4px !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 14px rgba(27,79,114,0.28) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(27,79,114,0.38) !important;
    filter: brightness(1.05) !important;
}

/* ALERTS */
.stAlert { border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important; font-size: 0.875rem !important; }
.stSuccess { background: var(--blue-mist) !important; border-color: var(--blue-light) !important; color: var(--blue-darker) !important; }

/* DATAFRAME */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important; overflow: hidden; }

/* DIVIDER */
hr.section-divider {
    height: 1px !important;
    background: linear-gradient(to right, var(--blue-primary), var(--blue-light), transparent) !important;
    margin: 2.25rem 0 !important;
    border: none !important;
    opacity: 0.8 !important;
}

/* TAG PILL */
.tag {
    display: inline-block;
    background: var(--blue-pale);
    color: var(--blue-dark);
    border: 1px solid var(--border-strong);
    border-radius: 99px;
    padding: 3px 13px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
    letter-spacing: 0.3px;
}

/* NATIVE METRICS */
[data-testid="metric-container"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.25rem 1.5rem !important;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}
[data-testid="metric-container"]:hover { border-color: var(--border-strong); box-shadow: var(--shadow-md); transform: translateY(-2px); }
[data-testid="metric-container"] label { color: var(--text-light) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1.1px; font-weight: 700 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--blue-dark) !important; font-family: 'DM Serif Display', serif !important; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color: var(--blue-primary) !important; }

[data-testid="collapsedControl"] {
    background-color: var(--blue-mist) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    color: var(--blue-dark) !important;
}
[data-testid="collapsedControl"] svg { fill: var(--blue-dark) !important; }

.stSpinner > div { border-top-color: var(--blue-primary) !important; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .page-header h1 { font-size: 1.75rem !important; }
    .metric-card .value { font-size: 1.8rem; }
    .card { padding: 1.25rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. HELPER UI
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
    <div class="card"><h3>{title}</h3><p>{body}</p></div>
    """, unsafe_allow_html=True)


def info_box(text):
    st.markdown(f'<div class="info-box"><p>{text}</p></div>', unsafe_allow_html=True)


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


# ─────────────────────────────────────────────
# 5. LOAD MODEL YOLOv8 (CACHE RESOURCE)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not ULTRALYTICS_OK:
        raise RuntimeError(
            "Library 'ultralytics' belum terpasang. Jalankan: pip install ultralytics"
        )

    model_dir = os.path.dirname(MODEL_CONFIG["local_path"])
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(MODEL_CONFIG["local_path"]):
        with st.spinner("Mengunduh bobot model YOLOv8 dari server... Mohon tunggu sebentar."):
            try:
                urllib.request.urlretrieve(MODEL_CONFIG["download_url"], MODEL_CONFIG["local_path"])
            except Exception as e:
                raise RuntimeError(
                    f"Gagal mengunduh model. Pastikan 'download_url' pada MODEL_CONFIG sudah benar "
                    f"dan ada koneksi internet. Error: {e}"
                )

    model = YOLO(MODEL_CONFIG["local_path"])
    return model


# ─────────────────────────────────────────────
# 6. SIDEBAR — Pengaturan Deteksi
# ─────────────────────────────────────────────
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div style="text-align:center;font-size:2.4rem;line-height:1;">🫁</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="text-align:center;font-family:\'DM Serif Display\',serif;'
            'font-size:1.15rem;color:#fff;margin-top:0.3rem;">PulmoVision</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-section-title">Pengaturan Deteksi</div>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold", min_value=0.05, max_value=0.95,
        value=MODEL_CONFIG["default_conf"], step=0.05
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)", min_value=0.05, max_value=0.95,
        value=MODEL_CONFIG["default_iou"], step=0.05
    )

    st.markdown("""
    <div class="sidebar-footer">
        <p>PulmoVision<br>YOLOv8 — Deteksi Tumor Paru<br>pada citra CT-Scan / X-Ray</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 7. KONTEN HALAMAN — SIMULASI / INFERENSI
# ─────────────────────────────────────────────
page_header(
    "Simulasi Deteksi",
    "Deteksi Tumor Paru",
    "Unggah gambar CT-Scan / X-Ray paru untuk menguji model YOLOv8 secara langsung."
)

info_box(
    "📌 Unggah gambar paru (format JPG/PNG/BMP). Model YOLOv8 akan melakukan inferensi dan "
    "menampilkan bounding box beserta label serta tingkat keyakinan (confidence) untuk setiap "
    "area yang terindikasi tumor."
)

with st.expander("⚙️ Catatan Konfigurasi Model"):
    st.markdown("""
    - Letakkan bobot model hasil training (`.pt`) pada folder `models/`, atau atur
      `download_url` pada `MODEL_CONFIG` agar otomatis diunduh saat pertama kali dijalankan.
    - Nama kelas **tidak perlu ditulis manual** — diambil otomatis dari `model.names` sesuai
      label yang digunakan saat training YOLOv8 (misalnya `Benign`, `Malignant`, dsb).
    - Sesuaikan `Confidence Threshold` dan `IoU Threshold` pada sidebar sesuai karakteristik
      model Anda.
    - Model ini menggunakan layer custom **TripletAttention**. Definisinya sudah disertakan
      langsung di bagian atas file ini dan didaftarkan ke `ultralytics.nn.tasks` sebelum model
      di-load, sehingga checkpoint `.pt` hasil training Anda bisa di-unpickle dengan benar.
      Jika arsitektur custom Anda berbeda dari yang ada di sini, salin ulang definisi
      class-nya persis seperti saat training agar bobot tetap cocok.
    """)

uploaded_file = st.file_uploader(
    "Seret & lepas gambar di sini, atau klik untuk memilih",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📷 Gambar Input")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    boxes, confs, cls_idx, names = [], [], [], {}
    inference_error = None

    with st.spinner("⏳ Memproses gambar dengan YOLOv8..."):
        try:
            model = load_model()
            results = model.predict(
                source=np.array(image),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            r = results[0]
            names = r.names  # dict {index: nama_kelas} sesuai training

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_idx = r.boxes.cls.cpu().numpy().astype(int)
        except Exception as e:
            inference_error = str(e)

    if inference_error:
        st.error(f"⚠️ Terjadi kesalahan saat inferensi: {inference_error}")
    else:
        img_result = np.array(image).copy()
        html_details = ""

        for box, label_idx, score in zip(boxes, cls_idx, confs):
            x_min, y_min, x_max, y_max = map(int, box)
            class_name = names.get(int(label_idx), f"Kelas {label_idx}")
            conf_pct = score * 100
            color = BOX_PALETTE[int(label_idx) % len(BOX_PALETTE)]

            cv2.rectangle(img_result, (x_min, y_min), (x_max, y_max), color, 2)
            label_text = f"{class_name}: {conf_pct:.1f}%"
            cv2.putText(
                img_result, label_text, (x_min, max(y_min - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

            html_details += f"""
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;flex-wrap:wrap;">
                <span class="tag">{class_name}</span>
                <span style="font-size:0.85rem;color:var(--text-dark);font-weight:500;">Confidence: {conf_pct:.1f}%</span>
                <span style="font-size:0.75rem;color:var(--text-light);margin-left:auto;">BBox: [{x_min}, {y_min}, {x_max}, {y_max}]</span>
            </div>
            """

        with col2:
            st.markdown("#### 🎯 Hasil Deteksi")
            st.image(img_result, use_column_width=True)

            if len(boxes) > 0:
                card_html = (
                    '<div style="background:var(--blue-pale);border:1px solid var(--border);'
                    'border-radius:12px;padding:1rem 1.25rem;margin-top:0.75rem;">'
                    '<div style="font-size:0.75rem;color:var(--text-light);letter-spacing:1px;'
                    'text-transform:uppercase;font-weight:600;margin-bottom:0.75rem;">'
                    'Rincian Prediksi'
                    '</div>'
                    + html_details +
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                _, buf = cv2.imencode(".png", cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
                st.download_button(
                    "⬇️ Unduh Gambar Hasil Deteksi",
                    data=buf.tobytes(),
                    file_name="hasil_deteksi_tumor_paru.png",
                    mime="image/png"
                )
            else:
                st.warning("Tidak ada area tumor yang terdeteksi pada confidence threshold saat ini.")

        if len(boxes) > 0:
            divider()
            st.markdown("### Ringkasan Deteksi")
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            with sum_col1:
                metric_card(str(len(boxes)), "Total Objek Terdeteksi")
            with sum_col2:
                metric_card(f"{confs.max() * 100:.1f}%", "Confidence Tertinggi")
            with sum_col3:
                kelas_terbanyak_idx = np.bincount(cls_idx).argmax()
                metric_card(names.get(int(kelas_terbanyak_idx), "-"), "Kelas Terbanyak")

divider()
st.markdown("""
<div style="text-align:center;padding:1.5rem;color:var(--text-light);font-size:0.85rem;">
    PulmoVision • YOLOv8 Lung Tumor Detection<br>
    <span style="color:var(--blue-primary);">🫁</span> Dibangun dengan Streamlit & Ultralytics YOLOv8
</div>
""", unsafe_allow_html=True)