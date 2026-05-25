import streamlit as st
import random
import math
import os
import pickle
import joblib
import hashlib
import json
import numpy as np
import pandas as pd
from groq import Groq
import plotly.graph_objects as go

# ─── CARGAR MODELOS PKL ───────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    import os
    models = {}
    # Buscar los PKL en el mismo directorio que este script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = {
        "xgb_zona":           "xgb_zona.pkl",
        "lgbm_zona":          "lgbm_zona.pkl",
        "le_zona":            "le_target_zona.pkl",
        "encoders_zona":      "encoders_zona.pkl",
        "xgb_gravedad":       "xgb_gravedad.pkl",
        "pipe_lgbm_gravedad": "pipe_lgbm_gravedad.pkl",
        "le_gravedad":        "le_target_gravedad.pkl",
        "preprocessor_grav":  "preprocessor_gravedad.pkl",
        "scaler_gravedad":    "scaler_gravedad.pkl",
    }
    for key, fname in files.items():
        # Intentar en el directorio del script y en el directorio de trabajo
        for path in [os.path.join(base_dir, fname), fname]:
            try:
                models[key] = joblib.load(path)
                break
            except Exception:
                models[key] = None
    return models

MODELS = load_models()
MODELS_OK = any(v is not None for v in MODELS.values())

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeHer Colombia · IA Protección",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<script>
(function injectSidebarStyles(){
    var id='sh-sidebar-css';
    if(document.getElementById(id))return;
    var s=document.createElement('style');
    s.id=id;
    s.textContent='section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div:first-child{background:#FDFBFF!important}'
    +'section[data-testid="stSidebar"]{border-right:1.5px solid #EDE9FE!important;}'
    +'section[data-testid="stSidebar"] [data-baseweb="radio"]>div:first-child{display:none!important}'
    +'section[data-testid="stSidebar"] label[data-baseweb="radio"]{'
    +'padding:9px 14px!important;border-radius:12px!important;font-size:13px!important;'
    +'font-weight:600!important;color:#6D28D9!important;border:1.5px solid transparent!important;'
    +'margin-bottom:2px!important;cursor:pointer!important;transition:background 0.15s!important;}'
    +'section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{'
    +'background:#F0EBFF!important;border-color:#DDD6FE!important;color:#4C1D95!important;}';
    document.head.appendChild(s);
})();
setTimeout(function(){
    var s=document.getElementById('sh-sidebar-css');
    if(s){s.parentNode.removeChild(s);}
    var ns=document.createElement('style');
    ns.id='sh-sidebar-css';
    ns.textContent='section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div:first-child{background:#FDFBFF!important}'
    +'section[data-testid="stSidebar"]{border-right:1.5px solid #EDE9FE!important;}'
    +'section[data-testid="stSidebar"] [data-baseweb="radio"]>div:first-child{display:none!important}'
    +'section[data-testid="stSidebar"] label[data-baseweb="radio"]{'
    +'padding:9px 14px!important;border-radius:12px!important;font-size:13px!important;'
    +'font-weight:600!important;color:#6D28D9!important;border:1.5px solid transparent!important;'
    +'margin-bottom:2px!important;cursor:pointer!important;}'
    +'section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{'
    +'background:#F0EBFF!important;border-color:#DDD6FE!important;color:#4C1D95!important;}';
    document.head.appendChild(ns);
},800);
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important;
    background: #F5F3FF !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #FDFBFF !important;
}

[data-testid="stSidebar"] {
    border-right: 1.5px solid #EDE9FE !important;
    min-width: 252px !important;
    max-width: 252px !important;
    box-shadow: 4px 0 24px rgba(124,58,237,0.08) !important;
}

[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6D28D9 !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
    margin-bottom: 2px !important;
    white-space: nowrap !important;
    border: 1.5px solid transparent !important;
    background: transparent !important;
}

[data-testid="stSidebar"] .stRadio label:hover,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #F0EBFF !important;
    color: #4C1D95 !important;
    border-color: #DDD6FE !important;
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #EDE9FE !important;
    color: #3B1FA8 !important;
    border-color: #C4B5FD !important;
}

[data-testid="stSidebar"] input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
[data-testid="stSidebar"] .element-container { margin: 0 !important; padding: 0 10px !important; }
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none !important; }

.main .block-container {
    padding: 24px 36px !important;
    max-width: 100% !important;
    background: #F5F3FF !important;
}

.sh-card {
    background: #FFFFFF;
    border-radius: 20px;
    border: 1px solid #EDE9FE;
    box-shadow: 0 2px 20px rgba(109,40,217,0.07);
    padding: 22px;
    margin-bottom: 16px;
}

.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    padding: 10px 20px !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.15) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(109,40,217,0.25) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5B21B6, #7C3AED) !important;
    color: white !important;
}

.stButton > button[kind="secondary"] {
    background: #F5F3FF !important;
    color: #5B21B6 !important;
    border: 1px solid #C4B5FD !important;
}

.stSelectbox [data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #DDD6FE !important;
    font-family: inherit !important;
    font-size: 13px !important;
    background: #FAFAFA !important;
    transition: border 0.18s !important;
}

.stSelectbox [data-baseweb="select"]:focus-within > div,
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #EDE9FE;
    border-radius: 14px;
    padding: 5px;
    border: none;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #5B21B6 !important;
    padding: 8px 16px !important;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.12) !important;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #EDE9FE;
    border-radius: 18px;
    padding: 16px !important;
    box-shadow: 0 2px 12px rgba(109,40,217,0.06);
}

.stCheckbox label, [data-testid="stCheckbox"] label { font-size: 13px !important; }
.stSpinner > div > div { border-top-color: #7C3AED !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, #A78BFA, #7C3AED) !important; border-radius: 99px !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F5F3FF; }
::-webkit-scrollbar-thumb { background: #C4B5FD; border-radius: 99px; }

.stAlert { border-radius: 16px !important; }

.risk-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
    white-space: nowrap;
}

.chat-user {
    background: linear-gradient(135deg, #5B21B6, #7C3AED);
    color: white;
    border-radius: 20px 4px 20px 20px;
    padding: 13px 18px;
    font-size: 13px;
    line-height: 1.75;
    max-width: 76%;
    margin-left: auto;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(124,58,237,0.22);
}

.chat-sara {
    background: #FAF8FF;
    color: #1E1B4B;
    border-radius: 4px 20px 20px 20px;
    padding: 13px 18px;
    font-size: 13px;
    line-height: 1.8;
    max-width: 76%;
    border: 1px solid #EDE9FE;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(109,40,217,0.05);
}

[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

a:hover { opacity: 0.88; }
.block-container { padding-top: 16px !important; }
.element-container:empty { display: none !important; }
div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA ─────────────────────────────────────────────────────────────────────

RISK_LEVELS = {
    "MÍNIMO":     {"color": "#059669", "bg": "#ECFDF5", "label": "Mínimo"},
    "MUY BAJO":   {"color": "#10B981", "bg": "#D1FAE5", "label": "Muy Bajo"},
    "BAJO":       {"color": "#3B82F6", "bg": "#EFF6FF", "label": "Bajo"},
    "MEDIO-BAJO": {"color": "#F59E0B", "bg": "#FFFBEB", "label": "Medio-Bajo"},
    "MEDIO-ALTO": {"color": "#EF4444", "bg": "#FEF2F2", "label": "Medio-Alto"},
    "ALTO":       {"color": "#DC2626", "bg": "#FEF2F2", "label": "Alto"},
    "MUY ALTO":   {"color": "#991B1B", "bg": "#FEE2E2", "label": "Muy Alto"},
    "CRÍTICO":    {"color": "#7F1D1D", "bg": "#FEE2E2", "label": "Crítico"},
}

DEPARTAMENTOS = [
    "AMAZONAS","ANTIOQUIA","ARAUCA","ATLÁNTICO","BOGOTÁ D.C.","BOLÍVAR","BOYACÁ","CALDAS",
    "CAQUETÁ","CASANARE","CAUCA","CESAR","CHOCÓ","CÓRDOBA","CUNDINAMARCA","GUAINÍA",
    "GUAVIARE","HUILA","LA GUAJIRA","MAGDALENA","META","NARIÑO","NORTE DE SANTANDER",
    "PUTUMAYO","QUINDÍO","RISARALDA","SAN ANDRÉS","SANTANDER","SUCRE","TOLIMA",
    "VALLE DEL CAUCA","VAUPÉS","VICHADA"
]

DELITOS = ["VIOLENCIA INTRAFAMILIAR","VIOLENCIA SEXUAL","LESIONES PERSONALES","AMENAZAS","HURTO","HOMICIDIO"]

MUNICIPIOS_SAMPLE = {
    "ANTIOQUIA":       ["MEDELLÍN","BELLO","ITAGÜÍ","ENVIGADO","APARTADÓ","TURBO"],
    "BOGOTÁ D.C.":     ["BOGOTÁ"],
    "VALLE DEL CAUCA": ["CALI","BUENAVENTURA","PALMIRA","TULUÁ","BUGA"],
    "CUNDINAMARCA":    ["SOACHA","FACATATIVÁ","ZIPAQUIRÁ","FUSAGASUGÁ","GIRARDOT"],
    "ATLÁNTICO":       ["BARRANQUILLA","SOLEDAD","MALAMBO","SABANAGRANDE","SABANALARGA"],
    "SANTANDER":       ["BUCARAMANGA","FLORIDABLANCA","GIRÓN","PIEDECUESTA","BARRANCABERMEJA"],
    "NARIÑO":          ["PASTO","TUMACO","IPIALES","TÚQUERRES","LA UNIÓN"],
    "CÓRDOBA":         ["MONTERÍA","CERETÉ","LORICA","SAHAGÚN","TIERRALTA"],
    "BOLÍVAR":         ["CARTAGENA","MAGANGUÉ","EL CARMEN","MOMPÓS","TURBACO"],
    "TOLIMA":          ["IBAGUÉ","ESPINAL","MELGAR","HONDA","CHAPARRAL"],
    "CAUCA":           ["POPAYÁN","SANTANDER DE QUILICHAO","MIRANDA","CALOTO","PATÍA"],
    "META":            ["VILLAVICENCIO","ACACÍAS","GRANADA","CUMARAL","RESTREPO"],
    "HUILA":           ["NEIVA","PITALITO","GARZÓN","LA PLATA","CAMPOALEGRE"],
    "BOYACÁ":          ["TUNJA","DUITAMA","SOGAMOSO","CHIQUINQUIRÁ","PAIPA"],
    "CALDAS":          ["MANIZALES","LA DORADA","RIOSUCIO","SUPÍA","MANZANARES"],
    "RISARALDA":       ["PEREIRA","DOSQUEBRADAS","SANTA ROSA","LA VIRGINIA","BELÉN DE UMBRÍA"],
    "QUINDÍO":         ["ARMENIA","CALARCÁ","Montenegro","QUIMBAYA","CIRCASIA"],
    "NORTE DE SANTANDER":["CÚCUTA","OCAÑA","PAMPLONA","VILLA DEL ROSARIO","LOS PATIOS"],
    "CESAR":           ["VALLEDUPAR","AGUACHICA","CODAZZI","LA PAZ","CHIMICHAGUA"],
    "MAGDALENA":       ["SANTA MARTA","CIÉNAGA","FUNDACIÓN","PLATO","EL BANCO"],
    "SUCRE":           ["SINCELEJO","COROZAL","SAMPUÉS","TOLÚ","SAN MARCOS"],
    "LA GUAJIRA":      ["RIOHACHA","MAICAO","URIBIA","MANAURE","FONSECA"],
    "CAQUETÁ":         ["FLORENCIA","SAN VICENTE","BELÉN DE LOS ANDAQUÍES","CARTAGENA DEL CHAIRÁ","MILÁN"],
    "ARAUCA":          ["ARAUCA","SARAVENA","TAME","ARAUQUITA","FORTUL"],
    "CASANARE":        ["YOPAL","AGUAZUL","MONTERREY","VILLANUEVA","OROCUÉ"],
    "PUTUMAYO":        ["MOCOA","PUERTO ASÍS","ORITO","VILLAGARZÓN","SIBUNDOY"],
    "CHOCÓ":           ["QUIBDÓ","ISTMINA","TUMACO","BAHÍA SOLANO","NUQUÍ"],
    "GUAVIARE":        ["SAN JOSÉ DEL GUAVIARE","CALAMAR","EL RETORNO","MIRAFLORES"],
    "VICHADA":         ["PUERTO CARREÑO","LA PRIMAVERA","SANTA ROSALÍA","CUMARIBO"],
    "GUAINÍA":         ["INÍRIDA","BARRANCO MINAS","MAPIRIPANA","SAN FELIPE"],
    "VAUPÉS":          ["MITÚ","CARURÚ","PACOA","YAVARATÉ","TARAIRA"],
    "AMAZONAS":        ["LETICIA","PUERTO NARIÑO","LA CHORRERA","LA PEDRERA","MIRITÍ-PARANÁ"],
    "SAN ANDRÉS":      ["SAN ANDRÉS","PROVIDENCIA"],
}

def get_municipios(dep):
    return MUNICIPIOS_SAMPLE.get(dep, ["Capital","Municipio 1","Municipio 2"])

CRIME_DATA = {
    "ANTIOQUIA":          {"score": 4.2, "zona": "ALTO",      "gravedad": "ALTO",      "municipios": 125},
    "BOGOTÁ D.C.":        {"score": 3.8, "zona": "MEDIO-ALTO","gravedad": "MEDIO-ALTO","municipios": 1},
    "VALLE DEL CAUCA":    {"score": 4.5, "zona": "MUY ALTO",  "gravedad": "ALTO",      "municipios": 42},
    "CUNDINAMARCA":       {"score": 2.9, "zona": "MEDIO-BAJO","gravedad": "BAJO",      "municipios": 116},
    "ATLÁNTICO":          {"score": 3.1, "zona": "MEDIO-BAJO","gravedad": "MEDIO-BAJO","municipios": 23},
    "SANTANDER":          {"score": 2.5, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 87},
    "NARIÑO":             {"score": 3.7, "zona": "MEDIO-ALTO","gravedad": "MEDIO-ALTO","municipios": 64},
    "CÓRDOBA":            {"score": 3.0, "zona": "MEDIO-BAJO","gravedad": "BAJO",      "municipios": 30},
    "BOLÍVAR":            {"score": 3.4, "zona": "MEDIO-BAJO","gravedad": "MEDIO-BAJO","municipios": 46},
    "TOLIMA":             {"score": 2.7, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 47},
    "HUILA":              {"score": 2.8, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 37},
    "CAUCA":              {"score": 4.0, "zona": "ALTO",      "gravedad": "ALTO",      "municipios": 42},
    "META":               {"score": 3.3, "zona": "MEDIO-BAJO","gravedad": "MEDIO-BAJO","municipios": 29},
    "CESAR":              {"score": 3.2, "zona": "MEDIO-BAJO","gravedad": "MEDIO-BAJO","municipios": 25},
    "MAGDALENA":          {"score": 3.0, "zona": "MEDIO-BAJO","gravedad": "BAJO",      "municipios": 30},
    "BOYACÁ":             {"score": 2.2, "zona": "MUY BAJO",  "gravedad": "MUY BAJO",  "municipios": 123},
    "CALDAS":             {"score": 2.6, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 27},
    "RISARALDA":          {"score": 2.8, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 14},
    "QUINDÍO":            {"score": 2.5, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 12},
    "NORTE DE SANTANDER": {"score": 3.6, "zona": "MEDIO-ALTO","gravedad": "MEDIO-ALTO","municipios": 40},
    "SUCRE":              {"score": 2.9, "zona": "MEDIO-BAJO","gravedad": "BAJO",      "municipios": 26},
    "LA GUAJIRA":         {"score": 3.5, "zona": "MEDIO-ALTO","gravedad": "MEDIO-BAJO","municipios": 15},
    "CAQUETÁ":            {"score": 3.8, "zona": "ALTO",      "gravedad": "MEDIO-ALTO","municipios": 16},
    "ARAUCA":             {"score": 3.9, "zona": "ALTO",      "gravedad": "ALTO",      "municipios": 7},
    "CASANARE":           {"score": 2.7, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 19},
    "VICHADA":            {"score": 2.3, "zona": "MUY BAJO",  "gravedad": "MUY BAJO",  "municipios": 4},
    "GUAINÍA":            {"score": 2.1, "zona": "MUY BAJO",  "gravedad": "MÍNIMO",    "municipios": 8},
    "GUAVIARE":           {"score": 3.2, "zona": "MEDIO-BAJO","gravedad": "MEDIO-BAJO","municipios": 4},
    "VAUPÉS":             {"score": 2.0, "zona": "MUY BAJO",  "gravedad": "MÍNIMO",    "municipios": 6},
    "AMAZONAS":           {"score": 2.1, "zona": "MUY BAJO",  "gravedad": "MÍNIMO",    "municipios": 9},
    "PUTUMAYO":           {"score": 3.6, "zona": "MEDIO-ALTO","gravedad": "MEDIO-ALTO","municipios": 13},
    "CHOCÓ":              {"score": 4.1, "zona": "ALTO",      "gravedad": "ALTO",      "municipios": 30},
    "SAN ANDRÉS":         {"score": 2.8, "zona": "BAJO",      "gravedad": "BAJO",      "municipios": 2},
}

DELIT_FACTOR = {
    "HOMICIDIO": 1.4, "VIOLENCIA SEXUAL": 1.3, "AMENAZAS": 1.1,
    "VIOLENCIA INTRAFAMILIAR": 1.0, "LESIONES PERSONALES": 0.9, "HURTO": 0.8
}

# ── Coordenadas de municipios conocidos ───────────────────────────────────────
MUN_COORDS = {
    "MEDELLÍN":     [6.244, -75.574], "BELLO":            [6.337, -75.559],
    "ITAGÜÍ":       [6.185, -75.600], "ENVIGADO":         [6.175, -75.587],
    "APARTADÓ":     [7.880, -76.630], "TURBO":            [8.097, -76.726],
    "BOGOTÁ":       [4.711, -74.072],
    "CALI":         [3.451, -76.532], "BUENAVENTURA":     [3.885, -77.024],
    "PALMIRA":      [3.533, -76.303], "TULUÁ":            [4.085, -76.200],
    "BUGA":         [3.900, -76.300],
    "SOACHA":       [4.579, -74.217], "FACATATIVÁ":       [4.815, -74.355],
    "ZIPAQUIRÁ":    [5.022, -74.005], "FUSAGASUGÁ":       [4.337, -74.364],
    "GIRARDOT":     [4.303, -74.802],
    "BARRANQUILLA": [10.964,-74.796], "SOLEDAD":          [10.918,-74.767],
    "MALAMBO":      [10.855,-74.773], "SABANAGRANDE":     [10.790,-74.756],
    "SABANALARGA":  [10.632,-74.921],
    "BUCARAMANGA":  [7.129, -73.126], "FLORIDABLANCA":    [7.064, -73.089],
    "GIRÓN":        [7.074, -73.168], "PIEDECUESTA":      [6.987, -73.050],
    "BARRANCABERMEJA":[7.064,-73.855],
    "PASTO":        [1.214, -77.280], "TUMACO":           [1.809, -78.808],
    "IPIALES":      [0.828, -77.644], "TÚQUERRES":        [1.088, -77.614],
    "LA UNIÓN":     [1.609, -77.132],
    "MONTERÍA":     [8.757, -75.880], "CERETÉ":           [8.882, -75.792],
    "LORICA":       [9.242, -75.816], "SAHAGÚN":          [8.950, -75.443],
    "TIERRALTA":    [8.172, -76.062],
    "CARTAGENA":    [10.391,-75.479], "MAGANGUÉ":         [9.240, -74.754],
    "EL CARMEN":    [9.718, -75.121], "MOMPÓS":           [9.242, -74.432],
    "TURBACO":      [10.329,-75.415],
    "IBAGUÉ":       [4.438, -75.232], "ESPINAL":          [4.153, -74.886],
    "MELGAR":       [4.210, -74.637], "HONDA":            [5.210, -74.742],
    "CHAPARRAL":    [3.726, -75.490],
    "POPAYÁN":      [2.441, -76.606], "SANTANDER DE QUILICHAO":[3.012,-76.484],
    "VILLAVICENCIO":[4.142, -73.626], "ACACÍAS":          [3.988, -73.759],
    "NEIVA":        [2.935, -75.282], "PITALITO":         [1.855, -76.054],
    "TUNJA":        [5.535, -73.368], "DUITAMA":          [5.827, -73.032],
    "MANIZALES":    [5.070, -75.520], "LA DORADA":        [5.453, -74.666],
    "PEREIRA":      [4.814, -75.696], "DOSQUEBRADAS":     [4.837, -75.671],
    "ARMENIA":      [4.534, -75.681], "CALARCÁ":          [4.524, -75.644],
    "CÚCUTA":       [7.894, -72.508], "OCAÑA":            [8.239, -73.357],
    "VALLEDUPAR":   [10.477,-73.250], "AGUACHICA":        [8.307, -73.619],
    "SANTA MARTA":  [11.240,-74.200], "CIÉNAGA":          [11.005,-74.251],
    "SINCELEJO":    [9.305, -75.398], "COROZAL":          [9.318, -75.292],
    "RIOHACHA":     [11.544,-72.908], "MAICAO":           [11.378,-72.243],
    "FLORENCIA":    [1.616, -75.608],
    "ARAUCA":       [7.090, -70.762], "SARAVENA":         [6.956, -71.862],
    "YOPAL":        [5.338, -72.395], "AGUAZUL":          [5.170, -72.551],
    "MOCOA":        [1.148, -76.648], "PUERTO ASÍS":      [0.506, -76.498],
    "QUIBDÓ":       [5.694, -76.657],
    "SAN JOSÉ DEL GUAVIARE":[2.567,-72.639],
    "PUERTO CARREÑO":[6.189,-67.484],
    "INÍRIDA":      [3.865, -67.924],
    "MITÚ":         [1.253, -70.233],
    "LETICIA":      [-4.215,-69.940], "PUERTO NARIÑO":    [-3.780,-70.376],
    "SAN ANDRÉS":   [12.544,-81.720], "PROVIDENCIA":      [13.350,-81.374],
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_risk_color(score):
    if score >= 4.5: return "#7F1D1D"
    if score >= 4.0: return "#DC2626"
    if score >= 3.5: return "#EF4444"
    if score >= 3.0: return "#F59E0B"
    if score >= 2.5: return "#3B82F6"
    if score >= 2.0: return "#10B981"
    return "#059669"

def risk_badge(level, small=False):
    cfg = RISK_LEVELS.get(level, {"color": "#888", "bg": "#f3f4f6", "label": level})
    p = "2px 9px" if small else "4px 14px"
    fs = "10px" if small else "11px"
    return (f'<span style="background:{cfg["bg"]};color:{cfg["color"]};border:1px solid {cfg["color"]}44;'
            f'border-radius:20px;padding:{p};font-size:{fs};font-weight:700;letter-spacing:0.4px;'
            f'display:inline-block;white-space:nowrap;">{cfg["label"]}</span>')

def card(content, extra=""):
    return (f'<div style="background:#fff;border-radius:20px;border:1px solid #EDE9FE;'
            f'box-shadow:0 2px 20px rgba(109,40,217,0.07);padding:22px;margin-bottom:16px;{extra}">{content}</div>')

def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key and key.strip():
            return key.strip()
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    return None

def call_claude(system_prompt, user_msg, history=None):
    try:
        api_key = get_api_key()
        if not api_key:
            return ("⚠️ API Key no configurada. "
                    "Ve a Streamlit Cloud → Settings → Secrets y agrega: "
                    "GROQ_API_KEY = \"gsk_tu-clave-aqui\"")
        client = Groq(api_key=api_key)
        if history:
            messages = [{"role": "system", "content": system_prompt}] + history
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error de conexión: {str(e)}"

def calc_prediction(dep, mun, delito, sexo, etario, año):
    base = CRIME_DATA.get(dep, {"score": 3.0, "zona": "MEDIO-BAJO", "gravedad": "BAJO", "municipios": 10})
    año_factor = 1.05 if año >= 2024 else (1.0 if año >= 2020 else 0.9)
    adjusted = base["score"] * DELIT_FACTOR.get(delito, 1.0) * año_factor
    zonas = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
    gravedades = ["MÍNIMO","MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO","CRÍTICO"]
    zona_idx = min(max(round(adjusted) - 1, 0), 5)
    grav_idx = min(max(round(adjusted), 0), 7)
    zona = zonas[zona_idx]
    gravedad = gravedades[grav_idx]
    victimas = round(adjusted * 18 + random.random() * 10)
    used_pkl = False

    # Mapeos: valores del UI → valores que esperan los modelos
    DEP_MAP = {
        "BOGOTÁ D.C.": "BOGOTÁ, D. C.",
        "BOYACÁ": "BOYACA",
        "SAN ANDRÉS": "ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA",
    }
    DELITO_MAP = {
        "HOMICIDIO": "HOMICIDIO DOLOSO",
        "VIOLENCIA SEXUAL": "DELITOS SEXUALES",
        "AMENAZAS": "AMENAZAS",
        "VIOLENCIA INTRAFAMILIAR": "VIOLENCIA INTRAFAMILIAR",
        "LESIONES PERSONALES": "LESIONES PERSONALES",
        "HURTO": "LESIONES PERSONALES",  # fallback más cercano
    }
    ETARIO_MAP = {
        "DE 0 A 17 AÑOS": "DE 14 A 17 AÑOS",
        "DE 18 A 26 AÑOS": "DE 18 A 26 AÑOS",
        "DE 27 A 59 AÑOS": "DE 27 A 59 AÑOS",
        "DE 60 Y MÁS": "MAYOR DE 60 AÑOS",
    }

    dep_model = DEP_MAP.get(dep, dep)
    delito_model = DELITO_MAP.get(delito, delito)
    etario_model = ETARIO_MAP.get(etario, etario)

    try:
        if MODELS.get("xgb_zona") and MODELS.get("encoders_zona") and MODELS.get("le_zona"):
            enc = MODELS["encoders_zona"]
            # Columnas en el orden exacto que espera xgb_zona
            zona_feature_cols = list(MODELS["xgb_zona"].feature_names_in_)
            row = pd.DataFrame([{
                "DEPARTAMENTO_HECHO": dep_model,
                "MUNICIPIO_HECHO": mun,
                "AÑO_HECHOS": año,
                "GRUPO_DELITO": delito_model,
                "SEXO": sexo,
                "GRUPO_ETARIO": etario_model,
            }])
            for col in ["DEPARTAMENTO_HECHO", "MUNICIPIO_HECHO", "GRUPO_DELITO", "SEXO", "GRUPO_ETARIO"]:
                if col in enc:
                    try:
                        row[col] = enc[col].transform(row[col].astype(str))
                    except Exception:
                        # Valor desconocido: usar índice 0 como fallback
                        row[col] = 0
            row = row[zona_feature_cols]
            zona_pred = MODELS["xgb_zona"].predict(row)[0]
            try:
                zona = MODELS["le_zona"].inverse_transform([zona_pred])[0]
            except Exception:
                zona = str(zona_pred)
            used_pkl = True
    except Exception:
        pass

    try:
        if MODELS.get("xgb_gravedad") and MODELS.get("le_gravedad") and MODELS.get("preprocessor_grav"):
            grav_feature_cols = list(MODELS["preprocessor_grav"].feature_names_in_)
            row2 = pd.DataFrame([{
                "MUNICIPIO_HECHO": mun,
                "DEPARTAMENTO_HECHO": dep_model,
                "GRUPO_DELITO": delito_model,
                "SEXO": sexo,
                "GRUPO_ETARIO": etario_model,
                "AÑO_HECHOS": año,
            }])[grav_feature_cols]
            row2_t = MODELS["preprocessor_grav"].transform(row2)
            grav_pred = MODELS["xgb_gravedad"].predict(row2_t)[0]
            try:
                gravedad = MODELS["le_gravedad"].inverse_transform([grav_pred])[0]
            except Exception:
                gravedad = str(grav_pred)
            used_pkl = True
    except Exception:
        pass
    probs_zona = {}
    for i, z in enumerate(zonas):
        dist = abs(i - zona_idx)
        probs_zona[z] = max(2, 100 - dist * 28 + (random.random() * 6 - 3))
    total_z = sum(probs_zona.values())
    probs_zona = {k: round(v / total_z * 100, 1) for k, v in probs_zona.items()}
    trend = []
    for y in [2019,2020,2021,2022,2023,2024,2025,2026,2027]:
        yf = 1.05 if y >= 2024 else (1.0 if y >= 2020 else 0.9)
        noise = random.random() * 0.3 - 0.15
        s = base["score"] * DELIT_FACTOR.get(delito, 1.0) * yf * (1 + (y - 2020) * 0.025) + noise
        trend.append({"year": y, "score": round(min(max(s, 0.5), 6.0), 2), "projected": y >= 2025})
    comparativa = []
    for d in DELITOS:
        df = DELIT_FACTOR.get(d, 1.0)
        sc = base["score"] * df * año_factor
        zi = min(max(round(sc) - 1, 0), 5)
        comparativa.append({"label": d, "value": round(sc, 1), "risk": zonas[zi]})
    comparativa.sort(key=lambda x: x["value"], reverse=True)
    months = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    seasonal = [0.85,0.8,0.9,0.95,1.0,1.05,1.1,1.15,1.0,0.95,1.1,1.3]
    monthly = [{"month": m, "value": round(adjusted * seasonal[i] * (1 + random.random()*0.1-0.05), 2),
                "cases": round(victimas/12 * seasonal[i] * (1+random.random()*0.2-0.1))}
               for i, m in enumerate(months)]
    return {"zona": zona, "gravedad": gravedad, "victimas": victimas, "probs_zona": probs_zona,
            "trend": trend, "comparativa": comparativa, "score": round(adjusted, 1),
            "zona_idx": zona_idx, "monthly": monthly, "used_pkl": used_pkl}

# ── Datos de municipios (score sintético determinista) ────────────────────────
@st.cache_data
def build_municipio_data():
    data = {}
    zonas = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
    for dep_name, dep_info in CRIME_DATA.items():
        muns = get_municipios(dep_name)
        base = dep_info["score"]
        mun_list = []
        for mun in muns:
            seed = int(hashlib.md5(f"{dep_name}{mun}".encode()).hexdigest(), 16) % 1000
            variation = (seed / 1000.0 - 0.5) * 1.4
            score = round(min(max(base + variation, 0.8), 5.9), 2)
            zona_idx = min(max(round(score) - 1, 0), 5)
            zona = zonas[zona_idx]
            mun_list.append({"name": mun, "score": score, "zona": zona, "dep": dep_name})
        data[dep_name] = mun_list
    return data

MUNICIPIO_DATA = build_municipio_data()

# ── GeoJSON de Colombia ───────────────────────────────────────────────────────
COLOMBIA_GEO = {"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"DPTO":"AMAZONAS"},"geometry":{"type":"Polygon","coordinates":[[[-73.85,-4.2],[-70.1,-4.2],[-70.1,-2.2],[-69.95,-1.75],[-70.1,-0.15],[-71.0,-0.25],[-72.0,-0.4],[-73.5,-1.5],[-73.85,-2.5],[-73.85,-4.2]]]}},
    {"type":"Feature","properties":{"DPTO":"ANTIOQUIA"},"geometry":{"type":"Polygon","coordinates":[[[-75.9,8.7],[-75.2,8.95],[-74.5,8.9],[-73.8,8.35],[-73.0,7.5],[-73.05,6.9],[-73.5,6.4],[-73.8,6.0],[-74.5,5.8],[-75.2,5.75],[-76.0,5.9],[-76.5,6.3],[-76.9,6.9],[-76.8,7.5],[-76.5,7.9],[-76.2,8.3],[-75.9,8.7]]]}},
    {"type":"Feature","properties":{"DPTO":"ARAUCA"},"geometry":{"type":"Polygon","coordinates":[[[-72.4,7.1],[-70.1,7.1],[-70.1,6.0],[-71.0,5.95],[-72.4,6.0],[-72.4,7.1]]]}},
    {"type":"Feature","properties":{"DPTO":"ATLÁNTICO"},"geometry":{"type":"Polygon","coordinates":[[[-74.85,11.05],[-74.4,11.1],[-74.3,10.85],[-74.5,10.5],[-75.1,10.6],[-75.05,10.9],[-74.85,11.05]]]}},
    {"type":"Feature","properties":{"DPTO":"BOGOTÁ D.C."},"geometry":{"type":"Polygon","coordinates":[[[-74.22,4.83],[-74.0,4.83],[-74.0,4.45],[-74.22,4.45],[-74.22,4.83]]]}},
    {"type":"Feature","properties":{"DPTO":"BOLÍVAR"},"geometry":{"type":"Polygon","coordinates":[[[-75.7,10.5],[-74.85,10.7],[-74.4,10.5],[-74.1,9.8],[-74.0,9.0],[-73.8,8.5],[-74.5,8.2],[-75.0,8.4],[-75.5,8.8],[-75.8,9.3],[-75.7,10.5]]]}},
    {"type":"Feature","properties":{"DPTO":"BOYACÁ"},"geometry":{"type":"Polygon","coordinates":[[[-74.5,6.9],[-73.2,7.1],[-72.4,7.1],[-72.4,6.0],[-72.6,5.7],[-73.2,5.5],[-73.8,5.5],[-74.2,5.7],[-74.5,6.2],[-74.5,6.9]]]}},
    {"type":"Feature","properties":{"DPTO":"CALDAS"},"geometry":{"type":"Polygon","coordinates":[[[-75.7,5.75],[-75.0,5.8],[-74.8,5.4],[-74.9,5.0],[-75.4,4.9],[-75.8,5.1],[-75.9,5.4],[-75.7,5.75]]]}},
    {"type":"Feature","properties":{"DPTO":"CAQUETÁ"},"geometry":{"type":"Polygon","coordinates":[[[-75.6,2.5],[-73.9,2.5],[-73.5,1.5],[-73.5,0.5],[-74.0,-0.5],[-75.2,-0.3],[-75.8,0.5],[-76.2,1.5],[-75.6,2.5]]]}},
    {"type":"Feature","properties":{"DPTO":"CASANARE"},"geometry":{"type":"Polygon","coordinates":[[[-72.4,6.0],[-70.1,6.0],[-70.1,4.8],[-71.2,4.5],[-72.4,5.0],[-72.4,6.0]]]}},
    {"type":"Feature","properties":{"DPTO":"CAUCA"},"geometry":{"type":"Polygon","coordinates":[[[-77.5,3.0],[-76.4,3.1],[-76.0,2.8],[-75.8,2.2],[-75.5,1.5],[-76.2,1.0],[-76.7,1.2],[-77.5,1.5],[-77.9,2.0],[-77.6,2.7],[-77.5,3.0]]]}},
    {"type":"Feature","properties":{"DPTO":"CESAR"},"geometry":{"type":"Polygon","coordinates":[[[-74.4,10.8],[-73.0,10.9],[-72.7,10.5],[-72.6,9.5],[-72.7,8.8],[-73.0,8.5],[-73.8,8.4],[-74.4,8.8],[-74.4,10.8]]]}},
    {"type":"Feature","properties":{"DPTO":"CHOCÓ"},"geometry":{"type":"Polygon","coordinates":[[[-77.3,8.8],[-76.7,8.5],[-76.2,8.3],[-76.5,7.9],[-76.8,7.5],[-76.9,6.9],[-77.2,6.2],[-77.5,5.0],[-77.1,4.5],[-76.9,4.0],[-77.3,3.5],[-77.5,3.0],[-77.6,3.5],[-78.0,4.5],[-77.8,5.5],[-77.5,6.5],[-77.5,7.5],[-77.3,8.8]]]}},
    {"type":"Feature","properties":{"DPTO":"CÓRDOBA"},"geometry":{"type":"Polygon","coordinates":[[[-76.0,8.7],[-75.5,9.0],[-75.2,9.5],[-75.1,10.0],[-74.9,9.8],[-74.7,9.2],[-74.5,8.5],[-75.0,8.4],[-75.5,8.8],[-76.0,8.7]]]}},
    {"type":"Feature","properties":{"DPTO":"CUNDINAMARCA"},"geometry":{"type":"Polygon","coordinates":[[[-74.5,5.5],[-73.8,5.5],[-73.2,5.2],[-73.0,4.5],[-73.2,4.0],[-74.0,3.8],[-74.5,4.0],[-74.8,4.5],[-74.7,5.0],[-74.5,5.5]]]}},
    {"type":"Feature","properties":{"DPTO":"GUAINÍA"},"geometry":{"type":"Polygon","coordinates":[[[-70.1,4.8],[-67.8,4.8],[-67.8,2.0],[-69.0,2.0],[-70.1,2.0],[-70.1,4.8]]]}},
    {"type":"Feature","properties":{"DPTO":"GUAVIARE"},"geometry":{"type":"Polygon","coordinates":[[[-73.5,2.5],[-71.5,2.5],[-71.0,1.5],[-71.5,0.8],[-73.0,0.5],[-73.9,1.0],[-73.5,2.5]]]}},
    {"type":"Feature","properties":{"DPTO":"HUILA"},"geometry":{"type":"Polygon","coordinates":[[[-75.8,3.2],[-75.5,3.5],[-74.8,3.2],[-74.5,2.8],[-74.5,2.0],[-75.0,1.5],[-75.8,2.0],[-76.2,2.5],[-76.0,3.0],[-75.8,3.2]]]}},
    {"type":"Feature","properties":{"DPTO":"LA GUAJIRA"},"geometry":{"type":"Polygon","coordinates":[[[-73.0,12.4],[-71.5,12.4],[-71.0,11.5],[-71.3,11.0],[-72.0,10.8],[-72.7,10.5],[-73.0,11.0],[-73.0,12.4]]]}},
    {"type":"Feature","properties":{"DPTO":"MAGDALENA"},"geometry":{"type":"Polygon","coordinates":[[[-74.4,11.05],[-73.8,11.1],[-73.0,10.9],[-72.7,10.5],[-73.0,10.0],[-73.5,9.5],[-74.0,9.5],[-74.4,10.0],[-74.5,10.8],[-74.4,11.05]]]}},
    {"type":"Feature","properties":{"DPTO":"META"},"geometry":{"type":"Polygon","coordinates":[[[-74.2,5.0],[-72.4,5.0],[-71.2,4.5],[-71.0,3.5],[-71.5,2.5],[-73.5,2.5],[-74.2,3.0],[-74.5,4.0],[-74.2,5.0]]]}},
    {"type":"Feature","properties":{"DPTO":"NARIÑO"},"geometry":{"type":"Polygon","coordinates":[[[-77.5,1.5],[-76.7,1.2],[-76.2,1.0],[-75.5,0.8],[-75.5,0.0],[-75.8,-0.5],[-77.0,-0.5],[-77.8,0.0],[-78.0,0.5],[-77.9,1.2],[-77.5,1.5]]]}},
    {"type":"Feature","properties":{"DPTO":"NORTE DE SANTANDER"},"geometry":{"type":"Polygon","coordinates":[[[-72.4,7.1],[-73.2,7.1],[-73.5,7.5],[-72.5,7.8],[-71.5,7.5],[-71.0,7.0],[-71.5,6.5],[-72.4,6.0],[-72.4,7.1]]]}},
    {"type":"Feature","properties":{"DPTO":"PUTUMAYO"},"geometry":{"type":"Polygon","coordinates":[[[-76.5,1.0],[-75.8,1.2],[-75.5,0.8],[-75.5,0.0],[-75.0,-0.5],[-74.5,-0.5],[-74.0,-1.0],[-75.0,-1.2],[-76.0,-0.5],[-76.5,0.5],[-76.5,1.0]]]}},
    {"type":"Feature","properties":{"DPTO":"QUINDÍO"},"geometry":{"type":"Polygon","coordinates":[[[-75.85,4.75],[-75.4,4.8],[-75.3,4.45],[-75.7,4.4],[-75.9,4.55],[-75.85,4.75]]]}},
    {"type":"Feature","properties":{"DPTO":"RISARALDA"},"geometry":{"type":"Polygon","coordinates":[[[-76.2,5.5],[-75.7,5.55],[-75.7,5.0],[-75.9,4.8],[-76.2,4.9],[-76.3,5.2],[-76.2,5.5]]]}},
    {"type":"Feature","properties":{"DPTO":"SAN ANDRÉS"},"geometry":{"type":"Polygon","coordinates":[[[-81.75,12.62],[-81.65,12.62],[-81.65,12.48],[-81.75,12.48],[-81.75,12.62]]]}},
    {"type":"Feature","properties":{"DPTO":"SANTANDER"},"geometry":{"type":"Polygon","coordinates":[[[-74.5,6.9],[-73.2,7.1],[-72.4,7.1],[-72.4,6.0],[-72.6,5.7],[-73.2,5.5],[-73.8,5.5],[-74.2,5.7],[-74.5,6.2],[-74.5,6.9]]]}},
    {"type":"Feature","properties":{"DPTO":"SUCRE"},"geometry":{"type":"Polygon","coordinates":[[[-75.7,9.8],[-75.2,10.0],[-74.9,9.8],[-74.8,9.2],[-75.0,8.8],[-75.5,8.8],[-75.8,9.3],[-75.7,9.8]]]}},
    {"type":"Feature","properties":{"DPTO":"TOLIMA"},"geometry":{"type":"Polygon","coordinates":[[[-75.5,4.5],[-74.7,5.0],[-74.5,4.5],[-74.5,3.8],[-74.8,3.2],[-75.5,3.2],[-76.0,3.5],[-76.0,4.0],[-75.5,4.5]]]}},
    {"type":"Feature","properties":{"DPTO":"VALLE DEL CAUCA"},"geometry":{"type":"Polygon","coordinates":[[[-77.3,4.2],[-76.4,4.5],[-76.2,4.0],[-76.0,3.5],[-76.5,3.0],[-77.5,3.0],[-77.6,3.5],[-77.5,4.0],[-77.3,4.2]]]}},
    {"type":"Feature","properties":{"DPTO":"VAUPÉS"},"geometry":{"type":"Polygon","coordinates":[[[-70.1,1.8],[-67.8,1.8],[-67.8,-0.1],[-70.1,-0.1],[-70.1,1.8]]]}},
    {"type":"Feature","properties":{"DPTO":"VICHADA"},"geometry":{"type":"Polygon","coordinates":[[[-70.1,6.2],[-67.8,6.2],[-67.8,4.8],[-70.1,4.8],[-70.1,6.2]]]}},
]}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:22px 18px 18px;background:#F5F0FF;border-bottom:1.5px solid #EDE9FE;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:44px;height:44px;background:#7C3AED;
                border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:21px;
                box-shadow:0 4px 14px rgba(124,58,237,0.3);">🛡️</div>
            <div>
                <div style="font-weight:900;font-size:18px;color:#2E1065;letter-spacing:-0.4px;">SafeHer</div>
                <div style="font-size:9px;color:#A78BFA;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-top:2px;">Colombia · IA Protección</div>
            </div>
        </div>
        <div style="display:inline-flex;align-items:center;gap:5px;background:#EDE9FE;border-radius:20px;padding:4px 10px;margin-top:11px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#10B981;"></div>
            <span style="font-size:10px;color:#6D28D9;font-weight:700;">Sistema activo · 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style='padding:12px 10px 6px;'>
        <div style='font-size:9px;color:#C4B5FD;font-weight:800;letter-spacing:2px;
            text-transform:uppercase;padding:0 8px;margin-bottom:6px;'>Navegación</div>
    </div>""", unsafe_allow_html=True)

    nav_options = [
        "🏠  Inicio",
        "📊  Predicción ML",
        "🗺️  Mapa de Riesgo",
        "✈️  Viaje Seguro",
        "🚨  Emergencias",
        "📋  Denuncias",
        "💜  SARA · IA Apoyo",
        "🚔  Ayuda Cercana",
        "i️  Acerca de",
    ]

    default_nav = nav_options.index(st.session_state.get("nav_page", "🏠  Inicio")) if st.session_state.get("nav_page") in nav_options else 0
    if "nav_page" in st.session_state:
        del st.session_state["nav_page"]

    page = st.radio("nav", options=nav_options, index=default_nav, label_visibility="collapsed")

    st.markdown("""
    <div style="padding:10px 12px 16px;border-top:1.5px solid #EDE9FE;margin-top:auto;">
        <div style="background:#FFF5F5;border:1.5px solid #FECACA;
            border-radius:16px;padding:14px;text-align:center;">
            <div style="font-size:9px;color:#DC2626;font-weight:800;letter-spacing:1.5px;
                text-transform:uppercase;margin-bottom:8px;">🚨 Emergencias</div>
            <div style="display:flex;justify-content:center;gap:20px;align-items:stretch;">
                <div style="text-align:center;">
                    <a href="tel:123" style="display:block;font-size:28px;font-weight:900;color:#DC2626;
                        text-decoration:none;font-family:Georgia,serif;line-height:1;letter-spacing:-1px;">123</a>
                    <div style="font-size:9px;color:#EF4444;font-weight:600;margin-top:2px;">Policía Nacional</div>
                </div>
                <div style="width:1px;background:#FECACA;"></div>
                <div style="text-align:center;">
                    <a href="tel:155" style="display:block;font-size:28px;font-weight:900;color:#7C3AED;
                        text-decoration:none;font-family:Georgia,serif;line-height:1;letter-spacing:-1px;">155</a>
                    <div style="font-size:9px;color:#8B5CF6;font-weight:600;margin-top:2px;">Línea Mujer 24/7</div>
                </div>
            </div>
        </div>
        <div style="text-align:center;margin-top:8px;font-size:9px;color:#C4B5FD;font-weight:500;">
            Prototipo académico v4.0 · Datos: Policía Nacional
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ─── PÁGINAS ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── INICIO ────────────────────────────────────────────────────────────────────
if "🏠" in page:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1E1B4B 0%,#312E81 45%,#4C1D95 100%);
        border-radius:28px;padding:52px 52px;margin-bottom:32px;position:relative;overflow:hidden;color:#fff;">
        <div style="position:absolute;top:-80px;right:-80px;width:380px;height:380px;border-radius:50%;background:rgba(255,255,255,0.04);"></div>
        <div style="position:absolute;bottom:-60px;right:100px;width:220px;height:220px;border-radius:50%;background:rgba(236,72,153,0.1);"></div>
        <div style="max-width:600px;position:relative;">
            <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,0.1);
                border:1px solid rgba(255,255,255,0.18);border-radius:24px;padding:6px 18px;
                font-size:11px;color:#E9D5FF;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:24px;">
                ⚡ Sistema Inteligente · Colombia 2025
            </div>
            <h1 style="font-size:46px;font-weight:900;line-height:1.08;margin:0 0 18px;letter-spacing:-1.5px;">
                Tu seguridad es<br>
                <span style="background:linear-gradient(90deg,#F0ABFC,#EC4899,#F59E0B);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">nuestra prioridad</span>
            </h1>
            <p style="color:#C4B5FD;font-size:15px;line-height:1.85;margin:0 0 32px;">
                Plataforma inteligente de predicción, prevención y apoyo para mujeres en Colombia.
                Modelos ML entrenados con datos reales de la Policía Nacional.
            </p>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <a href="?page=pred" style="background:linear-gradient(135deg,#EC4899,#C026D3);color:#fff;
                    padding:14px 26px;border-radius:14px;font-size:14px;font-weight:700;text-decoration:none;
                    box-shadow:0 4px 22px rgba(236,72,153,0.45);">📊 Ver Predicciones IA</a>
                <a href="tel:155" style="background:rgba(255,255,255,0.12);color:#fff;border:1px solid rgba(255,255,255,0.28);
                    padding:14px 26px;border-radius:14px;font-size:14px;font-weight:700;text-decoration:none;">
                    📞 Línea 155 — Mujer</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, color in [
        (c1, "📍", "1.121", "Municipios cubiertos", "#7C3AED"),
        (c2, "🗺️", "33", "Departamentos", "#2563EB"),
        (c3, "🤖", "3 ML", "Modelos de IA", "#059669"),
        (c4, "🛡️", "24/7", "Disponible siempre", "#EC4899"),
    ]:
        with col:
            st.markdown(f"""<div style="background:#fff;border-radius:20px;border:1px solid #EDE9FE;
                padding:22px 18px;text-align:center;box-shadow:0 2px 16px rgba(109,40,217,0.07);">
                <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
                <div style="font-size:28px;font-weight:900;color:{color};line-height:1;font-family:Georgia,serif;">{val}</div>
                <div style="font-size:11px;color:#6B7280;margin-top:6px;font-weight:600;">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:20px;font-weight:800;color:#1E1B4B;margin-bottom:18px;letter-spacing:-0.4px;">🧩 Módulos Disponibles</h2>', unsafe_allow_html=True)

    modules = [
        ("📊","Predicción ML","XGBoost + LightGBM con gráficos avanzados e interpretación IA","#7C3AED"),
        ("🗺️","Mapa de Riesgo","Mapa geográfico interactivo de Colombia por depto. y municipio","#2563EB"),
        ("✈️","Viaje Seguro","Analiza seguridad antes de viajar","#059669"),
        ("🚨","Emergencias","Alertas, botón pánico y líneas directas","#DC2626"),
        ("📋","Denuncias","Registro anónimo con orientación jurídica IA","#D97706"),
        ("💜","IA de Apoyo SARA","Chat empático 24/7 con apoyo psicológico","#7C3AED"),
        ("🚔","Ayuda Cercana","Policía, hospitales, refugios con direcciones","#0891B2"),
        ("i️","Acerca de","Equipo, tecnología y misión del proyecto","#6B7280"),
    ]

    page_map = {
        "Predicción ML": "📊  Predicción ML",
        "Mapa de Riesgo": "🗺️  Mapa de Riesgo",
        "Viaje Seguro": "✈️  Viaje Seguro",
        "Emergencias": "🚨  Emergencias",
        "Denuncias": "📋  Denuncias",
        "IA de Apoyo SARA": "💜  SARA · IA Apoyo",
        "Ayuda Cercana": "🚔  Ayuda Cercana",
        "Acerca de": "i️  Acerca de",
    }

    cols = st.columns(4)
    for i, (icon, title, desc, color) in enumerate(modules):
        with cols[i % 4]:
            st.markdown(f'''<div style="background:#fff;border-radius:20px;padding:22px;border:1px solid #EDE9FE;
                margin-bottom:4px;box-shadow:0 2px 12px rgba(109,40,217,0.06);">
                <div style="width:46px;height:46px;border-radius:14px;background:{color}15;
                    display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:14px;">{icon}</div>
                <div style="font-weight:800;font-size:14px;color:#1E1B4B;margin-bottom:6px;">{title}</div>
                <div style="font-size:12px;color:#6B7280;line-height:1.6;margin-bottom:10px;">{desc}</div>
            </div>''', unsafe_allow_html=True)
            if st.button(f"Ir a {title}", key=f"mod_{i}", use_container_width=True):
                st.session_state["nav_page"] = page_map.get(title, page)
                st.rerun()

    st.markdown("""<div style="background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:1px solid #FCD34D;
        border-radius:16px;padding:16px 22px;display:flex;align-items:center;gap:14px;margin-top:8px;">
        <span style="font-size:22px;">⚠️</span>
        <span style="font-size:13px;color:#92400E;line-height:1.6;">
            <strong>Aviso académico:</strong> Prototipo educativo. Para emergencias reales llama al
            <strong style="color:#DC2626;">123</strong> o la <strong style="color:#7C3AED;">Línea Mujer 155</strong> — gratuita, 24/7.
        </span>
    </div>""", unsafe_allow_html=True)

# ── PREDICCIÓN ML ─────────────────────────────────────────────────────────────
elif "📊" in page:
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#F5F3FF;
            border:1px solid #C4B5FD;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#5B21B6;font-weight:700;margin-bottom:12px;letter-spacing:0.5px;">📊 MÓDULO DE PREDICCIÓN ML</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 6px;letter-spacing:-0.5px;">Predicción de Riesgo</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">XGBoost + LightGBM con gráficos avanzados e interpretación IA para apoyo policial.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:13px;font-weight:700;color:#7C3AED;margin-bottom:16px;">⚙️ Parámetros de Análisis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        dep = st.selectbox("🗺️ Departamento", DEPARTAMENTOS, index=DEPARTAMENTOS.index("ANTIOQUIA"))
    munis = get_municipios(dep)
    with c2:
        mun = st.selectbox("📍 Municipio", munis)
    with c3:
        año = st.selectbox("📅 Año", list(range(2019, 2028)), index=5)
    c4, c5, c6 = st.columns(3)
    with c4:
        delito = st.selectbox("⚖️ Tipo de Delito", DELITOS)
    with c5:
        sexo = st.selectbox("👤 Sexo", ["FEMENINO","MASCULINO"])
    with c6:
        etario = st.selectbox("🎂 Grupo Etario", ["DE 0 A 17 AÑOS","DE 18 A 26 AÑOS","DE 27 A 59 AÑOS","DE 60 Y MÁS"], index=2)

    predict_btn = st.button("🔮 Ejecutar Predicción ML", type="primary", key="predict_btn")

    if predict_btn or st.session_state.get('pred_result'):
        if predict_btn:
            with st.spinner("⏳ Ejecutando modelos ML..."):
                result = calc_prediction(dep, mun, delito, sexo, etario, año)
                st.session_state["pred_result"] = result
                st.session_state["pred_form"] = {"dep": dep, "mun": mun, "delito": delito, "sexo": sexo, "etario": etario, "año": año}
                st.session_state.pop("interp_result", None)

        result = st.session_state.get("pred_result")
        form = st.session_state.get("pred_form", {})

        if result:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1E1B4B,#312E81);border-radius:18px;
                padding:16px 24px;margin-bottom:24px;display:flex;align-items:center;gap:14px;">
                <div style="width:40px;height:40px;background:rgba(255,255,255,0.12);border-radius:12px;
                    display:flex;align-items:center;justify-content:center;font-size:20px;">📍</div>
                <div>
                    <div style="font-weight:800;font-size:15px;color:#fff;">{form.get('dep',dep)} · {form.get('mun',mun)}</div>
                    <div style="font-size:12px;color:#A5B4FC;margin-top:2px;">{form.get('delito',delito)} · {form.get('sexo',sexo)} · {form.get('etario',etario)} · {form.get('año',año)}</div>
                </div>
                <div style="margin-left:auto;display:flex;gap:8px;">
                    {risk_badge(result["zona"])}&nbsp;{risk_badge(result["gravedad"])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            max_month = max(result["monthly"], key=lambda x: x["cases"])
            k1, k2, k3, k4 = st.columns(4)
            for col, label, model, display, extra, color in [
                (k1, "ZONA DE RIESGO", "XGBoost", risk_badge(result["zona"]),
                 f'<div style="font-size:28px;font-weight:900;color:{RISK_LEVELS.get(result["zona"],{}).get("color","#7C3AED")};font-family:Georgia,serif;line-height:1;">{result["score"]}<span style="font-size:14px;">/6.0</span></div>', "#7C3AED"),
                (k2, "NIVEL GRAVEDAD", "LightGBM", risk_badge(result["gravedad"]), "", "#2563EB"),
                (k3, "VÍCTIMAS ESTIMADAS", "Ensemble ML",
                 f'<div style="font-size:40px;font-weight:900;color:#DC2626;font-family:Georgia,serif;line-height:1;">{result["victimas"]}</div>',
                 "personas/año estimadas", "#DC2626"),
                (k4, "MES MÁS CRÍTICO", "Análisis estacional",
                 f'<div style="font-size:24px;font-weight:900;color:#D97706;font-family:Georgia,serif;">{max_month["month"]}</div>',
                 f'{max_month["cases"]} casos estimados', "#D97706"),
            ]:
                with col:
                    st.markdown(f"""<div style="background:#fff;border-radius:20px;border:1px solid #EDE9FE;
                        padding:20px;box-shadow:0 2px 16px rgba(109,40,217,0.07);min-height:130px;">
                        <div style="font-size:9px;font-weight:800;color:{color};letter-spacing:1.2px;
                            text-transform:uppercase;margin-bottom:6px;">{label}</div>
                        <div style="font-size:9px;color:#A78BFA;font-weight:600;margin-bottom:10px;">{model}</div>
                        <div style="margin-bottom:6px;">{display}</div>
                        <div style="font-size:10px;color:#6B7280;margin-top:4px;">{extra}</div>
                    </div>""", unsafe_allow_html=True)

            pkl_badge = ('✅ Modelos PKL reales activos' if result.get("used_pkl") else '⚙️ Modo simulación (PKL no cargados)')
            pkl_color = "#059669" if result.get("used_pkl") else "#D97706"
            pkl_bg = "#ECFDF5" if result.get("used_pkl") else "#FFFBEB"
            st.markdown(f'''<div style="background:{pkl_bg};border:1px solid {pkl_color}40;border-radius:12px;
                padding:10px 16px;margin:12px 0;display:inline-block;font-size:12px;font-weight:700;color:{pkl_color};">
                {pkl_badge}</div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Tendencia Histórica", "🎯 Distribución de Probabilidad", "📅 Variación Mensual", "🕸️ Radar de Riesgo"])

            with tab1:
                solid_x = [t["year"] for t in result["trend"] if not t["projected"]]
                solid_y = [t["score"] for t in result["trend"] if not t["projected"]]
                proj_x_start = solid_x[-1]
                proj_y_start = solid_y[-1]
                proj_x = [proj_x_start] + [t["year"] for t in result["trend"] if t["projected"]]
                proj_y = [proj_y_start] + [t["score"] for t in result["trend"] if t["projected"]]
                colors_pts = [get_risk_color(t["score"]) for t in result["trend"]]
                fig = go.Figure()
                fig.add_shape(type="rect", x0=2019, x1=2027, y0=4, y1=6, fillcolor="#FEE2E2", opacity=0.25, line_width=0)
                fig.add_shape(type="rect", x0=2019, x1=2027, y0=3, y1=4, fillcolor="#FEF9C3", opacity=0.25, line_width=0)
                fig.add_shape(type="rect", x0=2019, x1=2027, y0=0, y1=3, fillcolor="#F0FDF4", opacity=0.25, line_width=0)
                for y_pos, label_t, color_t in [(5.0,"⚠️ Alto riesgo","#DC2626"),(3.5,"⚡ Riesgo medio","#F59E0B"),(1.5,"✅ Controlado","#059669")]:
                    fig.add_annotation(x=2019.1, y=y_pos, text=label_t, showarrow=False, font=dict(size=9, color=color_t), xanchor="left")
                fig.add_trace(go.Scatter(x=solid_x, y=solid_y, mode="lines+markers", line=dict(color="#7C3AED", width=3),
                    marker=dict(size=9, color=colors_pts[:len(solid_x)], line=dict(color="white", width=2)),
                    name="Histórico", fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"))
                fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines+markers",
                    line=dict(color="#A78BFA", width=2.5, dash="dot"),
                    marker=dict(size=8, color="#A78BFA", line=dict(color="white", width=2)), name="Proyectado 2025–2027"))
                fig.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 6.5], gridcolor="#EDE9FE", tickfont=dict(size=10, color="#6B7280"), title="Score (0–6)", title_font=dict(size=10, color="#6B7280")),
                    xaxis=dict(gridcolor="#EDE9FE", tickfont=dict(size=10, color="#6B7280"), dtick=1),
                    legend=dict(orientation="h", y=1.04, font=dict(size=10)), font=dict(family="Plus Jakarta Sans"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with tab2:
                prob_sorted = sorted(result["probs_zona"].items(), key=lambda x: x[1], reverse=True)
                labels = [p[0] for p in prob_sorted]
                values = [p[1] for p in prob_sorted]
                bar_colors = [RISK_LEVELS.get(z, {"color": "#888"})["color"] for z in labels]
                fig2 = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                    marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
                    text=[f"{v}%" for v in values], textposition="outside", textfont=dict(size=11, color="#1E1B4B")))
                fig2.update_layout(height=280, margin=dict(l=80, r=60, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0, max(values)*1.2], gridcolor="#EDE9FE", tickfont=dict(size=10), ticksuffix="%"),
                    yaxis=dict(tickfont=dict(size=11, color="#1E1B4B")),
                    font=dict(family="Plus Jakarta Sans"), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

            with tab3:
                max_m = max(result["monthly"], key=lambda x: x["value"])
                bar_colors_m = ["#DC2626" if m["month"] == max_m["month"] else get_risk_color(m["value"]) for m in result["monthly"]]
                fig3 = go.Figure(go.Bar(x=[m["month"] for m in result["monthly"]], y=[m["cases"] for m in result["monthly"]],
                    marker=dict(color=bar_colors_m, opacity=0.88, line=dict(width=0)),
                    text=[str(m["cases"]) for m in result["monthly"]], textposition="outside", textfont=dict(size=10)))
                fig3.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=30),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="#EDE9FE", tickfont=dict(size=9, color="#6B7280")),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10, color="#6B7280")),
                    font=dict(family="Plus Jakarta Sans"), showlegend=False)
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

            with tab4:
                radar_cats = ["Frecuencia", "Gravedad", "Víctimas", "Estacionalidad", "Proyección", "Impacto Social"]
                score_norm = result["score"] / 6.0
                radar_vals = [
                    min(score_norm * 1.1, 1.0),
                    score_norm * 0.95,
                    min(result["victimas"] / 200, 1.0),
                    max(result["monthly"], key=lambda x: x["value"])["value"] / 6.0,
                    result["trend"][-1]["score"] / 6.0,
                    score_norm * 1.05,
                ]
                radar_vals = [round(min(v, 1.0), 2) for v in radar_vals]
                radar_vals_pct = [round(v * 100) for v in radar_vals]
                fig4 = go.Figure()
                fig4.add_trace(go.Scatterpolar(r=radar_vals_pct, theta=radar_cats, fill='toself',
                    fillcolor='rgba(124,58,237,0.15)', line=dict(color='#7C3AED', width=2.5),
                    marker=dict(size=7, color='#7C3AED'), name=f'{form.get("dep",dep)}'))
                fig4.add_trace(go.Scatterpolar(r=[50]*len(radar_cats), theta=radar_cats,
                    line=dict(color='#E2E8F0', width=1, dash='dot'), showlegend=False, mode='lines'))
                fig4.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9), gridcolor='#EDE9FE'),
                               angularaxis=dict(tickfont=dict(size=11, color='#1E1B4B')), bgcolor='rgba(0,0,0,0)'),
                    height=300, margin=dict(l=50,r=50,t=30,b=30), paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True, legend=dict(font=dict(size=10)), font=dict(family='Plus Jakarta Sans'))
                st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:14px;font-weight:800;color:#1E1B4B;margin-bottom:4px;">📊 Comparativa por Tipo de Delito — {form.get("dep",dep)}</div>', unsafe_allow_html=True)
            max_v = result["comparativa"][0]["value"] if result["comparativa"] else 1
            for item in result["comparativa"]:
                cfg = RISK_LEVELS.get(item["risk"], {"color": "#888"})
                is_sel = item["label"] == form.get("delito", delito)
                bg = cfg["color"] + "12" if is_sel else "#FAFAFA"
                border = cfg["color"] + "50" if is_sel else "#EDE9FE"
                marker = " ← seleccionado" if is_sel else ""
                st.markdown(f"""
                <div style="padding:12px 16px;background:{bg};border-radius:14px;
                    border:1.5px solid {border};margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
                        <span style="font-size:12px;color:#1E1B4B;font-weight:{'800' if is_sel else '600'};">{item['label']}<span style="font-size:10px;color:{cfg['color']};font-style:italic;">{marker}</span></span>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:13px;font-weight:900;color:{cfg['color']};">{item['value']}</span>
                            {risk_badge(item['risk'], small=True)}
                        </div>
                    </div>
                    <div style="background:#E8E4F9;border-radius:6px;height:8px;overflow:hidden;">
                        <div style="width:{item['value']/max_v*100:.0f}%;height:100%;background:linear-gradient(90deg,{cfg['color']}88,{cfg['color']});border-radius:6px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="sh-card">
                <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
                    <div style="width:46px;height:46px;background:linear-gradient(135deg,#1E1B4B,#5B21B6);
                        border-radius:14px;display:flex;align-items:center;justify-content:center;
                        font-size:22px;box-shadow:0 4px 14px rgba(91,33,182,0.3);">🤖</div>
                    <div>
                        <div style="font-size:16px;font-weight:800;color:#1E1B4B;">Interpretación IA para Fuerzas Policiales</div>
                        <div style="font-size:11px;color:#A78BFA;">Diagnóstico situacional · Factores de riesgo · Acciones operativas</div>
                    </div>
                    <div style="margin-left:auto;background:linear-gradient(135deg,#F5F3FF,#EDE9FE);
                        color:#5B21B6;font-size:11px;font-weight:700;padding:6px 14px;border-radius:20px;
                        border:1px solid #C4B5FD;">🔒 Uso Policial</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "interp_result" not in st.session_state or predict_btn:
                with st.spinner("🤖 Analizando con IA especializada..."):
                    interp = call_claude(
                        """Eres analista experto en seguridad pública de Colombia, asesor de la Policía Nacional.
Estructura tu respuesta con EXACTAMENTE estas 4 secciones (cada una máximo 3 puntos con bullet •):

🔍 DIAGNÓSTICO SITUACIONAL
⚠️ FACTORES DE RIESGO IDENTIFICADOS
🚨 ACCIONES INMEDIATAS RECOMENDADAS
🤝 RECURSOS INTERINSTITUCIONALES

Total máximo 280 palabras. Sé concreto, técnico y orientado a la acción policial.""",
                        f"""Analiza:
- Departamento: {form.get('dep',dep)} | Municipio: {form.get('mun',mun)}
- Delito: {form.get('delito',delito)} | Año: {form.get('año',año)}
- Zona de Riesgo: {result['zona']} | Gravedad: {result['gravedad']}
- Víctimas estimadas: {result['victimas']} | Score: {result['score']}/6.0"""
                    )
                    st.session_state["interp_result"] = interp

            interp = st.session_state.get("interp_result", "")
            if interp:
                st.markdown(f"""<div style="background:linear-gradient(135deg,#FAFAFA,#F5F3FF);border-radius:16px;
                    padding:20px 24px;border:1px solid #EDE9FE;font-size:13px;line-height:1.85;
                    white-space:pre-wrap;color:#1E1B4B;">{interp}</div>""", unsafe_allow_html=True)

            s = result["score"]
            if s >= 4.0:
                al_bg, al_border, al_color, al_icon, al_text = "#FEF2F2","#FECDD3","#991B1B","🚨","Zona de ALTO RIESGO. Se requiere refuerzo urgente de patrullaje y coordinación inmediata con la Fiscalía."
            elif s >= 3.0:
                al_bg, al_border, al_color, al_icon, al_text = "#FFFBEB","#FDE68A","#92400E","⚠️","Riesgo MODERADO. Monitoreo activo y campañas preventivas focalizadas."
            else:
                al_bg, al_border, al_color, al_icon, al_text = "#ECFDF5","#A7F3D0","#065F46","✅","Riesgo CONTROLADO. Mantener estrategias preventivas actuales."
            st.markdown(f"""<div style="background:{al_bg};border:1.5px solid {al_border};border-radius:16px;
                padding:16px 22px;display:flex;align-items:flex-start;gap:12px;margin-top:16px;">
                <span style="font-size:20px;">{al_icon}</span>
                <div>
                    <div style="font-weight:800;color:{al_color};font-size:13px;margin-bottom:4px;">Alerta Operacional</div>
                    <div style="font-size:12px;color:{al_color};opacity:0.85;line-height:1.6;">{al_text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ── MAPA DE RIESGO ─────────────────────────────────────────────────────────────
elif "🗺️" in page:
    import folium
    import streamlit.components.v1 as components

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#EFF6FF;
            border:1px solid #BFDBFE;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#1D4ED8;font-weight:700;margin-bottom:12px;">🗺️ MAPA INTERACTIVO</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 6px;letter-spacing:-0.5px;">Mapa de Riesgo — Colombia</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">Visualización geográfica del nivel de riesgo por departamento y municipio.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    for col, label, count, color, bg in [
        (c1, "Crítico / Muy Alto", sum(1 for d in CRIME_DATA.values() if d["score"] >= 4.0), "#DC2626", "linear-gradient(135deg,#FEF2F2,#FEE2E2)"),
        (c2, "Riesgo Alto",        sum(1 for d in CRIME_DATA.values() if 3.5 <= d["score"] < 4.0), "#EF4444", "linear-gradient(135deg,#FFF7ED,#FFEDD5)"),
        (c3, "Riesgo Medio",       sum(1 for d in CRIME_DATA.values() if 3.0 <= d["score"] < 3.5), "#F59E0B", "linear-gradient(135deg,#FFFBEB,#FEF3C7)"),
        (c4, "Controlado",         sum(1 for d in CRIME_DATA.values() if d["score"] < 3.0), "#059669", "linear-gradient(135deg,#ECFDF5,#D1FAE5)"),
    ]:
        with col:
            st.markdown(f"""<div style="background:{bg};border-radius:18px;padding:16px 18px;
                border:1px solid {color}25;text-align:center;margin-bottom:14px;">
                <div style="font-size:30px;font-weight:900;color:{color};font-family:Georgia,serif;line-height:1;">{count}</div>
                <div style="font-size:10px;color:{color};font-weight:700;margin-top:5px;">Departamentos</div>
                <div style="font-size:10px;color:#6B7280;margin-top:3px;">{label}</div>
            </div>""", unsafe_allow_html=True)

    # Controles
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        filter_zone = st.selectbox("🔍 Filtrar por nivel de riesgo:", ["TODOS","ALTO","MEDIO-ALTO","MEDIO-BAJO","BAJO"], key="map_filter")
    with ctrl2:
        map_view = st.radio("🗺️ Vista:", ["Por Departamento", "Por Municipio"], horizontal=True, key="map_view_mode")
    with ctrl3:
        if map_view == "Por Municipio":
            default_dep_idx = DEPARTAMENTOS.index(st.session_state.get("selected_dep", "ANTIOQUIA")) \
                if st.session_state.get("selected_dep") in DEPARTAMENTOS else 1
            dep_muni_sel = st.selectbox("📍 Departamento:", DEPARTAMENTOS, index=default_dep_idx, key="dep_muni_sel")
        else:
            dep_muni_sel = None

    def get_risk_zone_label(score):
        if score >= 4.0: return "ALTO"
        if score >= 3.5: return "MEDIO-ALTO"
        if score >= 2.5: return "MEDIO-BAJO"
        return "BAJO"

    sorted_deps = sorted(
        [{"name": k, **v} for k, v in CRIME_DATA.items()
         if filter_zone == "TODOS" or get_risk_zone_label(v["score"]) == filter_zone],
        key=lambda x: x["score"], reverse=True
    )

    st.markdown(f'<div style="font-size:12px;color:#6B7280;margin-bottom:16px;">Mostrando <strong style="color:#1E1B4B;">{len(sorted_deps)}</strong> departamentos</div>', unsafe_allow_html=True)

    col_map, col_detail = st.columns([2, 1])

    # ── Función leyenda ────────────────────────────────────────────────────────
    def _add_legend(m):
        legend_html = """
        <div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;
            border-radius:12px;padding:12px 16px;box-shadow:0 2px 16px rgba(0,0,0,0.15);
            font-family:'Segoe UI',sans-serif;border:1px solid #eee;">
            <div style="font-weight:800;font-size:12px;color:#1E1B4B;margin-bottom:8px;">🛡️ Nivel de Riesgo</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><div style="width:14px;height:14px;border-radius:3px;background:#7F1D1D;"></div><span style="font-size:11px;color:#374151;">≥4.5 Crítico</span></div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><div style="width:14px;height:14px;border-radius:3px;background:#DC2626;"></div><span style="font-size:11px;color:#374151;">≥4.0 Alto</span></div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><div style="width:14px;height:14px;border-radius:3px;background:#EF4444;"></div><span style="font-size:11px;color:#374151;">≥3.5 Medio-Alto</span></div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><div style="width:14px;height:14px;border-radius:3px;background:#F59E0B;"></div><span style="font-size:11px;color:#374151;">≥3.0 Medio</span></div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><div style="width:14px;height:14px;border-radius:3px;background:#3B82F6;"></div><span style="font-size:11px;color:#374151;">≥2.5 Bajo</span></div>
            <div style="display:flex;align-items:center;gap:8px;"><div style="width:14px;height:14px;border-radius:3px;background:#10B981;"></div><span style="font-size:11px;color:#374151;">&lt;2.5 Mínimo</span></div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    # ── Mapa por departamentos ─────────────────────────────────────────────────
    @st.cache_data
    def build_folium_map_dep(filter_z, sel_dep):
        m = folium.Map(location=[4.5, -74.0], zoom_start=5, tiles="CartoDB positron", scrollWheelZoom=True)
        for feature in COLOMBIA_GEO["features"]:
            dep_name = feature["properties"]["DPTO"]
            dep_data = CRIME_DATA.get(dep_name)
            if not dep_data:
                continue
            if filter_z != "TODOS" and get_risk_zone_label(dep_data["score"]) != filter_z:
                continue
            score = dep_data["score"]
            color = get_risk_color(score)
            is_sel = dep_name == sel_dep
            pct = int(score / 6 * 100)
            zona = dep_data["zona"]
            gravedad = dep_data["gravedad"]
            muns = dep_data["municipios"]
            tooltip_html = f"""
            <div style="font-family:'Segoe UI',sans-serif;min-width:200px;padding:2px;">
                <div style="font-weight:800;font-size:14px;color:#1E1B4B;border-bottom:2px solid {color};padding-bottom:4px;margin-bottom:8px;">{dep_name}</div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="font-size:22px;font-weight:900;color:{color};">{score:.1f}</span>
                    <span style="font-size:11px;color:#6B7280;">/ 6.0</span>
                    <span style="background:{color}22;color:{color};border-radius:20px;padding:2px 10px;font-size:10px;font-weight:700;">{zona}</span>
                </div>
                <div style="background:#eee;border-radius:4px;height:6px;margin-bottom:8px;">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
                </div>
                <div style="font-size:11px;color:#374151;"><b>Gravedad:</b> {gravedad}</div>
                <div style="font-size:11px;color:#374151;"><b>Municipios:</b> {muns}</div>
            </div>"""
            popup_html = f"""
            <div style="font-family:'Segoe UI',sans-serif;width:220px;">
                <div style="background:{color};color:white;padding:10px 14px;border-radius:8px 8px 0 0;font-weight:800;font-size:14px;">{dep_name}</div>
                <div style="padding:12px 14px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;">
                    <div style="font-size:28px;font-weight:900;color:{color};margin-bottom:4px;">{score:.1f} <span style="font-size:13px;color:#9CA3AF;">/ 6.0</span></div>
                    <div style="background:#F3F4F6;border-radius:4px;height:8px;margin-bottom:10px;">
                        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
                    </div>
                    <div style="font-size:12px;margin-bottom:4px;"><b style="color:#374151;">Zona:</b> <span style="color:{color};font-weight:700;">{zona}</span></div>
                    <div style="font-size:12px;margin-bottom:4px;"><b style="color:#374151;">Gravedad:</b> {gravedad}</div>
                    <div style="font-size:12px;"><b style="color:#374151;">Municipios:</b> {muns}</div>
                </div>
            </div>"""
            weight = 3 if is_sel else 1.5
            fill_op = 0.88 if is_sel else 0.72
            stroke_color = "#1E1B4B" if is_sel else "#ffffff"
            folium.GeoJson(feature,
                style_function=lambda x, c=color, w=weight, fo=fill_op, sc=stroke_color: {
                    "fillColor": c, "color": sc, "weight": w, "fillOpacity": fo},
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=240),
                highlight_function=lambda x, c=color: {"fillColor": c, "fillOpacity": 0.95, "weight": 3, "color": "#1E1B4B"},
            ).add_to(m)
            coords = feature["geometry"]["coordinates"][0]
            lons = [p[0] for p in coords]; lats = [p[1] for p in coords]
            cx = sum(lons)/len(lons); cy = sum(lats)/len(lats)
            short_name = dep_name.split()[0][:8] if len(dep_name) > 12 else dep_name[:10]
            folium.Marker(location=[cy, cx],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:8px;font-weight:800;color:white;text-shadow:0 1px 3px rgba(0,0,0,0.7);white-space:nowrap;text-align:center;line-height:1.2;"><div>{short_name}</div><div style="font-size:9px;">{score:.1f}</div></div>',
                    icon_size=(70, 28), icon_anchor=(35, 14)),
            ).add_to(m)
        _add_legend(m)
        return m._repr_html_()

    # ── Mapa por municipios ────────────────────────────────────────────────────
    @st.cache_data
    def build_folium_map_mun(dep_name_sel, mun_data_json):
        mun_data = json.loads(mun_data_json)
        dep_info = CRIME_DATA.get(dep_name_sel, {})
        dep_feature = next((f for f in COLOMBIA_GEO["features"] if f["properties"]["DPTO"] == dep_name_sel), None)
        if dep_feature:
            coords = dep_feature["geometry"]["coordinates"][0]
            center_lat = sum(p[1] for p in coords) / len(coords)
            center_lon = sum(p[0] for p in coords) / len(coords)
        else:
            center_lat, center_lon = 4.5, -74.0

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron", scrollWheelZoom=True)

        if dep_feature:
            folium.GeoJson(dep_feature,
                style_function=lambda x, c=get_risk_color(dep_info.get("score", 3.0)): {
                    "fillColor": c, "color": "#1E1B4B", "weight": 2, "fillOpacity": 0.08},
            ).add_to(m)

        for mun in mun_data:
            mun_name = mun["name"]
            score = mun["score"]
            zona = mun["zona"]
            color = get_risk_color(score)
            pct = int(score / 6 * 100)

            if mun_name in MUN_COORDS:
                lat, lon = MUN_COORDS[mun_name]
            else:
                seed = int(hashlib.md5(f"{dep_name_sel}{mun_name}".encode()).hexdigest(), 16)
                lat_off = ((seed % 1000) / 1000.0 - 0.5) * 1.5
                lon_off = (((seed // 1000) % 1000) / 1000.0 - 0.5) * 1.5
                lat = center_lat + lat_off
                lon = center_lon + lon_off

            tooltip_html = f"""
            <div style="font-family:'Segoe UI',sans-serif;min-width:180px;padding:2px;">
                <div style="font-weight:800;font-size:13px;color:#1E1B4B;border-bottom:2px solid {color};padding-bottom:4px;margin-bottom:8px;">
                    📍 {mun_name}
                </div>
                <div style="font-size:10px;color:#6B7280;margin-bottom:4px;">{dep_name_sel}</div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="font-size:20px;font-weight:900;color:{color};">{score:.1f}</span>
                    <span style="font-size:10px;color:#6B7280;">/ 6.0</span>
                    <span style="background:{color}22;color:{color};border-radius:20px;padding:2px 8px;font-size:9px;font-weight:700;">{zona}</span>
                </div>
                <div style="background:#eee;border-radius:4px;height:5px;">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
                </div>
            </div>"""

            popup_html = f"""
            <div style="font-family:'Segoe UI',sans-serif;width:200px;">
                <div style="background:{color};color:white;padding:8px 12px;border-radius:8px 8px 0 0;font-weight:800;font-size:13px;">
                    📍 {mun_name}
                </div>
                <div style="padding:10px 12px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;">
                    <div style="font-size:10px;color:#6B7280;margin-bottom:6px;">{dep_name_sel}</div>
                    <div style="font-size:26px;font-weight:900;color:{color};margin-bottom:4px;">{score:.1f}
                        <span style="font-size:12px;color:#9CA3AF;">/ 6.0</span>
                    </div>
                    <div style="background:#F3F4F6;border-radius:4px;height:7px;margin-bottom:8px;">
                        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
                    </div>
                    <div style="font-size:11px;font-weight:700;color:{color};">Zona: {zona}</div>
                </div>
            </div>"""

            radius = 8 + score * 3
            folium.CircleMarker(
                location=[lat, lon], radius=radius,
                color="#1E1B4B", weight=1.5,
                fill=True, fill_color=color, fill_opacity=0.85,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(m)

            folium.Marker(location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:7px;font-weight:800;color:#1E1B4B;'
                         f'text-shadow:0 0 3px white,0 0 3px white;white-space:nowrap;'
                         f'text-align:center;margin-top:{int(radius)+6}px;">'
                         f'{mun_name[:12]}</div>',
                    icon_size=(90, 20), icon_anchor=(45, 0)),
            ).add_to(m)

        _add_legend(m)
        return m._repr_html_()

    with col_map:
        view_label = f"Vista Municipal — {dep_muni_sel}" if map_view == "Por Municipio" else "Colombia — Por Departamento"
        st.markdown(f'<div style="font-size:12px;font-weight:700;color:#A78BFA;margin-bottom:16px;text-transform:uppercase;letter-spacing:1.2px;">🇨🇴 {view_label}</div>', unsafe_allow_html=True)

        if map_view == "Por Municipio":
            mun_data_list = MUNICIPIO_DATA.get(dep_muni_sel, [])
            mun_data_json = json.dumps(mun_data_list)
            map_html = build_folium_map_mun(dep_muni_sel, mun_data_json)
            components.html(map_html, height=540, scrolling=False)
            st.markdown(f'<div style="font-size:11px;color:#6B7280;text-align:center;margin-top:4px;">🖱️ Zoom · Clic en círculo para detalles · Tamaño proporcional al score · Depto: <strong>{dep_muni_sel}</strong></div>', unsafe_allow_html=True)
        else:
            sel_dep_map = st.session_state.get("selected_dep", "")
            map_html = build_folium_map_dep(filter_zone, sel_dep_map)
            components.html(map_html, height=540, scrolling=False)
            st.markdown('<div style="font-size:11px;color:#6B7280;text-align:center;margin-top:4px;">🖱️ Zoom con scroll · Clic en departamento para detalles · Pasa cursor para info rápida</div>', unsafe_allow_html=True)

        # Leyenda
        st.markdown("""
        <div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin:10px 0 4px;padding:8px 12px;
            background:#F8FAFF;border-radius:12px;border:1px solid #EDE9FE;">
            <span style="font-size:11px;color:#7F1D1D;font-weight:700;">● ≥4.5 Crítico</span>
            <span style="font-size:11px;color:#DC2626;font-weight:700;">● ≥4.0 Alto</span>
            <span style="font-size:11px;color:#EF4444;font-weight:700;">● ≥3.5 Medio-Alto</span>
            <span style="font-size:11px;color:#F59E0B;font-weight:700;">● ≥3.0 Medio</span>
            <span style="font-size:11px;color:#3B82F6;font-weight:700;">● ≥2.5 Bajo</span>
            <span style="font-size:11px;color:#10B981;font-weight:700;">● &lt;2.5 Mínimo</span>
        </div>
        """, unsafe_allow_html=True)

        # Ranking / botones
        if map_view == "Por Municipio":
            mun_data_list = MUNICIPIO_DATA.get(dep_muni_sel, [])
            mun_sorted = sorted(mun_data_list, key=lambda x: x["score"], reverse=True)
            st.markdown(f'<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin:18px 0 12px;">📊 Ranking de Municipios — {dep_muni_sel}</div>', unsafe_allow_html=True)
            max_mun_score = mun_sorted[0]["score"] if mun_sorted else 1
            for i, mun in enumerate(mun_sorted):
                color = get_risk_color(mun["score"])
                pct = mun["score"] / max_mun_score * 100
                num_color = "#DC2626" if i < 3 else "#6B7280"
                st.markdown(f"""<div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="font-size:10px;color:{num_color};font-weight:800;width:20px;">#{i+1}</span>
                        <span style="font-size:12px;color:#1E1B4B;font-weight:600;flex:1;padding:0 8px;">{mun['name']}</span>
                        <span style="font-size:12px;color:{color};font-weight:900;">{mun['score']:.1f}/6.0</span>
                        &nbsp;{risk_badge(mun['zona'], small=True)}
                    </div>
                    <div style="background:#EDE9FE;border-radius:8px;height:9px;overflow:hidden;">
                        <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{color}80,{color});border-radius:8px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin:18px 0 12px;">📊 Seleccionar Departamento</div>', unsafe_allow_html=True)
            cols_per_row = 5
            rows = [sorted_deps[i:i+cols_per_row] for i in range(0, len(sorted_deps), cols_per_row)]
            for row in rows:
                rcols = st.columns(cols_per_row)
                for j, dep_d in enumerate(row):
                    color = get_risk_color(dep_d["score"])
                    with rcols[j]:
                        if st.button(f"{dep_d['name'][:10]}\n{dep_d['score']:.1f}", key=f"dep_{dep_d['name']}", use_container_width=True):
                            st.session_state["selected_dep"] = dep_d["name"]
                        st.markdown(f"""<div style="background:{color}14;border-radius:8px;padding:2px 4px;
                            text-align:center;margin-top:-10px;margin-bottom:4px;">
                            <div style="font-size:8px;color:{color};font-weight:700;">{dep_d['zona']}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin:18px 0 12px;">🏆 Top 12 Departamentos por Score de Riesgo</div>', unsafe_allow_html=True)
            for i, d in enumerate(sorted_deps[:12]):
                color = get_risk_color(d["score"])
                pct = d["score"] / 6 * 100
                num_color = "#DC2626" if i < 3 else "#6B7280"
                st.markdown(f"""<div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="font-size:10px;color:{num_color};font-weight:800;width:20px;">#{i+1}</span>
                        <span style="font-size:12px;color:#1E1B4B;font-weight:600;flex:1;padding:0 8px;">{d['name']}</span>
                        <span style="font-size:12px;color:{color};font-weight:900;">{d['score']:.1f}/6.0</span>
                        &nbsp;{risk_badge(d['zona'], small=True)}
                    </div>
                    <div style="background:#EDE9FE;border-radius:8px;height:9px;overflow:hidden;">
                        <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{color}80,{color});border-radius:8px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_detail:
        if map_view == "Por Municipio":
            mun_data_list = MUNICIPIO_DATA.get(dep_muni_sel, [])
            mun_sorted_d = sorted(mun_data_list, key=lambda x: x["score"], reverse=True)
            dep_info = CRIME_DATA.get(dep_muni_sel, {})
            dep_color = get_risk_color(dep_info.get("score", 3.0))

            st.markdown(f"""<div class="sh-card">
                <div style="margin-bottom:14px;">
                    <div style="font-size:19px;font-weight:900;color:#1E1B4B;">{dep_muni_sel}</div>
                    <div style="font-size:11px;color:#A78BFA;margin-top:2px;">Vista Municipal · {len(mun_data_list)} municipios analizados</div>
                </div>
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;
                    background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;padding:14px;">
                    <div style="text-align:center;">
                        <div style="font-size:36px;font-weight:900;color:{dep_color};font-family:Georgia,serif;line-height:1;">{dep_info.get('score',3.0):.1f}</div>
                        <div style="font-size:10px;color:#A78BFA;font-weight:600;">Depto / 6.0</div>
                    </div>
                    <div>
                        <div style="font-size:12px;color:#6B7280;margin-bottom:6px;">Nivel departamento:</div>
                        <div style="margin-bottom:6px;">{risk_badge(dep_info.get('zona','MEDIO-BAJO'))}</div>
                        <div style="font-size:11px;color:#6B7280;margin-top:6px;">Municipio más crítico:</div>
                        <div style="font-size:13px;font-weight:800;color:{get_risk_color(mun_sorted_d[0]['score'])};">{mun_sorted_d[0]['name']} ({mun_sorted_d[0]['score']:.1f})</div>
                    </div>
                </div>
                <div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:10px;">🔴 Top 3 Municipios más críticos</div>
            """, unsafe_allow_html=True)

            for mun in mun_sorted_d[:3]:
                mcolor = get_risk_color(mun["score"])
                st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 10px;background:{mcolor}0d;border-radius:10px;margin-bottom:6px;
                    border:1px solid {mcolor}25;">
                    <span style="font-size:12px;color:#1E1B4B;font-weight:700;">📍 {mun['name']}</span>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <span style="font-size:12px;font-weight:900;color:{mcolor};">{mun['score']:.1f}</span>
                        {risk_badge(mun['zona'], small=True)}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin:14px 0 8px;">✅ Top 3 Municipios más seguros</div>', unsafe_allow_html=True)
            for mun in mun_sorted_d[-3:][::-1]:
                mcolor = get_risk_color(mun["score"])
                st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 10px;background:{mcolor}0d;border-radius:10px;margin-bottom:6px;
                    border:1px solid {mcolor}25;">
                    <span style="font-size:12px;color:#1E1B4B;font-weight:700;">✅ {mun['name']}</span>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <span style="font-size:12px;font-weight:900;color:{mcolor};">{mun['score']:.1f}</span>
                        {risk_badge(mun['zona'], small=True)}
                    </div>
                </div>""", unsafe_allow_html=True)

            if st.button(f"🤖 Análisis IA municipios de {dep_muni_sel}", key="ai_mun_btn", use_container_width=True, type="primary"):
                with st.spinner("Analizando con IA..."):
                    top_muns = ", ".join([f"{m['name']} ({m['score']:.1f})" for m in mun_sorted_d[:3]])
                    safe_muns = ", ".join([f"{m['name']} ({m['score']:.1f})" for m in mun_sorted_d[-3:][::-1]])
                    ai_mun = call_claude(
                        "Eres experto en seguridad pública colombiana. Análisis breve (máx 130 palabras) con bullets y emojis sobre distribución de riesgo entre municipios de un departamento. Menciona factores que explican la diferencia de riesgo entre municipios.",
                        f"Analiza distribución de riesgo en {dep_muni_sel}. Score depto: {dep_info.get('score',3.0):.1f}/6.0. "
                        f"Municipios más críticos: {top_muns}. Más seguros: {safe_muns}. Total: {len(mun_data_list)} municipios."
                    )
                    st.session_state[f"ai_mun_{dep_muni_sel}"] = ai_mun

            ai_mun_r = st.session_state.get(f"ai_mun_{dep_muni_sel}", "")
            if ai_mun_r:
                st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;padding:14px;
                    border:1px solid #C4B5FD;font-size:12px;color:#1E1B4B;line-height:1.75;white-space:pre-wrap;">{ai_mun_r}</div>""",
                    unsafe_allow_html=True)

        else:
            sel_name = st.session_state.get("selected_dep")
            if sel_name and sel_name in CRIME_DATA:
                sel = {"name": sel_name, **CRIME_DATA[sel_name]}
                color = get_risk_color(sel["score"])
                st.markdown(f"""<div class="sh-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                        <div>
                            <div style="font-size:19px;font-weight:900;color:#1E1B4B;">{sel["name"]}</div>
                            <div style="font-size:11px;color:#A78BFA;margin-top:2px;">Colombia · {sel["municipios"]} municipios</div>
                        </div>
                        <div style="width:42px;height:42px;background:{color}15;border-radius:12px;
                            display:flex;align-items:center;justify-content:center;font-size:22px;">🗺️</div>
                    </div>
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;
                        background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;padding:16px;">
                        <div style="text-align:center;">
                            <div style="font-size:40px;font-weight:900;color:{color};font-family:Georgia,serif;line-height:1;">{sel['score']:.1f}</div>
                            <div style="font-size:10px;color:#A78BFA;font-weight:600;">/ 6.0</div>
                        </div>
                        <div>
                            <div style="font-size:12px;color:#6B7280;margin-bottom:8px;">Nivel de riesgo:</div>
                            <div style="margin-bottom:6px;">{risk_badge(sel['zona'])}</div>
                            <div>{risk_badge(sel['gravedad'])}</div>
                        </div>
                    </div>
                    <div style="background:#EDE9FE;border-radius:8px;height:10px;margin-bottom:18px;">
                        <div style="width:{sel['score']/6*100:.0f}%;height:100%;background:linear-gradient(90deg,{color}88,{color});border-radius:8px;"></div>
                    </div>
                    <div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:10px;">⚖️ Riesgo por tipo de delito</div>
                """, unsafe_allow_html=True)

                zonas_list = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
                for d in DELITOS:
                    fac = DELIT_FACTOR.get(d, 1.0)
                    sc = sel["score"] * fac
                    z = zonas_list[min(max(round(sc) - 1, 0), 5)]
                    cfg = RISK_LEVELS.get(z, {"color": "#888"})
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;border-bottom:1px solid #EDE9FE;">
                        <span style="font-size:11px;color:#1E1B4B;">{d}</span>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <span style="font-size:11px;font-weight:700;color:{cfg['color']};">{sc:.1f}</span>
                            {risk_badge(z, small=True)}
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin:14px 0 8px;">🕐 Riesgo por Horario</div>', unsafe_allow_html=True)
                h_risks = [("Madrugada (0–6h)","BAJO"),("Mañana (6–12h)","MUY BAJO"),
                           ("Tarde (12–18h)","MEDIO-BAJO"),("Noche (18–24h)","MUY ALTO" if sel["score"]>=4.0 else "ALTO" if sel["score"]>=3.0 else "MEDIO-ALTO")]
                for h_label, h_risk in h_risks:
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid #EDE9FE;">
                        <span style="font-size:11px;color:#1E1B4B;">{h_label}</span>
                        {risk_badge(h_risk, small=True)}
                    </div>""", unsafe_allow_html=True)

                if st.button(f"🤖 Análisis IA completo de {sel_name}", key="ai_map_btn", use_container_width=True, type="primary"):
                    with st.spinner("Analizando con IA..."):
                        ai_text = call_claude(
                            "Eres experto en seguridad pública colombiana. Análisis breve (máx 130 palabras) con bullets y emojis: contexto del departamento, amenazas principales para mujeres, horarios de mayor riesgo, recomendación operativa clave.",
                            f"Analiza seguridad para mujeres en {sel_name}, Colombia. Score: {sel['score']}/6.0, zona: {sel['zona']}, gravedad: {sel['gravedad']}."
                        )
                        st.session_state[f"ai_map_{sel_name}"] = ai_text

                ai_result = st.session_state.get(f"ai_map_{sel_name}", "")
                if ai_result:
                    st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;padding:16px;
                        border:1px solid #C4B5FD;font-size:12px;color:#1E1B4B;line-height:1.75;white-space:pre-wrap;">{ai_result}</div>""",
                        unsafe_allow_html=True)

                if sel["score"] >= 4.0:
                    rec_color, rec_bg, rec_text = "#991B1B","#FEF2F2","🚨 Zona de alto riesgo. Refuerzo urgente de patrullaje y coordinación con Fiscalía."
                elif sel["score"] >= 3.0:
                    rec_color, rec_bg, rec_text = "#92400E","#FFFBEB","⚠️ Riesgo moderado. Monitoreo activo y campañas de prevención."
                else:
                    rec_color, rec_bg, rec_text = "#065F46","#ECFDF5","✅ Riesgo controlado. Mantener estrategias preventivas."
                st.markdown(f"""<div style="background:{rec_bg};border-radius:14px;padding:14px 16px;font-size:12px;
                    line-height:1.6;margin-top:10px;border:1px solid {rec_color}22;">
                    <span style="color:{rec_color};font-weight:700;">{rec_text}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:20px;
                    border:2px dashed #C4B5FD;padding:48px 24px;text-align:center;">
                    <div style="font-size:48px;margin-bottom:14px;">🗺️</div>
                    <div style="font-size:15px;font-weight:700;color:#5B21B6;margin-bottom:6px;">Selecciona un departamento</div>
                    <div style="font-size:12px;color:#A78BFA;line-height:1.7;">Haz clic en cualquier departamento de la lista para ver el análisis detallado de riesgo y obtener interpretación con IA.</div>
                </div>""", unsafe_allow_html=True)

# ── VIAJE SEGURO ──────────────────────────────────────────────────────────────
elif "✈️" in page:
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#ECFDF5;
            border:1px solid #A7F3D0;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#059669;font-weight:700;margin-bottom:12px;">✈️ PLANIFICACIÓN DE VIAJE SEGURO</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 6px;letter-spacing:-0.5px;">Viaje Seguro</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">Selecciona departamento y municipio para ver el análisis de seguridad personalizado antes de viajar.</p>
    </div>
    """, unsafe_allow_html=True)

    col_dep, col_mun, col_btn = st.columns([2, 2, 1])
    with col_dep:
        dep_viaje = st.selectbox("🗺️ Departamento de destino:", DEPARTAMENTOS, key="dep_viaje")
    with col_mun:
        munis_viaje = get_municipios(dep_viaje)
        mun_viaje = st.selectbox("📍 Municipio (opcional):", ["— Todos —"] + munis_viaje, key="mun_viaje")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analizar_btn = st.button("🔍 Analizar", type="primary", use_container_width=True, key="viaje_btn")

    if analizar_btn or st.session_state.get("viaje_result"):
        if analizar_btn:
            data = CRIME_DATA.get(dep_viaje, {"score": 2.8, "zona": "BAJO", "gravedad": "BAJO", "municipios": 10})
            muns = get_municipios(dep_viaje)
            mun_sel = mun_viaje if mun_viaje != "— Todos —" else None
            mun_score = None
            if mun_sel and dep_viaje in MUNICIPIO_DATA:
                mun_info = next((m for m in MUNICIPIO_DATA[dep_viaje] if m["name"] == mun_sel), None)
                if mun_info:
                    mun_score = mun_info["score"]
            st.session_state["viaje_result"] = {
                "dep": dep_viaje, "data": data, "muns": muns,
                "mun_sel": mun_sel, "mun_score": mun_score
            }
            st.session_state.pop("viaje_tips", None)

        vr = st.session_state.get("viaje_result")
        if vr:
            score_base = vr["data"]["score"]
            score = vr.get("mun_score") or score_base
            mun_label = f' · {vr["mun_sel"]}' if vr.get("mun_sel") else ""

            if score <= 2.0:
                safety = {"label": "Muy Seguro", "color": "#059669", "bg": "linear-gradient(135deg,#ECFDF5,#D1FAE5)", "icon": "🟢", "stars": 5, "emoji": "😊"}
            elif score <= 3.0:
                safety = {"label": "Precaución Moderada", "color": "#F59E0B", "bg": "linear-gradient(135deg,#FFFBEB,#FEF3C7)", "icon": "🟡", "stars": 3, "emoji": "⚠️"}
            elif score <= 4.0:
                safety = {"label": "Riesgo Medio-Alto", "color": "#EF4444", "bg": "linear-gradient(135deg,#FEF2F2,#FEE2E2)", "icon": "🟠", "stars": 2, "emoji": "🚨"}
            else:
                safety = {"label": "Alto Riesgo — Precaución Máxima", "color": "#DC2626", "bg": "linear-gradient(135deg,#FEF2F2,#FECDD3)", "icon": "🔴", "stars": 1, "emoji": "🛑"}

            stars_html = "".join([
                f'<span style="font-size:22px;color:{"#F59E0B" if i < safety["stars"] else "#E2E8F0"};">★</span>'
                for i in range(5)
            ])

            st.markdown(f"""<div style="background:{safety['bg']};border:2px solid {safety['color']}30;
                border-radius:24px;padding:24px 32px;margin-bottom:24px;display:flex;align-items:center;gap:24px;
                box-shadow:0 4px 20px {safety['color']}18;">
                <div style="font-size:52px;">{safety['icon']}</div>
                <div style="flex:1;">
                    <div style="font-size:22px;font-weight:900;color:#1E1B4B;margin-bottom:2px;">{vr['dep']}{mun_label}</div>
                    <div style="font-size:15px;font-weight:700;color:{safety['color']};margin-bottom:10px;">{safety['label']} {safety['emoji']}</div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div>{stars_html}</div>
                        <span style="font-size:11px;color:#6B7280;font-weight:600;">índice de seguridad para mujeres</span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:50px;font-weight:900;color:{safety['color']};font-family:Georgia,serif;line-height:1;">{score:.1f}</div>
                    <div style="font-size:11px;color:#6B7280;font-weight:600;">Score / 6.0</div>
                    {f'<div style="font-size:10px;color:#A78BFA;margin-top:4px;">Depto: {score_base:.1f}</div>' if vr.get("mun_score") else ""}
                </div>
            </div>""", unsafe_allow_html=True)

            zonas_list = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
            delito_scores = sorted(
                [{"d": d, "sc": round(score * DELIT_FACTOR.get(d, 1.0), 1)} for d in DELITOS],
                key=lambda x: x["sc"], reverse=True
            )
            delito_mas_alto = delito_scores[0]["d"] if delito_scores else "N/A"
            hora_seg = "6am – 9pm" if score <= 3.0 else "8am – 7pm"
            transporte_seg = "Taxi/App, bus urbano" if score <= 3.0 else "Solo taxi/app verificado"

            k1, k2, k3, k4 = st.columns(4)
            for col, icon, label, val, color in [
                (k1, "⚠️", "Delito más frecuente", delito_mas_alto.title(), safety["color"]),
                (k2, "🕐", "Horario seguro", hora_seg, "#059669"),
                (k3, "🚗", "Transporte recomendado", transporte_seg, "#2563EB"),
                (k4, "📍", "Municipios analizados", str(vr["data"]["municipios"]), "#7C3AED"),
            ]:
                with col:
                    st.markdown(f"""<div style="background:#fff;border-radius:16px;border:1px solid #EDE9FE;
                        padding:14px 16px;box-shadow:0 2px 12px rgba(109,40,217,0.06);margin-bottom:16px;">
                        <div style="font-size:20px;margin-bottom:6px;">{icon}</div>
                        <div style="font-size:10px;color:#A78BFA;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">{label}</div>
                        <div style="font-size:13px;font-weight:800;color:{color};line-height:1.3;">{val}</div>
                    </div>""", unsafe_allow_html=True)

            tab_muns, tab_graficas, tab_consejos = st.tabs([
                "🏙️ Municipios del Departamento",
                "📊 Análisis Visual de Riesgo",
                "💡 Consejos de Seguridad"
            ])

            with tab_muns:
                st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin:14px 0 16px;">🏙️ Selecciona un municipio para ver su nivel de riesgo individual</div>', unsafe_allow_html=True)
                mun_data_dep = MUNICIPIO_DATA.get(vr["dep"], [])
                mun_sorted_v = sorted(mun_data_dep, key=lambda x: x["score"], reverse=True)
                max_mun_s = mun_sorted_v[0]["score"] if mun_sorted_v else 1

                cols_mun = st.columns(3)
                for i, m in enumerate(mun_sorted_v):
                    mcolor = get_risk_color(m["score"])
                    is_sel = vr.get("mun_sel") == m["name"]
                    bg_m = f"{mcolor}12" if is_sel else "#FAFAFA"
                    border_m = f"2px solid {mcolor}" if is_sel else "1px solid #EDE9FE"
                    with cols_mun[i % 3]:
                        st.markdown(f"""<div style="background:{bg_m};border:{border_m};border-radius:14px;
                            padding:12px 14px;margin-bottom:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                <span style="font-size:12px;font-weight:{"800" if is_sel else "600"};color:#1E1B4B;">{m['name']}</span>
                                {risk_badge(m['zona'], small=True)}
                            </div>
                            <div style="display:flex;align-items:center;gap:8px;">
                                <div style="flex:1;background:#EDE9FE;border-radius:6px;height:7px;overflow:hidden;">
                                    <div style="width:{m['score']/max_mun_s*100:.0f}%;height:100%;background:{mcolor};border-radius:6px;"></div>
                                </div>
                                <span style="font-size:11px;font-weight:900;color:{mcolor};flex-shrink:0;">{m['score']:.1f}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"{'✓ Seleccionado' if is_sel else 'Analizar'}", key=f"mun_v_{i}_{m['name'][:8]}", use_container_width=True,
                                     type="primary" if is_sel else "secondary"):
                            data2 = CRIME_DATA.get(vr["dep"], vr["data"])
                            muns2 = get_municipios(vr["dep"])
                            st.session_state["viaje_result"] = {
                                "dep": vr["dep"], "data": data2, "muns": muns2,
                                "mun_sel": m["name"] if not is_sel else None,
                                "mun_score": m["score"] if not is_sel else None
                            }
                            st.session_state.pop("viaje_tips", None)
                            st.rerun()

            with tab_graficas:
                import plotly.graph_objects as go

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin-bottom:12px;">⚖️ Score de Riesgo por Tipo de Delito</div>', unsafe_allow_html=True)
                    labels_d = [item["d"].title() for item in delito_scores]
                    values_d = [item["sc"] for item in delito_scores]
                    zonas_d = [zonas_list[min(max(round(v)-1, 0), 5)] for v in values_d]
                    bar_colors_d = [RISK_LEVELS.get(z, {"color": "#888"})["color"] for z in zonas_d]
                    fig_bar = go.Figure(go.Bar(
                        x=values_d, y=labels_d, orientation="h",
                        marker=dict(color=bar_colors_d, opacity=0.88, line=dict(width=0)),
                        text=[f"{v:.1f}" for v in values_d],
                        textposition="outside", textfont=dict(size=11, color="#1E1B4B")
                    ))
                    fig_bar.update_layout(
                        height=240, margin=dict(l=10, r=50, t=10, b=20),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(range=[0, max(values_d)*1.2], gridcolor="#EDE9FE",
                                   tickfont=dict(size=9, color="#6B7280"), title="Score"),
                        yaxis=dict(tickfont=dict(size=10, color="#1E1B4B")),
                        font=dict(family="Plus Jakarta Sans"), showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

                with col_g2:
                    st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin-bottom:12px;">🎯 Indicador General de Riesgo</div>', unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=score,
                        delta={"reference": 3.0, "increasing": {"color": "#DC2626"}, "decreasing": {"color": "#059669"}},
                        number={"suffix": "/6.0", "font": {"size": 28, "color": safety["color"]}},
                        gauge={
                            "axis": {"range": [0, 6], "tickwidth": 1, "tickcolor": "#6B7280",
                                     "tickfont": {"size": 9}},
                            "bar": {"color": safety["color"], "thickness": 0.28},
                            "bgcolor": "white",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 2], "color": "#D1FAE5"},
                                {"range": [2, 3], "color": "#FEF3C7"},
                                {"range": [3, 4], "color": "#FFEDD5"},
                                {"range": [4, 5], "color": "#FEE2E2"},
                                {"range": [5, 6], "color": "#FECDD3"},
                            ],
                            "threshold": {"line": {"color": "#1E1B4B", "width": 3}, "thickness": 0.75, "value": score}
                        },
                        title={"text": f"{vr['dep']}{mun_label}<br><span style='font-size:11px;color:#6B7280;'>{safety['label']}</span>",
                               "font": {"size": 13, "color": "#1E1B4B"}}
                    ))
                    fig_gauge.update_layout(
                        height=240, margin=dict(l=20, r=20, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans")
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

                col_g3, col_g4 = st.columns(2)

                with col_g3:
                    st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin-bottom:12px;">🕸️ Radar de Dimensiones de Riesgo</div>', unsafe_allow_html=True)
                    radar_cats_v = ["Violencia física", "Riesgo nocturno", "Transporte", "Zonas remotas", "Riesgo digital", "Acoso"]
                    s_n = score / 6.0
                    radar_vals_v = [
                        round(min(s_n * 1.05, 1.0) * 100),
                        round(min(s_n * 1.25, 1.0) * 100),
                        round(min(s_n * 0.75, 1.0) * 100),
                        round(min(s_n * 1.15, 1.0) * 100),
                        round(min(s_n * 0.65, 1.0) * 100),
                        round(min(s_n * 0.90, 1.0) * 100),
                    ]
                    # Convertir color hex a rgb para fillcolor
                    h = safety["color"].lstrip("#")
                    r_val, g_val, b_val = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_vals_v, theta=radar_cats_v, fill='toself',
                        fillcolor=f"rgba({r_val},{g_val},{b_val},0.18)",
                        line=dict(color=safety["color"], width=2.5),
                        marker=dict(size=7, color=safety["color"]),
                        name=vr["dep"]
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[50]*6, theta=radar_cats_v,
                        line=dict(color="#E2E8F0", width=1, dash="dot"),
                        showlegend=False, mode="lines"
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8), gridcolor="#EDE9FE"),
                            angularaxis=dict(tickfont=dict(size=10, color="#1E1B4B")),
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        height=240, margin=dict(l=40, r=40, t=20, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        font=dict(family="Plus Jakarta Sans")
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

                with col_g4:
                    st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin-bottom:12px;">🍩 Distribución de Municipios por Nivel</div>', unsafe_allow_html=True)
                    mun_dep_list = MUNICIPIO_DATA.get(vr["dep"], [])
                    zona_counts = {}
                    for m in mun_dep_list:
                        z = m["zona"]
                        zona_counts[z] = zona_counts.get(z, 0) + 1
                    dona_labels = list(zona_counts.keys())
                    dona_values = list(zona_counts.values())
                    dona_colors = [RISK_LEVELS.get(z, {"color": "#888"})["color"] for z in dona_labels]
                    fig_dona = go.Figure(go.Pie(
                        labels=dona_labels, values=dona_values,
                        hole=0.55,
                        marker=dict(colors=dona_colors, line=dict(color="white", width=2)),
                        textinfo="label+percent",
                        textfont=dict(size=9),
                        showlegend=False
                    ))
                    fig_dona.update_layout(
                        height=240, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans"),
                        annotations=[dict(
                            text=f"{len(mun_dep_list)}<br>munic.",
                            x=0.5, y=0.5, font_size=14, showarrow=False,
                            font=dict(color="#1E1B4B", family="Georgia, serif")
                        )]
                    )
                    st.plotly_chart(fig_dona, use_container_width=True, config={"displayModeBar": False})

            with tab_consejos:
                import json as _json

                st.markdown(f'<div style="font-size:15px;font-weight:800;color:#1E1B4B;margin:14px 0 18px;">🤖 Consejos Personalizados — {vr["dep"]}{mun_label}</div>', unsafe_allow_html=True)

                if "viaje_tips" not in st.session_state:
                    with st.spinner("✨ Preparando consejos personalizados con IA..."):
                        tips_raw = call_claude(
                            """Eres experta en seguridad para mujeres viajeras en Colombia. Responde en español.
Devuelve EXACTAMENTE este JSON (sin markdown, sin texto extra, sin comillas adicionales):
{
  "seguridad": ["consejo1","consejo2","consejo3"],
  "alojamiento": ["consejo1","consejo2","consejo3"],
  "horarios": ["consejo1","consejo2"],
  "transporte": ["consejo1","consejo2","consejo3"],
  "emergencias": ["Policia Nacional: 123","Linea Mujer: 155","consejo local"],
  "cultura": ["consejo1","consejo2"],
  "tecnologia": ["consejo1","consejo2"],
  "salud": ["consejo1","consejo2"]
}
Los consejos deben ser concretos, practicos y especificos para el departamento indicado.""",
                            f"Departamento: {vr['dep']}, Colombia. Score de riesgo: {score:.1f}/6.0 (zona: {vr['data']['zona']}). Municipio: {vr.get('mun_sel', 'todos')}."
                        )
                        try:
                            clean = tips_raw.strip().replace("```json", "").replace("```", "").strip()
                            tips_dict = _json.loads(clean)
                        except Exception:
                            tips_dict = None
                        st.session_state["viaje_tips"] = tips_raw
                        st.session_state["viaje_tips_dict"] = tips_dict

                tips_dict = st.session_state.get("viaje_tips_dict")

                SECCIONES_CONSEJOS = [
                    ("seguridad",   "🛡️", "Seguridad Personal",         "#DC2626", "#FEF2F2", "#FECDD3"),
                    ("alojamiento", "🏠", "Mejores Zonas para Alojarse", "#7C3AED", "#F5F3FF", "#DDD6FE"),
                    ("horarios",    "🕐", "Horarios Seguros",            "#059669", "#ECFDF5", "#A7F3D0"),
                    ("transporte",  "🚗", "Transporte Recomendado",      "#2563EB", "#EFF6FF", "#BFDBFE"),
                    ("tecnologia",  "📱", "Tecnología y Conectividad",   "#0891B2", "#ECFEFF", "#A5F3FC"),
                    ("cultura",     "🌺", "Cultura y Costumbres Locales","#D97706", "#FFFBEB", "#FDE68A"),
                    ("salud",       "💊", "Salud y Prevención",          "#16A34A", "#F0FDF4", "#BBF7D0"),
                    ("emergencias", "📞", "Emergencias Locales",         "#991B1B", "#FFF1F2", "#FECDD3"),
                ]

                if tips_dict:
                    cols_tips = st.columns(2)
                    for idx, (key, icon, titulo, color, bg, border_c) in enumerate(SECCIONES_CONSEJOS):
                        items = tips_dict.get(key, [])
                        if not items:
                            continue
                        with cols_tips[idx % 2]:
                            bullets = "".join([
                                f'<div style="display:flex;gap:10px;margin-bottom:10px;align-items:flex-start;">'
                                f'<span style="color:{color};font-size:14px;flex-shrink:0;margin-top:1px;">•</span>'
                                f'<span style="font-size:12px;color:#374151;line-height:1.65;">{item}</span>'
                                f'</div>'
                                for item in items
                            ])
                            st.markdown(f"""<div style="background:{bg};border:1.5px solid {border_c};
                                border-radius:18px;padding:16px 18px;margin-bottom:14px;
                                border-left:4px solid {color};">
                                <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                                    <div style="width:36px;height:36px;background:{color}18;border-radius:10px;
                                        display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">{icon}</div>
                                    <span style="font-size:13px;font-weight:800;color:#1E1B4B;">{titulo}</span>
                                </div>
                                {bullets}
                            </div>""", unsafe_allow_html=True)
                else:
                    tips_text = st.session_state.get("viaje_tips", "")
                    if tips_text:
                        st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;
                            padding:20px 24px;border:1px solid #C4B5FD;font-size:13px;line-height:1.85;
                            white-space:pre-wrap;color:#1E1B4B;">{tips_text}</div>""", unsafe_allow_html=True)

                if score >= 4.0:
                    av_bg, av_border, av_color, av_icon, av_text = "#FEF2F2","#FECDD3","#991B1B","🚨","Zona de ALTO RIESGO. Se recomienda evitar viajes no esenciales y consultar a autoridades antes de viajar."
                elif score >= 3.0:
                    av_bg, av_border, av_color, av_icon, av_text = "#FFFBEB","#FDE68A","#92400E","⚠️","Riesgo MODERADO. Viaja informada, comparte tu itinerario con alguien de confianza y guarda los números de emergencia."
                else:
                    av_bg, av_border, av_color, av_icon, av_text = "#ECFDF5","#A7F3D0","#065F46","✅","Destino RELATIVAMENTE SEGURO. Mantén precauciones básicas para disfrutar tu viaje tranquila."

                st.markdown(f"""<div style="background:{av_bg};border:1.5px solid {av_border};border-radius:16px;
                    padding:16px 22px;display:flex;align-items:flex-start;gap:12px;margin-top:8px;">
                    <span style="font-size:22px;">{av_icon}</span>
                    <div>
                        <div style="font-weight:800;color:{av_color};font-size:13px;margin-bottom:4px;">Recomendación General</div>
                        <div style="font-size:12px;color:{av_color};opacity:0.9;line-height:1.6;">{av_text}</div>
                    </div>
                    <div style="margin-left:auto;display:flex;gap:6px;flex-shrink:0;">
                        <a href="tel:155" style="background:{av_color};color:#fff;padding:8px 14px;border-radius:12px;font-size:11px;font-weight:700;text-decoration:none;white-space:nowrap;">📞 155 Línea Mujer</a>
                        <a href="tel:123" style="background:#DC2626;color:#fff;padding:8px 14px;border-radius:12px;font-size:11px;font-weight:700;text-decoration:none;white-space:nowrap;">🚨 123 Policía</a>
                    </div>
                </div>""", unsafe_allow_html=True)
# ── EMERGENCIAS ────────────────────────────────────────────────────────────────
elif "🚨" in page:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#7F1D1D,#BE123C,#DC2626);border-radius:24px;
        padding:26px 32px;margin-bottom:30px;color:#fff;box-shadow:0 8px 32px rgba(220,38,38,0.3);">
        <h1 style="font-size:30px;font-weight:900;margin:0 0 8px;letter-spacing:-0.3px;">🚨 Centro de Emergencias</h1>
        <p style="color:#FECDD3;font-size:14px;margin:0;line-height:1.6;">
            Si estás en peligro, presiona el botón que describe tu situación. Todas las llamadas son gratuitas y confidenciales.
        </p>
    </div>
    """, unsafe_allow_html=True)

    emergencias = [
        ("🆘","Estoy en peligro","123 — Emergencias","tel:123","#DC2626"),
        ("👣","Me están siguiendo","123 — Policía","tel:123","#D97706"),
        ("🔇","No puedo hablar","SMS 123","sms:123","#7C3AED"),
        ("🏃","Estoy secuestrada","123 — Urgente","tel:123","#991B1B"),
        ("🚔","Necesito Policía","Policía Nacional","tel:123","#1D4ED8"),
        ("🚑","Necesito Ambulancia","Cruz Roja — 132","tel:132","#059669"),
        ("💜","Apoyo psicológico","Línea 137","tel:137","#8B5CF6"),
        ("👩","Línea Mujer","155 — 24/7 Gratis","tel:155","#EC4899"),
    ]

    cols = st.columns(4)
    for i, (icon, label, sub, href, color) in enumerate(emergencias):
        with cols[i % 4]:
            st.markdown(f"""<a href="{href}" style="text-decoration:none;">
            <div style="background:#fff;border:2px solid {color}22;border-radius:22px;padding:24px 18px;
                text-align:center;cursor:pointer;min-height:136px;margin-bottom:16px;box-shadow:0 3px 16px rgba(0,0,0,0.06);"
                onmouseover="this.style.borderColor='{color}';this.style.transform='translateY(-2px)';"
                onmouseout="this.style.borderColor='{color}22';this.style.transform='translateY(0)';">
                <div style="font-size:32px;margin-bottom:10px;">{icon}</div>
                <div style="font-weight:800;font-size:13px;color:#1E1B4B;margin-bottom:6px;">{label}</div>
                <div style="font-size:11px;font-weight:700;color:{color};background:{color}12;
                    padding:4px 10px;border-radius:20px;display:inline-block;">{sub}</div>
            </div></a>""", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:18px;font-weight:800;color:#1E1B4B;margin:20px 0 14px;'>📞 Líneas de Emergencia</h2>", unsafe_allow_html=True)
    lineas = [
        ("tel:123","123","Policía","🚔","#1D4ED8"),("tel:155","155","Línea Mujer","💜","#7C3AED"),
        ("tel:125","125","Defensa Civil","🟢","#059669"),("tel:132","132","Cruz Roja","❤️","#DC2626"),
        ("tel:137","137","Salud Mental","🧠","#8B5CF6"),("tel:106","106","Bomberos","🔥","#D97706"),
    ]
    cols_l = st.columns(6)
    for i, (href, num, desc, ic, co) in enumerate(lineas):
        with cols_l[i]:
            st.markdown(f"""<a href="{href}" style="text-decoration:none;">
            <div style="background:#fff;border-radius:18px;border:1px solid #EDE9FE;padding:18px 12px;
                text-align:center;margin-bottom:12px;box-shadow:0 2px 12px rgba(109,40,217,0.07);">
                <div style="font-size:22px;margin-bottom:6px;">{ic}</div>
                <div style="font-family:Georgia,serif;font-size:28px;font-weight:900;color:{co};line-height:1;">{num}</div>
                <div style="font-size:10px;color:#6B7280;margin-top:6px;font-weight:600;">{desc}</div>
            </div></a>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div style="background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:1.5px solid #FCD34D;
            border-radius:20px;padding:24px;">
            <div style="font-weight:800;color:#92400E;font-size:15px;margin-bottom:12px;">🔒 Salida Rápida</div>
            <p style="font-size:13px;color:#78350F;line-height:1.7;margin-bottom:16px;">
                Presiona para ir a una página neutra si alguien está mirando tu pantalla:
            </p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <a href="https://www.google.com" target="_blank" style="background:#FEF3C7;color:#92400E;padding:10px 18px;border-radius:12px;font-size:12px;text-decoration:none;font-weight:700;border:1px solid #FCD34D;">🔍 Google</a>
                <a href="https://weather.com" target="_blank" style="background:#FEF3C7;color:#92400E;padding:10px 18px;border-radius:12px;font-size:12px;text-decoration:none;font-weight:700;border:1px solid #FCD34D;">🌦️ Clima</a>
                <a href="https://www.eltiempo.com" target="_blank" style="background:#FEF3C7;color:#92400E;padding:10px 18px;border-radius:12px;font-size:12px;text-decoration:none;font-weight:700;border:1px solid #FCD34D;">📰 Noticias</a>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        tips_list = ["🔵 Mantén la calma y ve a un lugar concurrido",
                     "🔵 Llama o envía tu ubicación a alguien de confianza",
                     "🔵 Memoriza: 123 Policía · 155 Mujer · 132 Ambulancia",
                     "🔵 No confrontes al agresor directamente",
                     "🔵 Documenta evidencia si es completamente seguro",
                     "🔵 Activa la alerta de tu celular o smartwatch"]
        tips_html = "".join([f'<div style="font-size:12px;color:#1E1B4B;margin-bottom:9px;line-height:1.65;">{t}</div>' for t in tips_list])
        st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border:1.5px solid #C4B5FD;
            border-radius:20px;padding:24px;">
            <div style="font-weight:800;color:#5B21B6;font-size:15px;margin-bottom:14px;">💡 En caso de emergencia</div>
            {tips_html}
        </div>""", unsafe_allow_html=True)

# ── DENUNCIAS ─────────────────────────────────────────────────────────────────
elif "📋" in page:
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#FFFBEB;
            border:1px solid #FDE68A;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#D97706;font-weight:700;margin-bottom:12px;">📋 CENTRO DE DENUNCIAS SEGURO</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 8px;letter-spacing:-0.5px;">Registro de Denuncia</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">Registra un hecho de forma segura y confidencial. Orientación jurídica personalizada con IA.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("denuncia_sent"):
        step = st.session_state.get("denuncia_step", 1)
        steps = ["Clasificación", "Descripción", "Opciones y Envío"]
        step_html = '<div style="display:flex;align-items:center;gap:0;margin-bottom:24px;">'
        for si, sl in enumerate(steps, 1):
            if si < step:
                s_bg, s_color, s_text = "#059669","#fff","✓"
            elif si == step:
                s_bg, s_color, s_text = "#5B21B6","#fff",str(si)
            else:
                s_bg, s_color, s_text = "#EDE9FE","#A78BFA",str(si)
            step_html += f'<div style="display:flex;align-items:center;{"flex:1;" if si < len(steps) else ""}">'
            step_html += f'<div style="width:30px;height:30px;border-radius:50%;background:{s_bg};color:{s_color};display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;flex-shrink:0;">{s_text}</div>'
            step_html += f'<span style="font-size:12px;font-weight:{"700" if si==step else "500"};color:{"#5B21B6" if si==step else "#6B7280"};margin-left:8px;white-space:nowrap;">{sl}</span>'
            if si < len(steps):
                step_html += f'<div style="flex:1;height:2px;background:{"#059669" if si<step else "#EDE9FE"};margin:0 12px;"></div>'
            step_html += '</div>'
        step_html += '</div>'
        st.markdown(step_html, unsafe_allow_html=True)

        col_form, col_info = st.columns([2, 1])
        with col_form:
            anon = st.toggle("🔒 Denuncia Anónima (Recomendado)", value=st.session_state.get("d_anon", True), key="d_anon_toggle")
            st.session_state["d_anon"] = anon
            anon_bg = "linear-gradient(135deg,#ECFDF5,#D1FAE5)" if anon else "linear-gradient(135deg,#F5F3FF,#EDE9FE)"
            anon_color = "#059669" if anon else "#5B21B6"
            anon_label = "✅ Denuncia 100% Anónima — Tu identidad está protegida" if anon else "👤 Denuncia con Identidad"
            st.markdown(f"""<div style="background:{anon_bg};border-radius:14px;padding:12px 16px;
                border:1px solid {anon_color}30;margin-bottom:20px;font-size:13px;font-weight:700;color:{anon_color};">
                {anon_label}</div>""", unsafe_allow_html=True)

            if step == 1:
                st.markdown('<div style="font-size:13px;font-weight:800;color:#5B21B6;margin-bottom:14px;">📌 Paso 1: Clasificación del hecho</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    d_delito = st.selectbox("⚖️ Tipo de Delito", ["— Selecciona —"] + DELITOS, key="d_delito")
                with c2:
                    d_dep = st.selectbox("🗺️ Departamento", DEPARTAMENTOS, key="d_dep")
                c3, c4 = st.columns(2)
                with c3:
                    d_fecha = st.date_input("📅 Fecha aproximada", key="d_fecha", value=None)
                with c4:
                    d_hora = st.time_input("🕐 Hora aproximada", key="d_hora", value=None)
                d_lugar = st.text_input("📍 Lugar del hecho", placeholder="Ej: Centro Comercial El Tesoro, Cll 45...", key="d_lugar")
                st.markdown('<div style="font-size:12px;font-weight:700;color:#5B21B6;margin:14px 0 8px;">🚨 Nivel de Urgencia</div>', unsafe_allow_html=True)
                urg_cols = st.columns(3)
                urgencias = [("🔴","INMEDIATA","En peligro ahora","#DC2626","#FEF2F2"),
                             ("🟡","URGENTE","Ocurrió recientemente","#D97706","#FFFBEB"),
                             ("🟢","NORMAL","Para registro y seguimiento","#059669","#ECFDF5")]
                sel_urg = st.session_state.get("d_urgencia","NORMAL")
                for ui, (uic, ulabel, udesc, ucolor, ubg) in enumerate(urgencias):
                    with urg_cols[ui]:
                        if st.button(f"{uic} {ulabel}", key=f"urg_{ulabel}", use_container_width=True):
                            st.session_state["d_urgencia"] = ulabel
                            st.rerun()
                        st.markdown(f'<div style="font-size:9px;color:{ucolor};text-align:center;margin-top:-10px;margin-bottom:8px;">{udesc}</div>', unsafe_allow_html=True)
                if st.button("Siguiente → Descripción del hecho", type="primary", use_container_width=True):
                    st.session_state["denuncia_step"] = 2
                    st.rerun()

            elif step == 2:
                st.markdown('<div style="font-size:13px;font-weight:800;color:#5B21B6;margin-bottom:14px;">📝 Paso 2: Descripción del hecho</div>', unsafe_allow_html=True)
                d_desc = st.text_area("Describe lo que ocurrió con detalle:", height=160,
                    placeholder="Describe con el mayor detalle posible. Todo es completamente confidencial...", key="d_desc")
                char_color = "#059669" if len(d_desc) > 50 else "#D97706"
                st.markdown(f'<div style="font-size:11px;color:{char_color};margin-top:-6px;margin-bottom:16px;font-weight:600;">{len(d_desc)} caracteres{"  ✓ Descripción suficiente" if len(d_desc)>50 else " — agrega más detalles"}</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:12px;font-weight:700;color:#5B21B6;margin-bottom:8px;">📎 Evidencia (fotos, audio, video)</div>', unsafe_allow_html=True)
                uploaded = st.file_uploader("Adjunta archivos de evidencia (opcional):", accept_multiple_files=True,
                    type=["jpg","jpeg","png","mp4","mp3","pdf","wav"], label_visibility="collapsed", key="d_files")
                if uploaded:
                    for f in uploaded:
                        st.markdown(f'<div style="font-size:12px;color:#059669;margin-bottom:4px;">✅ {f.name} adjuntado</div>', unsafe_allow_html=True)
                c_prev, c_next = st.columns(2)
                with c_prev:
                    if st.button("← Anterior", use_container_width=True):
                        st.session_state["denuncia_step"] = 1
                        st.rerun()
                with c_next:
                    if st.button("Siguiente → Opciones y Envío", type="primary", use_container_width=True, disabled=not st.session_state.get("d_desc","").strip()):
                        st.session_state["denuncia_step"] = 3
                        st.rerun()

            elif step == 3:
                st.markdown('<div style="font-size:13px;font-weight:800;color:#5B21B6;margin-bottom:14px;">✅ Paso 3: Opciones adicionales y envío</div>', unsafe_allow_html=True)
                opciones = ["Violencia física","Violencia verbal","Violencia psicológica","Violencia económica",
                            "Seguimiento / acoso","Violencia digital","Tengo evidencia","Quiero acompañamiento",
                            "Necesito protección urgente","Quiero mantener anonimato total","Involucra menores de edad"]
                d_opts = st.multiselect("Selecciona todas las que apliquen:", opciones, key="d_opts")
                st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;
                    padding:18px;margin-top:14px;border:1px solid #C4B5FD;margin-bottom:18px;">
                    <div style="font-size:13px;font-weight:800;color:#5B21B6;margin-bottom:12px;">📋 Resumen del Reporte</div>
                    {''.join([f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #DDD6FE;font-size:12px;"><span style="color:#6B7280;">{k}</span><span style="color:#1E1B4B;font-weight:700;">{v}</span></div>' for k,v in [
                        ("Modo", "🔒 Anónimo" if st.session_state.get("d_anon",True) else "👤 Con identidad"),
                        ("Urgencia", st.session_state.get("d_urgencia","NORMAL")),
                        ("Delito", st.session_state.get("d_delito","No especificado")),
                        ("Departamento", st.session_state.get("d_dep","No especificado")),
                        ("Opciones", f"{len(d_opts)} seleccionadas" if d_opts else "Ninguna"),
                    ]])}
                </div>""", unsafe_allow_html=True)
                c_prev2, c_send = st.columns(2)
                with c_prev2:
                    if st.button("← Anterior", use_container_width=True, key="prev2"):
                        st.session_state["denuncia_step"] = 2
                        st.rerun()
                with c_send:
                    send_btn = st.button("📤 Registrar y Obtener Orientación Legal",
                                         type="primary", use_container_width=True, key="denuncia_send",
                                         disabled=not st.session_state.get("d_desc","").strip())
                if send_btn and st.session_state.get("d_desc","").strip():
                    st.session_state["denuncia_sent"] = True
                    with st.spinner("⚖️ Preparando orientación jurídica personalizada..."):
                        legal_text = call_claude(
                            """Eres asistente jurídica especializada en derechos de la mujer en Colombia (Ley 1257/2008).
Usa EXACTAMENTE estas secciones con emojis:
⚖️ TUS DERECHOS INMEDIATOS
📋 PASOS A SEGUIR (ordenados)
🏢 ENTIDADES A CONTACTAR
📱 EVIDENCIA A RECOLECTAR
⏰ PLAZOS IMPORTANTES
Máximo 3 puntos por sección con bullet •. Tono cálido, empático y empoderador.""",
                            f"""Mujer reporta en Colombia:
Delito: {st.session_state.get('d_delito','No especificado')}
Departamento: {st.session_state.get('d_dep','No especificado')}
Lugar: {st.session_state.get('d_lugar','No especificado')}
Urgencia: {st.session_state.get('d_urgencia','NORMAL')}
Descripción: {st.session_state.get('d_desc','')}
Opciones seleccionadas: {', '.join(d_opts) if d_opts else 'Ninguna'}"""
                        )
                        st.session_state["legal_text"] = legal_text
                    st.rerun()

        with col_info:
            st.markdown("""<div class="sh-card">
                <div style="font-size:13px;font-weight:800;color:#5B21B6;margin-bottom:16px;">🏢 Entidades Oficiales</div></div>""", unsafe_allow_html=True)
            entidades_d = [
                ("⚖️","Fiscalía General","Denuncias penales en línea","https://www.fiscalia.gov.co","#7C3AED"),
                ("🏠","Comisaría de Familia","Violencia intrafamiliar","tel:123","#1D4ED8"),
                ("👨‍👩‍👧","Instituto ICBF","Protección familiar","https://www.icbf.gov.co","#059669"),
                ("📞","Línea 155","Mujer 24/7 — Gratis","tel:155","#EC4899"),
                ("🚨","URI Fiscalía 24h","Denuncia urgente sin cita","tel:018000919748","#DC2626"),
            ]
            for icon_e, name_e, desc_e, href_e, color_e in entidades_d:
                st.markdown(f"""<a href="{href_e}" target="{'_blank' if href_e.startswith('http') else '_self'}" style="text-decoration:none;">
                <div style="display:flex;align-items:center;gap:10px;padding:11px 0;border-bottom:1px solid #EDE9FE;">
                    <div style="width:34px;height:34px;background:linear-gradient(135deg,{color_e}18,{color_e}10);border-radius:10px;
                        display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">{icon_e}</div>
                    <div>
                        <div style="font-size:12px;font-weight:700;color:#1E1B4B;">{name_e}</div>
                        <div style="font-size:10px;color:#A78BFA;font-weight:500;">{desc_e}</div>
                    </div>
                </div></a>""", unsafe_allow_html=True)
            razones = ["✅ Protege a otras mujeres","✅ Genera registros estadísticos","✅ Activa medidas de protección",
                       "✅ Accedes a apoyo psicológico","✅ Rompe el ciclo de violencia"]
            r_html = "".join([f'<div style="font-size:12px;color:#1E1B4B;margin-bottom:8px;line-height:1.6;">{r}</div>' for r in razones])
            st.markdown(f'<div class="sh-card"><div style="font-size:13px;font-weight:800;color:#5B21B6;margin-bottom:12px;">🧠 ¿Por qué es importante denunciar?</div>{r_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:linear-gradient(135deg,#ECFDF5,#D1FAE5);border:2px solid #6EE7B7;
            border-radius:24px;padding:28px;margin-bottom:24px;display:flex;align-items:center;gap:18px;">
            <div style="width:56px;height:56px;background:#059669;border-radius:50%;display:flex;
                align-items:center;justify-content:center;font-size:26px;flex-shrink:0;
                box-shadow:0 4px 16px rgba(5,150,105,0.35);">✅</div>
            <div>
                <div style="font-size:20px;font-weight:900;color:#065F46;margin-bottom:4px;">Reporte registrado de forma segura</div>
                <div style="font-size:13px;color:#047857;line-height:1.6;">Tu información es completamente confidencial. Tu valentía importa.</div>
            </div>
        </div>""", unsafe_allow_html=True)
        legal = st.session_state.get("legal_text", "")
        st.markdown(f"""<div class="sh-card">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
                <div style="width:46px;height:46px;background:linear-gradient(135deg,#1E1B4B,#5B21B6);
                    border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:22px;">⚖️</div>
                <div>
                    <div style="font-size:16px;font-weight:800;color:#1E1B4B;">Orientación Jurídica Personalizada</div>
                    <div style="font-size:11px;color:#A78BFA;">Generada con IA especializada en Ley 1257/2008 — Colombia</div>
                </div>
            </div>
            <div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:16px;padding:20px 24px;
                border:1px solid #C4B5FD;font-size:13px;line-height:1.85;white-space:pre-wrap;color:#1E1B4B;">{legal}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("← Registrar nueva denuncia", type="secondary"):
            st.session_state["denuncia_sent"] = False
            st.session_state["denuncia_step"] = 1
            st.session_state.pop("legal_text", None)
            st.rerun()

# ── SARA · IA APOYO ───────────────────────────────────────────────────────────
elif "💜" in page:
    SARA_SYSTEM = """Eres SARA, asistente de apoyo empática, cálida y experta de SafeHer Colombia.
ROL PRINCIPAL: Acompañar y orientar a mujeres que pueden estar en situaciones de riesgo, violencia, crisis emocional o que buscan información.
PRINCIPIOS FUNDAMENTALES:
- Eres cálida, empática, paciente y NUNCA juzgas
- Siempre valida los sentimientos ANTES de dar cualquier consejo
- Si hay peligro inmediato (golpes, secuestro, amenazas): responde en ≤3 oraciones con 🚨 Llama al 123 INMEDIATAMENTE
- Nunca minimizas ni normalizas la violencia
APOYO PSICOLÓGICO:
- Ansiedad → técnica 4-7-8 (inhala 4s, retén 7s, exhala 8s)
- Crisis → grounding 5-4-3-2-1 (5 cosas ves, 4 tocas, 3 escuchas, 2 hueles, 1 sabores)
- Usa preguntas abiertas para entender mejor
- Celebra pequeños pasos: "Es muy valiente que estés buscando ayuda"
CONOCIMIENTO LEGAL:
- Ley 1257/2008, medidas de protección, órdenes de alejamiento
- Comisaría de Familia, Fiscalía URI 24h, Línea 155, ICBF Línea 141
FORMATO: Español cálido y cercano, máx 200 palabras, emojis con moderación (💜🌸). Crisis inmediata: máx 3 oraciones."""

    if "sara_messages" not in st.session_state:
        st.session_state.sara_messages = [
            {"role": "assistant", "content": "Hola 💜 Soy SARA, tu asistente de apoyo de SafeHer.\n\nEstoy aquí para escucharte, orientarte y acompañarte — sin juzgarte, completamente confidencial. Puedes contarme lo que estás viviendo, preguntar sobre tus derechos, buscar apoyo emocional, o simplemente desahogarte.\n\nEstoy aquí 24/7 para ti. ¿Cómo te puedo ayudar hoy? 🌸"}
        ]

    col_sidebar_sara, col_chat = st.columns([1, 3])
    with col_sidebar_sara:
        st.markdown("""<div style="background:linear-gradient(160deg,#1E1B4B 0%,#4C1D95 55%,#7C3AED 100%);
            border-radius:22px;padding:24px;color:#fff;text-align:center;margin-bottom:14px;
            box-shadow:0 6px 24px rgba(91,33,182,0.3);">
            <div style="width:68px;height:68px;background:rgba(255,255,255,0.14);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:32px;margin:0 auto 12px;
                border:2px solid rgba(255,255,255,0.24);">💜</div>
            <div style="font-weight:900;font-size:22px;letter-spacing:-0.5px;">SARA</div>
            <div style="font-size:11px;color:#C4B5FD;margin-bottom:14px;">Asistente SafeHer · IA Empática</div>
            <div style="display:flex;align-items:center;gap:7px;justify-content:center;">
                <div style="width:9px;height:9px;border-radius:50%;background:#4ADE80;box-shadow:0 0 10px #4ADE80;"></div>
                <span style="font-size:11px;color:#A7F3D0;font-weight:600;">En línea · 24/7</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin-bottom:10px;">¿Cómo te sientes ahora?</div>', unsafe_allow_html=True)
        mood_options = [("😰","Asustada"),("😢","Triste"),("😡","Enojada"),("😔","Sola"),("🙂","Bien"),("🆘","Urgente")]
        cols_mood = st.columns(3)
        for i, (emoji, label) in enumerate(mood_options):
            with cols_mood[i % 3]:
                if st.button(f"{emoji}", key=f"mood_{label}", use_container_width=True, help=label):
                    msg_content = f"Me siento {label.lower()} {emoji}"
                    st.session_state.sara_messages.append({"role": "user", "content": msg_content})
                    with st.spinner("💜"):
                        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.sara_messages]
                        reply = call_claude(SARA_SYSTEM, "", history=history)
                    st.session_state.sara_messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                st.markdown(f'<div style="font-size:9px;color:#A78BFA;text-align:center;margin-top:-8px;margin-bottom:6px;">{label}</div>', unsafe_allow_html=True)

        for ic, txt in [("🔒","Confidencial 100%"),("⚡","Respuesta empática"),("🧠","Técnicas de calma"),
                        ("⚖️","Orientación legal"),("📍","Recursos cercanos"),("💬","Escucharte sin juzgar"),
                        ("🌱","Apoyo psicológico"),("📱","Navegar la app")]:
            st.markdown(f'<div style="display:flex;gap:9px;margin-bottom:9px;align-items:flex-start;">'
                       f'<span style="font-size:15px;line-height:1.4;">{ic}</span>'
                       f'<span style="font-size:11px;color:#6B7280;line-height:1.5;">{txt}</span></div>', unsafe_allow_html=True)

        for num, desc, sub in [("123","Policía","24/7"),("155","Línea Mujer","Gratis 24/7"),("137","Salud Mental","Apoyo")]:
            st.markdown(f'<a href="tel:{num}" style="display:flex;justify-content:space-between;align-items:center;text-decoration:none;padding:9px 0;border-bottom:1px solid #FECDD3;">'
                       f'<span style="font-size:18px;color:#BE123C;font-weight:900;font-family:Georgia,serif;">{num}</span>'
                       f'<div style="text-align:right;"><div style="font-size:11px;color:#DC2626;font-weight:700;">{desc}</div>'
                       f'<div style="font-size:9px;color:#9F1239;font-weight:500;">{sub}</div></div></a>', unsafe_allow_html=True)

    with col_chat:
        st.markdown("""<div style="padding:18px 24px;border-bottom:1px solid #EDE9FE;display:flex;align-items:center;
            gap:14px;background:linear-gradient(135deg,#1E1B4B,#4C1D95);border-radius:22px 22px 0 0;
            box-shadow:0 4px 16px rgba(30,27,75,0.3);">
            <div style="width:42px;height:42px;background:rgba(255,255,255,0.14);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:22px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);">💜</div>
            <div>
                <div style="font-weight:800;font-size:15px;color:#fff;letter-spacing:-0.3px;">SARA — Asistente SafeHer</div>
                <div style="font-size:11px;color:#A7F3D0;display:flex;align-items:center;gap:6px;margin-top:3px;">
                    <span style="width:7px;height:7px;border-radius:50%;background:#4ADE80;display:inline-block;box-shadow:0 0 6px #4ADE80;"></span>
                    En línea · Siempre disponible · Completamente confidencial
                </div>
            </div>
            <div style="margin-left:auto;display:flex;gap:8px;">
                <a href="tel:155" style="background:rgba(255,255,255,0.1);color:#E9D5FF;padding:7px 14px;
                    border-radius:20px;font-size:11px;text-decoration:none;font-weight:700;
                    border:1px solid rgba(255,255,255,0.2);">📞 155</a>
                <a href="tel:123" style="background:rgba(220,38,38,0.35);color:#FCA5A5;padding:7px 14px;
                    border-radius:20px;font-size:11px;text-decoration:none;font-weight:700;
                    border:1px solid rgba(220,38,38,0.4);">🚨 123</a>
            </div>
        </div>""", unsafe_allow_html=True)

        msgs_html = ""
        for msg in st.session_state.sara_messages:
            if msg["role"] == "user":
                msgs_html += f"""<div style="display:flex;justify-content:flex-end;gap:10px;margin-bottom:16px;">
                    <div class="chat-user">{msg['content']}</div>
                    <div style="width:34px;height:34px;background:linear-gradient(135deg,#EDE9FE,#DDD6FE);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;margin-top:2px;">👤</div>
                </div>"""
            else:
                content = msg['content'].replace('\n', '<br>')
                msgs_html += f"""<div style="display:flex;gap:10px;margin-bottom:16px;">
                    <div style="width:34px;height:34px;background:linear-gradient(135deg,#1E1B4B,#7C3AED);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;margin-top:2px;
                        box-shadow:0 3px 10px rgba(124,58,237,0.3);">💜</div>
                    <div class="chat-sara">{content}</div>
                </div>"""

        st.markdown(f"""<div style="background:#fff;padding:24px;min-height:380px;max-height:420px;
            border-left:1px solid #EDE9FE;border-right:1px solid #EDE9FE;overflow-y:auto;">
            {msgs_html}
        </div>""", unsafe_allow_html=True)

        quick_replies = ["Necesito ayuda urgente 🆘","¿Cómo denuncio?","Me siento sola y asustada",
                         "¿Cuáles son mis derechos?","Ejercicio para calmarme 🧘","Me están amenazando",
                         "¿Qué hace esta app?","Apoyo psicológico"]
        qcols = st.columns(4)
        for i, q in enumerate(quick_replies):
            with qcols[i % 4]:
                if st.button(q, key=f"qr_{i}", use_container_width=True):
                    st.session_state.sara_messages.append({"role": "user", "content": q})
                    with st.spinner("SARA está escribiendo..."):
                        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.sara_messages]
                        reply = call_claude(SARA_SYSTEM, "", history=history)
                    st.session_state.sara_messages.append({"role": "assistant", "content": reply})
                    st.rerun()

        with st.form("sara_form", clear_on_submit=True):
            col_inp, col_send = st.columns([5, 1])
            with col_inp:
                user_input = st.text_input("", placeholder="Escribe tu mensaje... Estoy aquí para escucharte 💜",
                                           label_visibility="collapsed", key="sara_input")
            with col_send:
                send_sara = st.form_submit_button("➤ Enviar", use_container_width=True)

        if send_sara and user_input.strip():
            st.session_state.sara_messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("SARA está escribiendo..."):
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.sara_messages]
                reply = call_claude(SARA_SYSTEM, "", history=history)
            st.session_state.sara_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        col_clear, col_info_sara = st.columns([1, 2])
        with col_clear:
            if st.button("🔄 Nueva conversación", use_container_width=True):
                st.session_state.sara_messages = [
                    {"role": "assistant", "content": "Hola 💜 Soy SARA. Estoy aquí para escucharte. ¿Cómo te puedo ayudar?"}
                ]
                st.rerun()
        with col_info_sara:
            st.markdown('<div style="font-size:11px;color:#A78BFA;padding:8px 0;">🔒 Esta conversación es completamente confidencial y no se almacena de forma permanente.</div>', unsafe_allow_html=True)


# ── AYUDA CERCANA ─────────────────────────────────────────────────────────────
elif "🚔" in page:
    # ── Base de datos de entidades por ciudad colombiana ──────────────────────
    ENTIDADES_COL = {

        # ══════════════════════════════════════════════════════════════════════
        # BOGOTÁ D.C.
        # ══════════════════════════════════════════════════════════════════════
        "bogotá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Centro",
             "dir": "Carrera 9 #15-55, Bogotá", "barrio": "La Candelaria",
             "lat": 4.5981, "lon": -74.0721, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana. Denuncia inmediata y medidas de protección.",
             "transporte": "TransMilenio: Portal Centro (5 min) · Bus: múltiples rutas Cra 10"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital La Victoria",
             "dir": "Calle 1 #18-98, Bogotá", "barrio": "Santa Inés",
             "lat": 4.5883, "lon": -74.0927, "color": "#059669",
             "href": "tel:3649900", "phone": "364-9900", "horario": "24/7 Urgencias",
             "desc": "Hospital público de alta complejidad. Urgencias, medicina forense, apoyo psicológico para víctimas.",
             "transporte": "TransMilenio: Av. Jiménez (10 min caminando) · Bus: Cll 1"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Bogotá 24h",
             "dir": "Calle 8 #69B-41, Bogotá", "barrio": "Paloquemao",
             "lat": 4.6250, "lon": -74.0980, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas, sin cita previa.",
             "transporte": "TransMilenio: Paloquemao (3 min caminando) · Bus: Av. Calle 6"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia N°1",
             "dir": "Cra 24 #34-17, Bogotá", "barrio": "Teusaquillo",
             "lat": 4.6330, "lon": -74.0860, "color": "#0891B2",
             "href": "tel:123", "phone": "123 / Presencial", "horario": "Lun–Vie 7am–4pm",
             "desc": "Medidas de protección por violencia intrafamiliar. Órdenes de alejamiento inmediatas.",
             "transporte": "TransMilenio: Parkway (8 min caminando)"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Refugio Benposta",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.6200, "lon": -74.0800, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro y gratuito para mujeres víctimas de violencia y sus hijos e hijas.",
             "transporte": "Llama al 155 (gratuito). Coordinan transporte seguro y discreto"},
            {"tipo": "Psicología", "icon": "🧠", "nom": "CAIVAS Bogotá",
             "dir": "Carrera 52 #42-43, Bogotá", "barrio": "Paloquemao",
             "lat": 4.6330, "lon": -74.0990, "color": "#8B5CF6",
             "href": "tel:3159700", "phone": "315-9700", "horario": "Lun–Vie 8am–5pm",
             "desc": "Centro de Atención Integral a Víctimas de Violencia Sexual. Atención psicológica, jurídica y social gratuita.",
             "transporte": "TransMilenio: Paloquemao (6 min caminando)"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CUNDINAMARCA
        # ══════════════════════════════════════════════════════════════════════
        "chía": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Chía",
             "dir": "Calle 11 #10-50, Chía", "barrio": "Centro",
             "lat": 4.8600, "lon": -74.0462, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Chía. Denuncias y medidas de protección inmediatas.",
             "transporte": "Bus intermunicipal Portal Norte o Zipaquirá · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital de Chía",
             "dir": "Carrera 11 #15-20, Chía", "barrio": "Centro",
             "lat": 4.8610, "lon": -74.0470, "color": "#059669",
             "href": "tel:8616200", "phone": "861-6200", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal con urgencias. Atención médica y apoyo a víctimas de violencia.",
             "transporte": "Bus desde Portal Norte · 30 min desde Bogotá"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "Fiscalía Seccional Chía",
             "dir": "Carrera 10 #14-30, Chía", "barrio": "Centro",
             "lat": 4.8595, "lon": -74.0455, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748",
             "horario": "Lun–Vie 8am–5pm · Urgencias 24/7",
             "desc": "Seccional Fiscalía. Denuncias penales. Para urgencias fuera de horario llama al 018000919748.",
             "transporte": "Bus desde Portal Norte · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Chía",
             "dir": "Calle 12 #9-40, Chía", "barrio": "Centro",
             "lat": 4.8605, "lon": -74.0465, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial / 123", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar. Apoyo psicosocial municipal.",
             "transporte": "Bus desde Portal Norte · 30 min desde Bogotá"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Chía",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.8590, "lon": -74.0450, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro para mujeres víctimas de violencia. Coordinado con la Línea 155 nacional.",
             "transporte": "Llama al 155 — coordinan transporte seguro desde cualquier punto"},
        ],
        "soacha": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Soacha",
             "dir": "Cra 5 #13-20, Soacha", "barrio": "Centro",
             "lat": 4.5790, "lon": -74.2170, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Soacha. Denuncias, emergencias y medidas de protección.",
             "transporte": "TransMilenio: Portal Sur (5 min) · Bus a Soacha Centro"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Mario Gaitán Yanguas",
             "dir": "Calle 13 #7-20, Soacha", "barrio": "Centro",
             "lat": 4.5800, "lon": -74.2160, "color": "#059669",
             "href": "tel:7299000", "phone": "729-9000", "horario": "24/7 Urgencias",
             "desc": "Principal hospital de Soacha. Urgencias, medicina forense y apoyo a víctimas de violencia.",
             "transporte": "TransMilenio: Portal Sur + bus alimentador · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Soacha",
             "dir": "Calle 14 #6-50, Soacha", "barrio": "Centro",
             "lat": 4.5785, "lon": -74.2180, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes sin cita previa.",
             "transporte": "Portal Sur + bus a Soacha Centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Soacha",
             "dir": "Cra 6 #12-10, Soacha", "barrio": "Centro",
             "lat": 4.5795, "lon": -74.2155, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección, conciliación y apoyo psicosocial gratuito.",
             "transporte": "TransMilenio: Portal Sur + bus · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Refugio Soacha",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.5788, "lon": -74.2165, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro para mujeres víctimas de violencia. Gratuito con coordinación de la Línea 155.",
             "transporte": "Llama al 155 — coordinan transporte discreto"},
        ],
        "zipaquirá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Zipaquirá",
             "dir": "Calle 3 #8-30, Zipaquirá", "barrio": "Centro",
             "lat": 5.0220, "lon": -74.0050, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Zipaquirá. Denuncias y atención permanente.",
             "transporte": "Tren de cercanías desde Bogotá (1h) · Bus intermunicipal · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Juan de Dios Zipaquirá",
             "dir": "Calle 5 #6-25, Zipaquirá", "barrio": "Centro",
             "lat": 5.0215, "lon": -74.0060, "color": "#059669",
             "href": "tel:8522700", "phone": "852-2700", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias, medicina y apoyo a víctimas de violencia intrafamiliar.",
             "transporte": "Tren o bus desde Bogotá · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Zipaquirá",
             "dir": "Cra 9 #3-15, Zipaquirá", "barrio": "Centro",
             "lat": 5.0225, "lon": -74.0045, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar. Apoyo psicosocial.",
             "transporte": "Tren o bus desde Bogotá · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Zipaquirá",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.0218, "lon": -74.0055, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con la Línea 155 nacional.",
             "transporte": "Llama al 155 — coordinan transporte seguro"},
        ],
        "tocancipá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Tocancipá",
             "dir": "Calle 5 #5-10, Tocancipá", "barrio": "Centro",
             "lat": 5.0060, "lon": -73.9100, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Tocancipá. Denuncias y emergencias las 24 horas.",
             "transporte": "Bus intermunicipal desde Portal Norte (Bogotá) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Antonio Tocancipá",
             "dir": "Carrera 6 #4-20, Tocancipá", "barrio": "Centro",
             "lat": 5.0065, "lon": -73.9095, "color": "#059669",
             "href": "tel:8562200", "phone": "856-2200", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal con urgencias. Atención médica y apoyo inicial a víctimas de violencia.",
             "transporte": "Bus desde Portal Norte · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Tocancipá",
             "dir": "Cra 5 #3-50, Tocancipá", "barrio": "Centro",
             "lat": 5.0055, "lon": -73.9105, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial / 123", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección, atención psicosocial y orientación jurídica.",
             "transporte": "Bus desde Portal Norte · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Tocancipá",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.0060, "lon": -73.9100, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Atención y alojamiento seguro para mujeres víctimas de violencia. Coordinado con Línea 155.",
             "transporte": "Llama al 155 — coordinan transporte discreto"},
        ],
        "facatativá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Facatativá",
             "dir": "Calle 9 #10-30, Facatativá", "barrio": "Centro",
             "lat": 4.8150, "lon": -74.3550, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Facatativá. Denuncias y medidas de protección.",
             "transporte": "Bus Bogotá–Facatativá (1h) · Sitp / Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael Facatativá",
             "dir": "Cra 11 #7-25, Facatativá", "barrio": "Centro",
             "lat": 4.8155, "lon": -74.3545, "color": "#059669",
             "href": "tel:8920880", "phone": "892-0880", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y apoyo a víctimas de violencia intrafamiliar.",
             "transporte": "Bus desde Bogotá (terminal occidente) · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Facatativá",
             "dir": "Calle 8 #9-50, Facatativá", "barrio": "Centro",
             "lat": 4.8145, "lon": -74.3555, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar. Apoyo psicosocial.",
             "transporte": "Bus desde Bogotá (terminal occidente) · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Facatativá",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.8150, "lon": -74.3550, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con la Línea 155 nacional.",
             "transporte": "Llama al 155 — coordinan transporte seguro"},
        ],
        "fusagasugá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Fusagasugá",
             "dir": "Cra 6 #8-20, Fusagasugá", "barrio": "Centro",
             "lat": 4.3370, "lon": -74.3640, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Fusagasugá. Denuncias y emergencias.",
             "transporte": "Bus Bogotá–Fusagasugá (1.5h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael Fusagasugá",
             "dir": "Calle 7 #5-30, Fusagasugá", "barrio": "Centro",
             "lat": 4.3375, "lon": -74.3635, "color": "#059669",
             "href": "tel:8741516", "phone": "874-1516", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal con urgencias. Atención médica y apoyo a víctimas de violencia.",
             "transporte": "Bus desde Bogotá · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Fusagasugá",
             "dir": "Cra 7 #9-15, Fusagasugá", "barrio": "Centro",
             "lat": 4.3365, "lon": -74.3645, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección, conciliación y apoyo psicosocial gratuito.",
             "transporte": "Bus desde Bogotá · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Fusagasugá",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.3370, "lon": -74.3640, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155 nacional.",
             "transporte": "Llama al 155 — coordinan transporte"},
        ],
        "girardot": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Girardot",
             "dir": "Calle 8 #9-50, Girardot", "barrio": "Centro",
             "lat": 4.3030, "lon": -74.8020, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Girardot. Denuncias y medidas de protección.",
             "transporte": "Bus Bogotá–Girardot (2h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael Girardot",
             "dir": "Cra 6 #25-50, Girardot", "barrio": "Centro",
             "lat": 4.3025, "lon": -74.8030, "color": "#059669",
             "href": "tel:8332020", "phone": "833-2020", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y apoyo a víctimas.",
             "transporte": "Bus desde Bogotá · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Girardot",
             "dir": "Cra 7 #8-30, Girardot", "barrio": "Centro",
             "lat": 4.3035, "lon": -74.8015, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Bogotá · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Girardot",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.3030, "lon": -74.8020, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # ANTIOQUIA
        # ══════════════════════════════════════════════════════════════════════
        "medellín": [
            {"tipo": "Policía", "icon": "🚔", "nom": "CAI Centro Medellín",
             "dir": "Carrera 45 #54-20, Medellín", "barrio": "Centro",
             "lat": 6.2442, "lon": -75.5742, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Centro de Atención Inmediata. Atención permanente para denuncias y emergencias policiales.",
             "transporte": "Metro: Prado (5 min caminando) · Bus: múltiples rutas Cra 45"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital General de Medellín",
             "dir": "Calle 24 #29-6, Medellín", "barrio": "Bomboná",
             "lat": 6.2358, "lon": -75.5730, "color": "#059669",
             "href": "tel:4441227", "phone": "444-1227", "horario": "24/7 Urgencias",
             "desc": "Hospital público con urgencias completas, medicina forense y apoyo psicológico para víctimas de violencia.",
             "transporte": "Bus: rutas por Cll 24 · Metro: Industriales (12 min caminando)"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Medellín 24h",
             "dir": "Calle 57 #45-129, Medellín", "barrio": "Niquitao",
             "lat": 6.2570, "lon": -75.5690, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas, sin necesidad de cita.",
             "transporte": "Metro: Hospital (10 min caminando) · Bus: Cll 57"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia N°1",
             "dir": "Carrera 52 #48-10, Medellín", "barrio": "El Centro",
             "lat": 6.2510, "lon": -75.5700, "color": "#0891B2",
             "href": "tel:123", "phone": "123 / Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección familiar, conciliación y apoyo psicosocial integral. Sin costo.",
             "transporte": "Metro: Alpujarra (12 min caminando) · Bus: Cra 52"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Refugio Luz y Esperanza",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.2420, "lon": -75.5750, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento temporal gratuito y seguro para mujeres víctimas de violencia y sus hijos.",
             "transporte": "Llama al 155 para coordinación de transporte seguro y discreto"},
            {"tipo": "Psicología", "icon": "🧠", "nom": "CAIVAS Medellín",
             "dir": "Calle 50 #40-20, Medellín", "barrio": "Prado",
             "lat": 6.2540, "lon": -75.5720, "color": "#8B5CF6",
             "href": "tel:3856600", "phone": "385-6600", "horario": "Lun–Sáb 8am–8pm",
             "desc": "Centro de Atención Integral a Víctimas. Psicología gratuita, terapia individual y grupos de apoyo.",
             "transporte": "Bus: Cll 50 (múltiples rutas) · Metro: Prado (10 min caminando)"},
        ],
        "bello": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Bello",
             "dir": "Cra 50 #34-20, Bello", "barrio": "Centro",
             "lat": 6.3370, "lon": -75.5590, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Bello. Denuncias y medidas de protección inmediatas.",
             "transporte": "Metro: Bello (5 min caminando)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Marco Fidel Suárez",
             "dir": "Calle 30 #48-50, Bello", "barrio": "Centro",
             "lat": 6.3380, "lon": -75.5580, "color": "#059669",
             "href": "tel:4502050", "phone": "450-2050", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal de Bello. Urgencias y apoyo a víctimas de violencia.",
             "transporte": "Metro: Bello + bus alimentador"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Bello",
             "dir": "Cra 51 #33-10, Bello", "barrio": "Centro",
             "lat": 6.3375, "lon": -75.5585, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Metro: Bello (8 min caminando)"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Bello",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.3370, "lon": -75.5590, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "itagüí": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Itagüí",
             "dir": "Cra 52 #41-20, Itagüí", "barrio": "Centro",
             "lat": 6.1850, "lon": -75.6000, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Itagüí. Denuncias y emergencias las 24 horas.",
             "transporte": "Metro: Itagüí (8 min caminando)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital del Sur Itagüí",
             "dir": "Calle 50 #55-30, Itagüí", "barrio": "El Centro",
             "lat": 6.1860, "lon": -75.5995, "color": "#059669",
             "href": "tel:3735600", "phone": "373-5600", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y atención a víctimas de violencia.",
             "transporte": "Metro: Itagüí + taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Itagüí",
             "dir": "Cra 53 #40-15, Itagüí", "barrio": "Centro",
             "lat": 6.1845, "lon": -75.6005, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar y apoyo psicosocial.",
             "transporte": "Metro: Itagüí (10 min caminando)"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Itagüí",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.1850, "lon": -75.6000, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "envigado": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Envigado",
             "dir": "Cra 43C #36Sur-20, Envigado", "barrio": "El Centro",
             "lat": 6.1750, "lon": -75.5870, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Envigado. Denuncias y medidas de protección.",
             "transporte": "Metro: Envigado (10 min caminando)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Manuel Uribe Ángel",
             "dir": "Calle 38Sur #50-50, Envigado", "barrio": "El Centro",
             "lat": 6.1755, "lon": -75.5865, "color": "#059669",
             "href": "tel:3390380", "phone": "339-0380", "horario": "24/7 Urgencias",
             "desc": "Hospital de Envigado. Urgencias y apoyo a víctimas de violencia.",
             "transporte": "Metro: Envigado + taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Envigado",
             "dir": "Cra 44 #37Sur-10, Envigado", "barrio": "El Centro",
             "lat": 6.1745, "lon": -75.5875, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección, conciliación y apoyo psicosocial.",
             "transporte": "Metro: Envigado (12 min caminando)"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Envigado",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.1750, "lon": -75.5870, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "apartadó": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Apartadó",
             "dir": "Cra 100 #95-30, Apartadó", "barrio": "Centro",
             "lat": 7.8800, "lon": -76.6300, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Apartadó. Denuncias y medidas de protección.",
             "transporte": "Bus local · Mototaxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Antonio Roldán Betancur",
             "dir": "Calle 90 #106-50, Apartadó", "barrio": "Centro",
             "lat": 7.8810, "lon": -76.6310, "color": "#059669",
             "href": "tel:8284040", "phone": "828-4040", "horario": "24/7 Urgencias",
             "desc": "Hospital de Apartadó. Urgencias y atención a víctimas de violencia.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Apartadó",
             "dir": "Cra 101 #94-20, Apartadó", "barrio": "Centro",
             "lat": 7.8795, "lon": -76.6295, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Apartadó",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.8800, "lon": -76.6300, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "turbo": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Turbo",
             "dir": "Cra 14 #102-20, Turbo", "barrio": "Centro",
             "lat": 8.0970, "lon": -76.7260, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Turbo. Denuncias y emergencias.",
             "transporte": "Bus local · Mototaxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Francisco Valderrama",
             "dir": "Calle 100 #12-50, Turbo", "barrio": "Centro",
             "lat": 8.0980, "lon": -76.7270, "color": "#059669",
             "href": "tel:8279000", "phone": "827-9000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Turbo. Urgencias y atención a víctimas.",
             "transporte": "Taxi · Bus local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Turbo",
             "dir": "Cra 13 #101-30, Turbo", "barrio": "Centro",
             "lat": 8.0965, "lon": -76.7255, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Turbo",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 8.0970, "lon": -76.7260, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # VALLE DEL CAUCA
        # ══════════════════════════════════════════════════════════════════════
        "cali": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Cali",
             "dir": "Carrera 6 #10-35, Cali", "barrio": "San Pedro",
             "lat": 3.4516, "lon": -76.5320, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana de Cali. Denuncias y medidas de protección inmediatas.",
             "transporte": "MÍO: Estación San Bosco (7 min caminando)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario del Valle",
             "dir": "Calle 5 #36-8, Cali", "barrio": "San Fernando",
             "lat": 3.4500, "lon": -76.5470, "color": "#059669",
             "href": "tel:5547374", "phone": "554-7374", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias, medicina forense y atención psicológica para víctimas.",
             "transporte": "MÍO: Estación Meléndez (10 min caminando)"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Cali 24h",
             "dir": "Carrera 3 #17-08, Cali", "barrio": "San Nicolás",
             "lat": 3.4553, "lon": -76.5290, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas.",
             "transporte": "MÍO: Estación Belalcázar (5 min caminando)"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Cali",
             "dir": "Cra 8 #9-56, Cali", "barrio": "Centro",
             "lat": 3.4525, "lon": -76.5310, "color": "#0891B2",
             "href": "tel:8816060", "phone": "881-6060", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar. Apoyo psicosocial.",
             "transporte": "MÍO: Estación Santa Librada (8 min caminando)"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Acogida Mujer Cali",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.4490, "lon": -76.5350, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Refugio seguro y gratuito para mujeres en situación de violencia y sus hijos.",
             "transporte": "Llama al 155 — coordinan transporte seguro"},
            {"tipo": "Psicología", "icon": "🧠", "nom": "CAIVAS Cali",
             "dir": "Carrera 5 #16-40, Cali", "barrio": "El Peñón",
             "lat": 3.4560, "lon": -76.5300, "color": "#8B5CF6",
             "href": "tel:8831434", "phone": "883-1434", "horario": "Lun–Vie 8am–5pm",
             "desc": "Centro de Atención Integral a Víctimas de Violencia Sexual. Atención gratuita.",
             "transporte": "MÍO: Estación Chapinero (12 min caminando)"},
        ],
        "palmira": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Palmira",
             "dir": "Cra 29 #28-50, Palmira", "barrio": "Centro",
             "lat": 3.5330, "lon": -76.3030, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Palmira. Denuncias y medidas de protección.",
             "transporte": "Bus Cali–Palmira (45 min) · MÍO conexión"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Raúl Orejuela Bueno",
             "dir": "Calle 28 #26-25, Palmira", "barrio": "Centro",
             "lat": 3.5335, "lon": -76.3025, "color": "#059669",
             "href": "tel:2744848", "phone": "274-4848", "horario": "24/7 Urgencias",
             "desc": "Hospital de Palmira. Urgencias y apoyo a víctimas de violencia.",
             "transporte": "Bus desde Cali · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Palmira",
             "dir": "Cra 30 #27-10, Palmira", "barrio": "Centro",
             "lat": 3.5325, "lon": -76.3035, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Cali · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Palmira",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.5330, "lon": -76.3030, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "buenaventura": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Buenaventura",
             "dir": "Cra 5 #2-30, Buenaventura", "barrio": "El Centro",
             "lat": 3.8850, "lon": -77.0240, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Buenaventura. Denuncias y emergencias.",
             "transporte": "Bus desde Cali (3h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Luis Ablanque de la Plata",
             "dir": "Calle 3 #3-50, Buenaventura", "barrio": "El Centro",
             "lat": 3.8855, "lon": -77.0235, "color": "#059669",
             "href": "tel:2428080", "phone": "242-8080", "horario": "24/7 Urgencias",
             "desc": "Principal hospital de Buenaventura. Urgencias y medicina forense.",
             "transporte": "Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Buenaventura",
             "dir": "Cra 6 #1-40, Buenaventura", "barrio": "El Centro",
             "lat": 3.8845, "lon": -77.0245, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar y apoyo psicosocial.",
             "transporte": "Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Buenaventura",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.8850, "lon": -77.0240, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "tuluá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Tuluá",
             "dir": "Cra 26 #24-30, Tuluá", "barrio": "Centro",
             "lat": 4.0850, "lon": -76.2000, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Tuluá. Denuncias y medidas de protección.",
             "transporte": "Bus Cali–Tuluá (1.5h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Tomás Uribe Uribe",
             "dir": "Calle 25 #27-50, Tuluá", "barrio": "Centro",
             "lat": 4.0860, "lon": -76.2010, "color": "#059669",
             "href": "tel:2243000", "phone": "224-3000", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y apoyo a víctimas.",
             "transporte": "Bus desde Cali · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Tuluá",
             "dir": "Cra 27 #23-40, Tuluá", "barrio": "Centro",
             "lat": 4.0845, "lon": -76.1995, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Cali · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Tuluá",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.0850, "lon": -76.2000, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "buga": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Buga",
             "dir": "Cra 14 #6-20, Buga", "barrio": "Centro",
             "lat": 3.9000, "lon": -76.3000, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Guadalajara de Buga. Denuncias y emergencias.",
             "transporte": "Bus desde Cali (1.5h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Juan de Dios Buga",
             "dir": "Calle 8 #13-50, Buga", "barrio": "Centro",
             "lat": 3.9005, "lon": -76.3005, "color": "#059669",
             "href": "tel:2368888", "phone": "236-8888", "horario": "24/7 Urgencias",
             "desc": "Hospital de Buga. Urgencias y atención a víctimas de violencia.",
             "transporte": "Bus desde Cali · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Buga",
             "dir": "Cra 15 #5-30, Buga", "barrio": "Centro",
             "lat": 3.8995, "lon": -76.2995, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Cali · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Buga",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.9000, "lon": -76.3000, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # ATLÁNTICO
        # ══════════════════════════════════════════════════════════════════════
        "barranquilla": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro",
             "dir": "Calle 35 #43-50, Barranquilla", "barrio": "El Centro",
             "lat": 10.9639, "lon": -74.7964, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana de Barranquilla. Denuncias y emergencias.",
             "transporte": "Bus: Cll 35 (múltiples rutas)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario CARI",
             "dir": "Calle 23 #16-16, Barranquilla", "barrio": "Barrio Abajo",
             "lat": 10.9822, "lon": -74.7896, "color": "#059669",
             "href": "tel:3440001", "phone": "344-0001", "horario": "24/7 Urgencias",
             "desc": "Hospital público. Urgencias, medicina forense y apoyo psicológico para víctimas de violencia.",
             "transporte": "Bus: Cll 23 (rutas directas)"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Barranquilla",
             "dir": "Calle 32 #51-12, Barranquilla", "barrio": "San Roque",
             "lat": 10.9630, "lon": -74.8130, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Denuncias penales urgentes las 24 horas, sin necesidad de cita previa.",
             "transporte": "Bus: Av. Murillo (5 min caminando)"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Barranquilla",
             "dir": "Cra 46 #48-50, Barranquilla", "barrio": "Centro",
             "lat": 10.9600, "lon": -74.8050, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus: Av. Olaya Herrera (rutas directas)"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Barranquilla",
             "dir": "Dirección confidencial — Línea 155", "barrio": "Confidencial",
             "lat": 10.9650, "lon": -74.8000, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro y gratuito para mujeres víctimas de violencia.",
             "transporte": "Llama al 155 — coordinan transporte seguro y discreto"},
        ],
        "soledad": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Soledad",
             "dir": "Cra 21 #18-30, Soledad", "barrio": "Centro",
             "lat": 10.9180, "lon": -74.7670, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Soledad. Denuncias y emergencias las 24 horas.",
             "transporte": "Bus desde Barranquilla (20 min) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital de Soledad",
             "dir": "Calle 18 #19-50, Soledad", "barrio": "Centro",
             "lat": 10.9185, "lon": -74.7665, "color": "#059669",
             "href": "tel:3751515", "phone": "375-1515", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal de Soledad. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Barranquilla · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Soledad",
             "dir": "Cra 22 #17-40, Soledad", "barrio": "Centro",
             "lat": 10.9175, "lon": -74.7675, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Barranquilla · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Soledad",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.9180, "lon": -74.7670, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "malambo": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Malambo",
             "dir": "Cra 5 #8-20, Malambo", "barrio": "Centro",
             "lat": 10.8550, "lon": -74.7730, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Malambo. Denuncias y emergencias.",
             "transporte": "Bus desde Barranquilla (25 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital de Malambo",
             "dir": "Calle 9 #4-50, Malambo", "barrio": "Centro",
             "lat": 10.8555, "lon": -74.7725, "color": "#059669",
             "href": "tel:3682200", "phone": "368-2200", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Barranquilla · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Malambo",
             "dir": "Cra 6 #7-30, Malambo", "barrio": "Centro",
             "lat": 10.8545, "lon": -74.7735, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Barranquilla · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Malambo",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.8550, "lon": -74.7730, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "sabanagrande": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Sabanagrande",
             "dir": "Cra 6 #4-20, Sabanagrande", "barrio": "Centro",
             "lat": 10.7900, "lon": -74.7560, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Sabanagrande. Denuncias y emergencias.",
             "transporte": "Bus desde Barranquilla · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Centro de Salud Sabanagrande",
             "dir": "Calle 5 #5-30, Sabanagrande", "barrio": "Centro",
             "lat": 10.7905, "lon": -74.7555, "color": "#059669",
             "href": "tel:3211234", "phone": "321-1234",
             "horario": "Lun–Dom 6am–10pm · Urgencias básicas",
             "desc": "Centro de salud municipal. Para urgencias críticas remite al HUB de Barranquilla.",
             "transporte": "Bus o taxi desde Barranquilla"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Sabanagrande",
             "dir": "Cra 7 #3-40, Sabanagrande", "barrio": "Centro",
             "lat": 10.7895, "lon": -74.7565, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus o taxi desde Barranquilla"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Sabanagrande",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.7900, "lon": -74.7560, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Coordinado con Línea 155. Transporte discreto a refugio seguro.",
             "transporte": "Llama al 155"},
        ],
        "sabanalarga": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Sabanalarga",
             "dir": "Cra 18 #8-30, Sabanalarga", "barrio": "Centro",
             "lat": 10.6320, "lon": -74.9210, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Sabanalarga. Denuncias y medidas de protección.",
             "transporte": "Bus desde Barranquilla (45 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental de Sabanalarga",
             "dir": "Calle 9 #20-50, Sabanalarga", "barrio": "Centro",
             "lat": 10.6325, "lon": -74.9215, "color": "#059669",
             "href": "tel:8491000", "phone": "849-1000", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Barranquilla · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Sabanalarga",
             "dir": "Cra 17 #9-20, Sabanalarga", "barrio": "Centro",
             "lat": 10.6315, "lon": -74.9205, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Barranquilla · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Sabanalarga",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.6320, "lon": -74.9210, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # SANTANDER
        # ══════════════════════════════════════════════════════════════════════
        "bucaramanga": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Cabecera",
             "dir": "Cra 35 #48-70, Bucaramanga", "barrio": "Cabecera del Llano",
             "lat": 7.0810, "lon": -73.1198, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana. Denuncias y medidas de protección inmediatas.",
             "transporte": "Bus: Cra 35 (múltiples rutas)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario de Santander",
             "dir": "Carrera 33 #28-126, Bucaramanga", "barrio": "Centro",
             "lat": 7.1185, "lon": -73.1250, "color": "#059669",
             "href": "tel:6346110", "phone": "634-6110", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus: Av. Quebrada Seca (rutas directas)"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Bucaramanga",
             "dir": "Calle 36 #22-50, Bucaramanga", "barrio": "Centro",
             "lat": 7.1198, "lon": -73.1260, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias las 24 horas.",
             "transporte": "Bus: Cll 36 (rutas directas)"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Bucaramanga",
             "dir": "Cra 20 #34-55, Bucaramanga", "barrio": "Centro",
             "lat": 7.1195, "lon": -73.1245, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar y apoyo psicosocial.",
             "transporte": "Bus: rutas Centro"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Bucaramanga",
             "dir": "Dirección confidencial — Línea 155", "barrio": "Confidencial",
             "lat": 7.1150, "lon": -73.1200, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro para mujeres víctimas de violencia.",
             "transporte": "Llama al 155 — coordinan transporte seguro"},
        ],
        "floridablanca": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Floridablanca",
             "dir": "Cra 8 #5-30, Floridablanca", "barrio": "El Centro",
             "lat": 7.0640, "lon": -73.0890, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Floridablanca. Denuncias y emergencias.",
             "transporte": "Bus desde Bucaramanga (20 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Manuela Beltrán Floridablanca",
             "dir": "Calle 6 #9-20, Floridablanca", "barrio": "Centro",
             "lat": 7.0645, "lon": -73.0885, "color": "#059669",
             "href": "tel:6382222", "phone": "638-2222", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y apoyo a víctimas de violencia.",
             "transporte": "Bus desde Bucaramanga · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Floridablanca",
             "dir": "Cra 9 #4-40, Floridablanca", "barrio": "Centro",
             "lat": 7.0635, "lon": -73.0895, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Bucaramanga · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Floridablanca",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.0640, "lon": -73.0890, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "girón": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Girón",
             "dir": "Cra 26 #12-30, Girón", "barrio": "Centro",
             "lat": 7.0740, "lon": -73.1680, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Girón. Denuncias y medidas de protección.",
             "transporte": "Bus desde Bucaramanga (15 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Juan de Dios Girón",
             "dir": "Calle 14 #24-50, Girón", "barrio": "Centro",
             "lat": 7.0745, "lon": -73.1685, "color": "#059669",
             "href": "tel:6466300", "phone": "646-6300", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal de Girón. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Bucaramanga · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Girón",
             "dir": "Cra 27 #11-40, Girón", "barrio": "Centro",
             "lat": 7.0735, "lon": -73.1675, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Bucaramanga · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Girón",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.0740, "lon": -73.1680, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "piedecuesta": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Piedecuesta",
             "dir": "Cra 10 #6-30, Piedecuesta", "barrio": "Centro",
             "lat": 6.9870, "lon": -73.0500, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Piedecuesta. Denuncias y medidas de protección.",
             "transporte": "Bus desde Bucaramanga (25 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Integrado San Juan de Dios Piedecuesta",
             "dir": "Calle 8 #11-50, Piedecuesta", "barrio": "Centro",
             "lat": 6.9875, "lon": -73.0505, "color": "#059669",
             "href": "tel:6488100", "phone": "648-8100", "horario": "24/7 Urgencias",
             "desc": "Hospital de Piedecuesta. Urgencias y atención a víctimas de violencia.",
             "transporte": "Bus desde Bucaramanga · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Piedecuesta",
             "dir": "Cra 11 #5-20, Piedecuesta", "barrio": "Centro",
             "lat": 6.9865, "lon": -73.0495, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Bucaramanga · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Piedecuesta",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.9870, "lon": -73.0500, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "barrancabermeja": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Barrancabermeja",
             "dir": "Cra 18 #34-20, Barrancabermeja", "barrio": "Centro",
             "lat": 7.0640, "lon": -73.8550, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Barrancabermeja. Denuncias y medidas de protección.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael Barrancabermeja",
             "dir": "Calle 43 #15-50, Barrancabermeja", "barrio": "Centro",
             "lat": 7.0645, "lon": -73.8560, "color": "#059669",
             "href": "tel:6202050", "phone": "620-2050", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Barrancabermeja",
             "dir": "Cra 17 #35-40, Barrancabermeja", "barrio": "Centro",
             "lat": 7.0635, "lon": -73.8545, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Barrancabermeja",
             "dir": "Cra 19 #33-30, Barrancabermeja", "barrio": "Centro",
             "lat": 7.0640, "lon": -73.8555, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Barrancabermeja",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.0640, "lon": -73.8550, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # BOLÍVAR
        # ══════════════════════════════════════════════════════════════════════
        "cartagena": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Cartagena",
             "dir": "Calle 34 #4-20, Cartagena", "barrio": "Getsemaní",
             "lat": 10.3910, "lon": -75.4790, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana de Cartagena. Denuncias y medidas de protección.",
             "transporte": "Bus: Av. Venezuela · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario del Caribe",
             "dir": "Cra 6 #36-100, Cartagena", "barrio": "Amberes",
             "lat": 10.3940, "lon": -75.4780, "color": "#059669",
             "href": "tel:6564477", "phone": "656-4477", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias, medicina forense y apoyo psicológico para víctimas.",
             "transporte": "Bus: Av. Crisanto Luque · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Cartagena 24h",
             "dir": "Calle 36 #8-50, Cartagena", "barrio": "San Diego",
             "lat": 10.3950, "lon": -75.4770, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas.",
             "transporte": "Taxi · Bus rutas centro histórico"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Cartagena",
             "dir": "Cra 7 #35-50, Cartagena", "barrio": "Centro",
             "lat": 10.3930, "lon": -75.4775, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar y apoyo psicosocial.",
             "transporte": "Taxi · Bus al centro histórico"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Cartagena",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.3920, "lon": -75.4785, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro para mujeres víctimas de violencia.",
             "transporte": "Llama al 155 — coordinan transporte seguro"},
        ],
        "magangué": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Magangué",
             "dir": "Cra 12 #16-30, Magangué", "barrio": "Centro",
             "lat": 9.2400, "lon": -74.7540, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Magangué. Denuncias y emergencias.",
             "transporte": "Bus o lancha desde Cartagena · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Juan de Dios Magangué",
             "dir": "Calle 15 #13-50, Magangué", "barrio": "Centro",
             "lat": 9.2405, "lon": -74.7545, "color": "#059669",
             "href": "tel:6870050", "phone": "687-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital de Magangué. Urgencias y atención a víctimas.",
             "transporte": "Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Magangué",
             "dir": "Cra 13 #14-40, Magangué", "barrio": "Centro",
             "lat": 9.2395, "lon": -74.7535, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Magangué",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 9.2400, "lon": -74.7540, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "turbaco": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Turbaco",
             "dir": "Cra 5 #7-20, Turbaco", "barrio": "Centro",
             "lat": 10.3290, "lon": -75.4150, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Turbaco. Denuncias y emergencias.",
             "transporte": "Bus desde Cartagena (20 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Local de Turbaco",
             "dir": "Calle 8 #4-50, Turbaco", "barrio": "Centro",
             "lat": 10.3295, "lon": -75.4155, "color": "#059669",
             "href": "tel:6602200", "phone": "660-2200", "horario": "24/7 Urgencias",
             "desc": "Hospital local. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Cartagena · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Turbaco",
             "dir": "Cra 6 #6-30, Turbaco", "barrio": "Centro",
             "lat": 10.3285, "lon": -75.4145, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Cartagena · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Turbaco",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.3290, "lon": -75.4150, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # NORTE DE SANTANDER
        # ══════════════════════════════════════════════════════════════════════
        "cúcuta": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Cúcuta",
             "dir": "Av. 4 #14-20, Cúcuta", "barrio": "Centro",
             "lat": 7.8940, "lon": -72.5080, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana de Cúcuta. Denuncias y medidas de protección.",
             "transporte": "Bus: Av. 4 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Erasmo Meoz",
             "dir": "Av. 11E #1E-33, Cúcuta", "barrio": "Comuneros",
             "lat": 7.8970, "lon": -72.5090, "color": "#059669",
             "href": "tel:5783400", "phone": "578-3400", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias, medicina forense y atención a víctimas.",
             "transporte": "Bus: Av. 11E · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Cúcuta 24h",
             "dir": "Calle 13 #5-30, Cúcuta", "barrio": "Centro",
             "lat": 7.8935, "lon": -72.5070, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas.",
             "transporte": "Bus: Centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Cúcuta",
             "dir": "Cra 3 #15-40, Cúcuta", "barrio": "Centro",
             "lat": 7.8930, "lon": -72.5085, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Cúcuta",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.8940, "lon": -72.5080, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "ocaña": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Ocaña",
             "dir": "Cra 12 #7-30, Ocaña", "barrio": "Centro",
             "lat": 8.2390, "lon": -73.3570, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Ocaña. Denuncias y medidas de protección.",
             "transporte": "Bus desde Cúcuta (3h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Emiro Quintero Cañizares",
             "dir": "Calle 9 #11-50, Ocaña", "barrio": "Centro",
             "lat": 8.2395, "lon": -73.3575, "color": "#059669",
             "href": "tel:5628000", "phone": "562-8000", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Ocaña",
             "dir": "Cra 13 #6-40, Ocaña", "barrio": "Centro",
             "lat": 8.2385, "lon": -73.3565, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Ocaña",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 8.2390, "lon": -73.3570, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "villa del rosario": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Villa del Rosario",
             "dir": "Cra 8 #4-20, Villa del Rosario", "barrio": "Centro",
             "lat": 7.8360, "lon": -72.4700, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Villa del Rosario. Denuncias y emergencias.",
             "transporte": "Bus desde Cúcuta (15 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael Villa del Rosario",
             "dir": "Calle 5 #9-50, Villa del Rosario", "barrio": "Centro",
             "lat": 7.8365, "lon": -72.4705, "color": "#059669",
             "href": "tel:5754400", "phone": "575-4400", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y atención a víctimas de violencia.",
             "transporte": "Bus desde Cúcuta · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Villa del Rosario",
             "dir": "Cra 9 #3-30, Villa del Rosario", "barrio": "Centro",
             "lat": 7.8355, "lon": -72.4695, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Cúcuta · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Villa del Rosario",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.8360, "lon": -72.4700, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "los patios": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Los Patios",
             "dir": "Cra 6 #5-20, Los Patios", "barrio": "Centro",
             "lat": 7.9200, "lon": -72.4940, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Los Patios. Denuncias y emergencias.",
             "transporte": "Bus desde Cúcuta (10 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Juan de Dios Los Patios",
             "dir": "Calle 6 #7-40, Los Patios", "barrio": "Centro",
             "lat": 7.9205, "lon": -72.4945, "color": "#059669",
             "href": "tel:5744800", "phone": "574-4800", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Cúcuta · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Los Patios",
             "dir": "Cra 7 #4-30, Los Patios", "barrio": "Centro",
             "lat": 7.9195, "lon": -72.4935, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Cúcuta · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Los Patios",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.9200, "lon": -72.4940, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # RISARALDA
        # ══════════════════════════════════════════════════════════════════════
        "pereira": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Pereira",
             "dir": "Cra 8 #20-35, Pereira", "barrio": "Centro",
             "lat": 4.8140, "lon": -75.6960, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía Metropolitana de Pereira. Denuncias y emergencias.",
             "transporte": "Megabús: Estación Centro (5 min caminando) · Bus: múltiples rutas"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario San Jorge",
             "dir": "Cra 10 #18-16, Pereira", "barrio": "El Centro",
             "lat": 4.8145, "lon": -75.6950, "color": "#059669",
             "href": "tel:3335800", "phone": "333-5800", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias, medicina forense y apoyo psicológico.",
             "transporte": "Megabús: Centro · Bus: Cra 10"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Pereira 24h",
             "dir": "Calle 19 #9-30, Pereira", "barrio": "El Centro",
             "lat": 4.8150, "lon": -75.6970, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas.",
             "transporte": "Megabús: Centro · Bus: Cll 19"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Pereira",
             "dir": "Cra 9 #21-50, Pereira", "barrio": "El Centro",
             "lat": 4.8148, "lon": -75.6965, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar y apoyo psicosocial.",
             "transporte": "Megabús · Bus rutas centro"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Pereira",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.8140, "lon": -75.6960, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro para mujeres víctimas de violencia.",
             "transporte": "Llama al 155 — coordinan transporte seguro"},
        ],
        "dosquebradas": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Dosquebradas",
             "dir": "Cra 18 #25-10, Dosquebradas", "barrio": "Centro",
             "lat": 4.8370, "lon": -75.6710, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Dosquebradas. Denuncias y emergencias.",
             "transporte": "Megabús conexión Pereira–Dosquebradas · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Santa Mónica Dosquebradas",
             "dir": "Calle 22 #20-50, Dosquebradas", "barrio": "Santa Mónica",
             "lat": 4.8375, "lon": -75.6705, "color": "#059669",
             "href": "tel:3306050", "phone": "330-6050", "horario": "24/7 Urgencias",
             "desc": "Hospital de Dosquebradas. Urgencias y apoyo a víctimas de violencia.",
             "transporte": "Bus desde Pereira · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Dosquebradas",
             "dir": "Cra 19 #23-30, Dosquebradas", "barrio": "Centro",
             "lat": 4.8365, "lon": -75.6715, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Pereira · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Dosquebradas",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.8370, "lon": -75.6710, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "santa rosa de cabal": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Santa Rosa de Cabal",
             "dir": "Cra 14 #12-20, Santa Rosa de Cabal", "barrio": "Centro",
             "lat": 4.8690, "lon": -75.6200, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Santa Rosa de Cabal. Denuncias y medidas de protección.",
             "transporte": "Bus desde Pereira (20 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Vicente de Paul Santa Rosa",
             "dir": "Calle 13 #15-50, Santa Rosa de Cabal", "barrio": "Centro",
             "lat": 4.8695, "lon": -75.6205, "color": "#059669",
             "href": "tel:3681050", "phone": "368-1050", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y atención a víctimas de violencia.",
             "transporte": "Bus desde Pereira · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Santa Rosa de Cabal",
             "dir": "Cra 15 #11-30, Santa Rosa de Cabal", "barrio": "Centro",
             "lat": 4.8685, "lon": -75.6195, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Pereira · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Santa Rosa de Cabal",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.8690, "lon": -75.6200, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "la virginia": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía La Virginia",
             "dir": "Cra 8 #7-30, La Virginia", "barrio": "Centro",
             "lat": 4.9010, "lon": -75.8820, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de La Virginia. Denuncias y emergencias.",
             "transporte": "Bus desde Pereira (30 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Pedro y San Pablo La Virginia",
             "dir": "Calle 8 #9-50, La Virginia", "barrio": "Centro",
             "lat": 4.9015, "lon": -75.8825, "color": "#059669",
             "href": "tel:3686000", "phone": "368-6000", "horario": "24/7 Urgencias",
             "desc": "Hospital municipal. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Pereira · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia La Virginia",
             "dir": "Cra 9 #6-20, La Virginia", "barrio": "Centro",
             "lat": 4.9005, "lon": -75.8815, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Pereira · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer La Virginia",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.9010, "lon": -75.8820, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CALDAS
        # ══════════════════════════════════════════════════════════════════════
        "manizales": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Manizales",
             "dir": "Cra 23 #22-10, Manizales", "barrio": "El Centro",
             "lat": 5.0700, "lon": -75.5200, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Manizales. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 23 (rutas directas) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital de Caldas",
             "dir": "Calle 23 #35-50, Manizales", "barrio": "El Centro",
             "lat": 5.0710, "lon": -75.5210, "color": "#059669",
             "href": "tel:8846560", "phone": "884-6560", "horario": "24/7 Urgencias",
             "desc": "Hospital de Caldas. Urgencias, medicina forense y atención a víctimas.",
             "transporte": "Bus: Cll 23 · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Manizales",
             "dir": "Cra 22 #24-30, Manizales", "barrio": "El Centro",
             "lat": 5.0705, "lon": -75.5195, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Manizales",
             "dir": "Cra 24 #21-50, Manizales", "barrio": "El Centro",
             "lat": 5.0695, "lon": -75.5205, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Manizales",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.0700, "lon": -75.5200, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "la dorada": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía La Dorada",
             "dir": "Cra 4 #15-20, La Dorada", "barrio": "Centro",
             "lat": 5.4530, "lon": -74.6660, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de La Dorada. Denuncias y emergencias.",
             "transporte": "Bus desde Manizales (2h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental Mario Correa Rengifo",
             "dir": "Calle 14 #5-50, La Dorada", "barrio": "Centro",
             "lat": 5.4535, "lon": -74.6665, "color": "#059669",
             "href": "tel:8854000", "phone": "885-4000", "horario": "24/7 Urgencias",
             "desc": "Hospital de La Dorada. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia La Dorada",
             "dir": "Cra 5 #13-40, La Dorada", "barrio": "Centro",
             "lat": 5.4525, "lon": -74.6655, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer La Dorada",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.4530, "lon": -74.6660, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # QUINDÍO
        # ══════════════════════════════════════════════════════════════════════
        "armenia": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Armenia",
             "dir": "Cra 14 #20-30, Armenia", "barrio": "El Centro",
             "lat": 4.5340, "lon": -75.6810, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Armenia. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 14 (rutas directas) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental San Juan de Dios",
             "dir": "Calle 19 #16-50, Armenia", "barrio": "El Centro",
             "lat": 4.5345, "lon": -75.6820, "color": "#059669",
             "href": "tel:7491700", "phone": "749-1700", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Armenia",
             "dir": "Cra 13 #21-40, Armenia", "barrio": "El Centro",
             "lat": 4.5335, "lon": -75.6805, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Armenia",
             "dir": "Cra 15 #19-20, Armenia", "barrio": "El Centro",
             "lat": 4.5338, "lon": -75.6815, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección y apoyo psicosocial.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Armenia",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.5340, "lon": -75.6810, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "calarcá": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Calarcá",
             "dir": "Cra 25 #20-30, Calarcá", "barrio": "Centro",
             "lat": 4.5240, "lon": -75.6440, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Calarcá. Denuncias y medidas de protección.",
             "transporte": "Bus desde Armenia (20 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital La Misericordia Calarcá",
             "dir": "Calle 19 #27-50, Calarcá", "barrio": "Centro",
             "lat": 4.5245, "lon": -75.6445, "color": "#059669",
             "href": "tel:7420050", "phone": "742-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital de Calarcá. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Armenia · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Calarcá",
             "dir": "Cra 26 #19-20, Calarcá", "barrio": "Centro",
             "lat": 4.5235, "lon": -75.6435, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Armenia · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Calarcá",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.5240, "lon": -75.6440, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # TOLIMA
        # ══════════════════════════════════════════════════════════════════════
        "ibagué": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Ibagué",
             "dir": "Cra 3 #10-20, Ibagué", "barrio": "El Centro",
             "lat": 4.4380, "lon": -75.2320, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Ibagué. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 3 (rutas directas) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Federico Lleras Acosta",
             "dir": "Cra 4 #20-50, Ibagué", "barrio": "El Centro",
             "lat": 4.4385, "lon": -75.2325, "color": "#059669",
             "href": "tel:2617200", "phone": "261-7200", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias, medicina forense y atención a víctimas.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Ibagué",
             "dir": "Calle 10 #3-50, Ibagué", "barrio": "El Centro",
             "lat": 4.4375, "lon": -75.2315, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Ibagué",
             "dir": "Cra 5 #9-30, Ibagué", "barrio": "El Centro",
             "lat": 4.4382, "lon": -75.2318, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Ibagué",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.4380, "lon": -75.2320, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "espinal": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Espinal",
             "dir": "Cra 6 #8-20, Espinal", "barrio": "Centro",
             "lat": 4.1530, "lon": -74.8860, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Espinal. Denuncias y medidas de protección.",
             "transporte": "Bus desde Ibagué (45 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael Espinal",
             "dir": "Calle 7 #5-50, Espinal", "barrio": "Centro",
             "lat": 4.1535, "lon": -74.8865, "color": "#059669",
             "href": "tel:2485000", "phone": "248-5000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Espinal. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Ibagué · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Espinal",
             "dir": "Cra 7 #7-30, Espinal", "barrio": "Centro",
             "lat": 4.1525, "lon": -74.8855, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Ibagué · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Espinal",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.1530, "lon": -74.8860, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "melgar": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Melgar",
             "dir": "Cra 8 #5-30, Melgar", "barrio": "Centro",
             "lat": 4.2100, "lon": -74.6370, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Melgar. Denuncias y emergencias.",
             "transporte": "Bus desde Bogotá (2.5h) o Ibagué (1h) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Antonio Melgar",
             "dir": "Calle 6 #9-50, Melgar", "barrio": "Centro",
             "lat": 4.2105, "lon": -74.6375, "color": "#059669",
             "href": "tel:2452020", "phone": "245-2020", "horario": "24/7 Urgencias",
             "desc": "Hospital de Melgar. Urgencias y atención a víctimas.",
             "transporte": "Bus o taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Melgar",
             "dir": "Cra 9 #4-20, Melgar", "barrio": "Centro",
             "lat": 4.2095, "lon": -74.6365, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus o taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Melgar",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.2100, "lon": -74.6370, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # NARIÑO
        # ══════════════════════════════════════════════════════════════════════
        "pasto": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Pasto",
             "dir": "Cra 27 #16-40, Pasto", "barrio": "El Centro",
             "lat": 1.2140, "lon": -77.2800, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Pasto. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 27 (rutas directas) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario Departamental",
             "dir": "Calle 22 #22-35, Pasto", "barrio": "El Centro",
             "lat": 1.2145, "lon": -77.2810, "color": "#059669",
             "href": "tel:7233040", "phone": "723-3040", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Pasto",
             "dir": "Cra 28 #15-60, Pasto", "barrio": "El Centro",
             "lat": 1.2135, "lon": -77.2795, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Pasto",
             "dir": "Calle 18 #24-20, Pasto", "barrio": "El Centro",
             "lat": 1.2138, "lon": -77.2805, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Pasto",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 1.2140, "lon": -77.2800, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "tumaco": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Tumaco",
             "dir": "Cra 4 #8-30, Tumaco", "barrio": "El Centro",
             "lat": 1.8090, "lon": -78.8080, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Tumaco. Denuncias y emergencias.",
             "transporte": "Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Andrés de Tumaco",
             "dir": "Calle 9 #3-50, Tumaco", "barrio": "El Centro",
             "lat": 1.8095, "lon": -78.8075, "color": "#059669",
             "href": "tel:7272800", "phone": "727-2800", "horario": "24/7 Urgencias",
             "desc": "Hospital de Tumaco. Urgencias y atención a víctimas.",
             "transporte": "Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Tumaco",
             "dir": "Cra 5 #7-20, Tumaco", "barrio": "El Centro",
             "lat": 1.8085, "lon": -78.8085, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Tumaco",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 1.8090, "lon": -78.8080, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "ipiales": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Ipiales",
             "dir": "Cra 6 #14-20, Ipiales", "barrio": "Centro",
             "lat": 0.8280, "lon": -77.6440, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Ipiales. Denuncias y medidas de protección.",
             "transporte": "Bus desde Pasto (2h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Civil de Ipiales",
             "dir": "Calle 13 #5-50, Ipiales", "barrio": "Centro",
             "lat": 0.8285, "lon": -77.6445, "color": "#059669",
             "href": "tel:7733000", "phone": "773-3000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Ipiales. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Pasto · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Ipiales",
             "dir": "Cra 7 #12-30, Ipiales", "barrio": "Centro",
             "lat": 0.8275, "lon": -77.6435, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Pasto · Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Ipiales",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 0.8280, "lon": -77.6440, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # HUILA
        # ══════════════════════════════════════════════════════════════════════
        "neiva": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Neiva",
             "dir": "Cra 5 #7-20, Neiva", "barrio": "El Centro",
             "lat": 2.9350, "lon": -75.2820, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Neiva. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 5 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Hernando Moncaleano",
             "dir": "Calle 9 #8-50, Neiva", "barrio": "El Centro",
             "lat": 2.9355, "lon": -75.2825, "color": "#059669",
             "href": "tel:8714400", "phone": "871-4400", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Neiva",
             "dir": "Cra 4 #8-30, Neiva", "barrio": "El Centro",
             "lat": 2.9345, "lon": -75.2815, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Neiva",
             "dir": "Cra 6 #6-40, Neiva", "barrio": "El Centro",
             "lat": 2.9352, "lon": -75.2818, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Neiva",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 2.9350, "lon": -75.2820, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "pitalito": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Pitalito",
             "dir": "Cra 4 #5-20, Pitalito", "barrio": "Centro",
             "lat": 1.8550, "lon": -76.0540, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Pitalito. Denuncias y medidas de protección.",
             "transporte": "Bus desde Neiva (3h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental San Antonio Pitalito",
             "dir": "Calle 6 #3-50, Pitalito", "barrio": "Centro",
             "lat": 1.8555, "lon": -76.0545, "color": "#059669",
             "href": "tel:8360050", "phone": "836-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Neiva · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Pitalito",
             "dir": "Cra 5 #4-30, Pitalito", "barrio": "Centro",
             "lat": 1.8545, "lon": -76.0535, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Neiva · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Pitalito",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 1.8550, "lon": -76.0540, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # META
        # ══════════════════════════════════════════════════════════════════════
        "villavicencio": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Villavicencio",
             "dir": "Cra 34 #36-50, Villavicencio", "barrio": "El Centro",
             "lat": 4.1420, "lon": -73.6260, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Villavicencio. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 34 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental de Villavicencio",
             "dir": "Calle 38 #32-60, Villavicencio", "barrio": "Barzal",
             "lat": 4.1425, "lon": -73.6265, "color": "#059669",
             "href": "tel:6629700", "phone": "662-9700", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Villavicencio",
             "dir": "Cra 33 #37-40, Villavicencio", "barrio": "El Centro",
             "lat": 4.1415, "lon": -73.6255, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Villavicencio",
             "dir": "Cra 35 #35-20, Villavicencio", "barrio": "El Centro",
             "lat": 4.1418, "lon": -73.6258, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Villavicencio",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 4.1420, "lon": -73.6260, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "acacías": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Acacías",
             "dir": "Cra 15 #12-20, Acacías", "barrio": "Centro",
             "lat": 3.9880, "lon": -73.7590, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Acacías. Denuncias y emergencias.",
             "transporte": "Bus desde Villavicencio (30 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Cristóbal Acacías",
             "dir": "Calle 13 #16-50, Acacías", "barrio": "Centro",
             "lat": 3.9885, "lon": -73.7595, "color": "#059669",
             "href": "tel:6566000", "phone": "656-6000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Acacías. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Villavicencio · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Acacías",
             "dir": "Cra 16 #11-30, Acacías", "barrio": "Centro",
             "lat": 3.9875, "lon": -73.7585, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Villavicencio · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Acacías",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.9880, "lon": -73.7590, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # MAGDALENA
        # ══════════════════════════════════════════════════════════════════════
        "santa marta": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Santa Marta",
             "dir": "Cra 3 #16-40, Santa Marta", "barrio": "El Centro",
             "lat": 11.2400, "lon": -74.2000, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Santa Marta. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 3 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Central Julio Méndez Barreneche",
             "dir": "Calle 30 #3-50, Santa Marta", "barrio": "El Prado",
             "lat": 11.2410, "lon": -74.2010, "color": "#059669",
             "href": "tel:4310660", "phone": "431-0660", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Santa Marta",
             "dir": "Cra 4 #17-30, Santa Marta", "barrio": "El Centro",
             "lat": 11.2395, "lon": -74.1995, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Santa Marta",
             "dir": "Cra 5 #15-20, Santa Marta", "barrio": "El Centro",
             "lat": 11.2398, "lon": -74.2005, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Santa Marta",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 11.2400, "lon": -74.2000, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "ciénaga": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Ciénaga",
             "dir": "Cra 8 #12-30, Ciénaga", "barrio": "Centro",
             "lat": 11.0050, "lon": -74.2510, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Ciénaga. Denuncias y emergencias.",
             "transporte": "Bus desde Santa Marta (30 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Cristóbal Ciénaga",
             "dir": "Calle 11 #9-50, Ciénaga", "barrio": "Centro",
             "lat": 11.0055, "lon": -74.2515, "color": "#059669",
             "href": "tel:4209000", "phone": "420-9000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Ciénaga. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Santa Marta · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Ciénaga",
             "dir": "Cra 9 #11-20, Ciénaga", "barrio": "Centro",
             "lat": 11.0045, "lon": -74.2505, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Santa Marta · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Ciénaga",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 11.0050, "lon": -74.2510, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CÓRDOBA
        # ══════════════════════════════════════════════════════════════════════
        "montería": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Montería",
             "dir": "Cra 6 #29-30, Montería", "barrio": "El Centro",
             "lat": 8.7570, "lon": -75.8800, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Montería. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 6 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Jerónimo de Montería",
             "dir": "Calle 25 #4-80, Montería", "barrio": "Montería 2000",
             "lat": 8.7575, "lon": -75.8810, "color": "#059669",
             "href": "tel:7895870", "phone": "789-5870", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Montería",
             "dir": "Cra 5 #28-50, Montería", "barrio": "El Centro",
             "lat": 8.7565, "lon": -75.8795, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Montería",
             "dir": "Cra 7 #27-40, Montería", "barrio": "El Centro",
             "lat": 8.7568, "lon": -75.8805, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Montería",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 8.7570, "lon": -75.8800, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "cereté": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Cereté",
             "dir": "Cra 10 #8-20, Cereté", "barrio": "Centro",
             "lat": 8.8820, "lon": -75.7920, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Cereté. Denuncias y emergencias.",
             "transporte": "Bus desde Montería (20 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Vicente de Paul Cereté",
             "dir": "Calle 9 #11-50, Cereté", "barrio": "Centro",
             "lat": 8.8825, "lon": -75.7925, "color": "#059669",
             "href": "tel:7878000", "phone": "787-8000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Cereté. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Montería · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Cereté",
             "dir": "Cra 11 #7-30, Cereté", "barrio": "Centro",
             "lat": 8.8815, "lon": -75.7915, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Montería · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Cereté",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 8.8820, "lon": -75.7920, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # SUCRE
        # ══════════════════════════════════════════════════════════════════════
        "sincelejo": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Sincelejo",
             "dir": "Cra 20 #25-30, Sincelejo", "barrio": "El Centro",
             "lat": 9.3050, "lon": -75.3980, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Sincelejo. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 20 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario de Sincelejo",
             "dir": "Calle 23 #21-50, Sincelejo", "barrio": "El Centro",
             "lat": 9.3055, "lon": -75.3985, "color": "#059669",
             "href": "tel:2823232", "phone": "282-3232", "horario": "24/7 Urgencias",
             "desc": "Hospital de Sincelejo. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Sincelejo",
             "dir": "Cra 21 #24-20, Sincelejo", "barrio": "El Centro",
             "lat": 9.3045, "lon": -75.3975, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Sincelejo",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 9.3050, "lon": -75.3980, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "corozal": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Corozal",
             "dir": "Cra 18 #20-30, Corozal", "barrio": "Centro",
             "lat": 9.3180, "lon": -75.2920, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Corozal. Denuncias y emergencias.",
             "transporte": "Bus desde Sincelejo (30 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Regional San Francisco de Asís Corozal",
             "dir": "Calle 19 #19-50, Corozal", "barrio": "Centro",
             "lat": 9.3185, "lon": -75.2925, "color": "#059669",
             "href": "tel:2860050", "phone": "286-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital de Corozal. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Sincelejo · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Corozal",
             "dir": "Cra 19 #21-20, Corozal", "barrio": "Centro",
             "lat": 9.3175, "lon": -75.2915, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Sincelejo · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Corozal",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 9.3180, "lon": -75.2920, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CESAR
        # ══════════════════════════════════════════════════════════════════════
        "valledupar": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Valledupar",
             "dir": "Cra 7 #16-30, Valledupar", "barrio": "El Centro",
             "lat": 10.4770, "lon": -73.2500, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Valledupar. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 7 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Eduardo Arredondo Daza",
             "dir": "Calle 18 #8-50, Valledupar", "barrio": "El Centro",
             "lat": 10.4775, "lon": -73.2505, "color": "#059669",
             "href": "tel:5740403", "phone": "574-0403", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Valledupar",
             "dir": "Cra 8 #15-40, Valledupar", "barrio": "El Centro",
             "lat": 10.4765, "lon": -73.2495, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Valledupar",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 10.4770, "lon": -73.2500, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "aguachica": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Aguachica",
             "dir": "Cra 12 #8-20, Aguachica", "barrio": "Centro",
             "lat": 8.3070, "lon": -73.6190, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Aguachica. Denuncias y emergencias.",
             "transporte": "Bus desde Valledupar (3h) o Bucaramanga (4h) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Regional Noroccidental Aguachica",
             "dir": "Calle 9 #13-50, Aguachica", "barrio": "Centro",
             "lat": 8.3075, "lon": -73.6195, "color": "#059669",
             "href": "tel:5671000", "phone": "567-1000", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Aguachica",
             "dir": "Cra 13 #7-30, Aguachica", "barrio": "Centro",
             "lat": 8.3065, "lon": -73.6185, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Aguachica",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 8.3070, "lon": -73.6190, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # BOYACÁ
        # ══════════════════════════════════════════════════════════════════════
        "tunja": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Tunja",
             "dir": "Cra 10 #20-30, Tunja", "barrio": "El Centro",
             "lat": 5.5350, "lon": -73.3680, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Tunja. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 10 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Regional de Boyacá",
             "dir": "Calle 18 #11-22, Tunja", "barrio": "El Centro",
             "lat": 5.5355, "lon": -73.3685, "color": "#059669",
             "href": "tel:7426666", "phone": "742-6666", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Tunja",
             "dir": "Cra 9 #21-50, Tunja", "barrio": "El Centro",
             "lat": 5.5345, "lon": -73.3675, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Tunja",
             "dir": "Cra 11 #19-40, Tunja", "barrio": "El Centro",
             "lat": 5.5348, "lon": -73.3678, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Tunja",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.5350, "lon": -73.3680, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "duitama": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Duitama",
             "dir": "Cra 18 #14-20, Duitama", "barrio": "Centro",
             "lat": 5.8270, "lon": -73.0320, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Duitama. Denuncias y medidas de protección.",
             "transporte": "Bus desde Tunja (1h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Regional de Duitama",
             "dir": "Cra 20 #12-50, Duitama", "barrio": "Centro",
             "lat": 5.8275, "lon": -73.0325, "color": "#059669",
             "href": "tel:7600050", "phone": "760-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Tunja · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Duitama",
             "dir": "Cra 19 #13-30, Duitama", "barrio": "Centro",
             "lat": 5.8265, "lon": -73.0315, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Tunja · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Duitama",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.8270, "lon": -73.0320, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "sogamoso": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Sogamoso",
             "dir": "Cra 11 #10-30, Sogamoso", "barrio": "Centro",
             "lat": 5.7140, "lon": -72.9310, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Sogamoso. Denuncias y emergencias.",
             "transporte": "Bus desde Tunja (1.5h) · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Regional de Sogamoso",
             "dir": "Cra 13 #8-50, Sogamoso", "barrio": "Centro",
             "lat": 5.7145, "lon": -72.9315, "color": "#059669",
             "href": "tel:7702000", "phone": "770-2000", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Tunja · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Sogamoso",
             "dir": "Cra 12 #9-20, Sogamoso", "barrio": "Centro",
             "lat": 5.7135, "lon": -72.9305, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Tunja · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Sogamoso",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.7140, "lon": -72.9310, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CAUCA
        # ══════════════════════════════════════════════════════════════════════
        "popayán": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Popayán",
             "dir": "Cra 7 #5-30, Popayán", "barrio": "El Centro",
             "lat": 2.4410, "lon": -76.6060, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Popayán. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 7 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Universitario San José",
             "dir": "Cra 6 #8-60, Popayán", "barrio": "El Centro",
             "lat": 2.4415, "lon": -76.6065, "color": "#059669",
             "href": "tel:8241000", "phone": "824-1000", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Popayán",
             "dir": "Cra 8 #4-50, Popayán", "barrio": "El Centro",
             "lat": 2.4405, "lon": -76.6055, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Popayán",
             "dir": "Cra 7 #7-20, Popayán", "barrio": "El Centro",
             "lat": 2.4408, "lon": -76.6058, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Popayán",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 2.4410, "lon": -76.6060, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "santander de quilichao": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Santander de Quilichao",
             "dir": "Cra 10 #6-30, Santander de Quilichao", "barrio": "Centro",
             "lat": 3.0120, "lon": -76.4840, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Santander de Quilichao. Denuncias y emergencias.",
             "transporte": "Bus desde Cali (1h) o Popayán (1h) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Susana López de Valencia",
             "dir": "Cra 11 #5-50, Santander de Quilichao", "barrio": "Centro",
             "lat": 3.0125, "lon": -76.4845, "color": "#059669",
             "href": "tel:8278000", "phone": "827-8000", "horario": "24/7 Urgencias",
             "desc": "Hospital de mediana complejidad. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Cali o Popayán · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Santander de Quilichao",
             "dir": "Cra 12 #4-20, Santander de Quilichao", "barrio": "Centro",
             "lat": 3.0115, "lon": -76.4835, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Santander de Quilichao",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.0120, "lon": -76.4840, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # LA GUAJIRA
        # ══════════════════════════════════════════════════════════════════════
        "riohacha": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación Policía Centro Riohacha",
             "dir": "Cra 7 #3-20, Riohacha", "barrio": "El Centro",
             "lat": 11.5440, "lon": -72.9080, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Riohacha. Denuncias y medidas de protección.",
             "transporte": "Bus: Cra 7 (rutas directas) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Nuestra Señora de los Remedios",
             "dir": "Cra 8 #10-50, Riohacha", "barrio": "El Centro",
             "lat": 11.5445, "lon": -72.9085, "color": "#059669",
             "href": "tel:7272222", "phone": "727-2222", "horario": "24/7 Urgencias",
             "desc": "Hospital de Riohacha. Urgencias y atención a víctimas.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Riohacha",
             "dir": "Cra 9 #9-30, Riohacha", "barrio": "El Centro",
             "lat": 11.5435, "lon": -72.9075, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus al centro · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Riohacha",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 11.5440, "lon": -72.9080, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "maicao": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Maicao",
             "dir": "Cra 10 #14-20, Maicao", "barrio": "Centro",
             "lat": 11.3780, "lon": -72.2430, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Maicao. Denuncias y emergencias.",
             "transporte": "Bus desde Riohacha (1.5h) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San José Maicao",
             "dir": "Calle 13 #11-50, Maicao", "barrio": "Centro",
             "lat": 11.3785, "lon": -72.2435, "color": "#059669",
             "href": "tel:7266000", "phone": "726-6000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Maicao. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Riohacha · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Maicao",
             "dir": "Cra 11 #12-30, Maicao", "barrio": "Centro",
             "lat": 11.3775, "lon": -72.2425, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Riohacha · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Maicao",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 11.3780, "lon": -72.2430, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CASANARE
        # ══════════════════════════════════════════════════════════════════════
        "yopal": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Yopal",
             "dir": "Cra 22 #16-30, Yopal", "barrio": "Centro",
             "lat": 5.3380, "lon": -72.3950, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Yopal. Denuncias y medidas de protección.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Regional de Yopal",
             "dir": "Calle 17 #24-50, Yopal", "barrio": "Centro",
             "lat": 5.3385, "lon": -72.3955, "color": "#059669",
             "href": "tel:6349000", "phone": "634-9000", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Yopal",
             "dir": "Cra 23 #15-40, Yopal", "barrio": "Centro",
             "lat": 5.3375, "lon": -72.3945, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Yopal",
             "dir": "Cra 24 #14-20, Yopal", "barrio": "Centro",
             "lat": 5.3378, "lon": -72.3948, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Yopal",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.3380, "lon": -72.3950, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "aguazul": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Aguazul",
             "dir": "Cra 18 #10-20, Aguazul", "barrio": "Centro",
             "lat": 5.1700, "lon": -72.5510, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Aguazul. Denuncias y emergencias.",
             "transporte": "Bus desde Yopal (30 min) · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Municipal de Aguazul",
             "dir": "Calle 11 #19-50, Aguazul", "barrio": "Centro",
             "lat": 5.1705, "lon": -72.5515, "color": "#059669",
             "href": "tel:6356000", "phone": "635-6000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Aguazul. Urgencias y atención a víctimas.",
             "transporte": "Bus desde Yopal · Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Aguazul",
             "dir": "Cra 19 #9-30, Aguazul", "barrio": "Centro",
             "lat": 5.1695, "lon": -72.5505, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus desde Yopal · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Aguazul",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.1700, "lon": -72.5510, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # ARAUCA
        # ══════════════════════════════════════════════════════════════════════
        "arauca": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Arauca",
             "dir": "Cra 20 #19-30, Arauca", "barrio": "Centro",
             "lat": 7.0900, "lon": -70.7620, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Arauca. Denuncias y medidas de protección.",
             "transporte": "Taxi local · Bus urbano"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Juan de Dios Arauca",
             "dir": "Calle 18 #21-50, Arauca", "barrio": "Centro",
             "lat": 7.0905, "lon": -70.7625, "color": "#059669",
             "href": "tel:8856000", "phone": "885-6000", "horario": "24/7 Urgencias",
             "desc": "Hospital departamental. Urgencias y atención a víctimas de violencia.",
             "transporte": "Taxi local"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Arauca",
             "dir": "Cra 21 #17-40, Arauca", "barrio": "Centro",
             "lat": 7.0895, "lon": -70.7615, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Arauca",
             "dir": "Cra 22 #16-20, Arauca", "barrio": "Centro",
             "lat": 7.0898, "lon": -70.7618, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Arauca",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 7.0900, "lon": -70.7620, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "saravena": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Saravena",
             "dir": "Cra 14 #6-20, Saravena", "barrio": "Centro",
             "lat": 6.9560, "lon": -71.8620, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Saravena. Denuncias y emergencias.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Antonio Saravena",
             "dir": "Calle 7 #15-50, Saravena", "barrio": "Centro",
             "lat": 6.9565, "lon": -71.8625, "color": "#059669",
             "href": "tel:8862000", "phone": "886-2000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Saravena. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Saravena",
             "dir": "Cra 15 #5-30, Saravena", "barrio": "Centro",
             "lat": 6.9555, "lon": -71.8615, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Saravena",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.9560, "lon": -71.8620, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # PUTUMAYO
        # ══════════════════════════════════════════════════════════════════════
        "mocoa": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Mocoa",
             "dir": "Cra 10 #8-20, Mocoa", "barrio": "Centro",
             "lat": 1.1480, "lon": -76.6480, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Mocoa. Denuncias y medidas de protección.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental de Mocoa",
             "dir": "Calle 9 #11-50, Mocoa", "barrio": "Centro",
             "lat": 1.1485, "lon": -76.6485, "color": "#059669",
             "href": "tel:4206000", "phone": "420-6000", "horario": "24/7 Urgencias",
             "desc": "Hospital de Mocoa. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Mocoa",
             "dir": "Cra 11 #7-30, Mocoa", "barrio": "Centro",
             "lat": 1.1475, "lon": -76.6475, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Mocoa",
             "dir": "Cra 12 #6-20, Mocoa", "barrio": "Centro",
             "lat": 1.1478, "lon": -76.6478, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Mocoa",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 1.1480, "lon": -76.6480, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
        "puerto asís": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Puerto Asís",
             "dir": "Cra 22 #8-20, Puerto Asís", "barrio": "Centro",
             "lat": 0.5060, "lon": -76.4980, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Puerto Asís. Denuncias y emergencias.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Porfirio Rodríguez Puerto Asís",
             "dir": "Calle 9 #23-50, Puerto Asís", "barrio": "Centro",
             "lat": 0.5065, "lon": -76.4985, "color": "#059669",
             "href": "tel:4350050", "phone": "435-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital de Puerto Asís. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Puerto Asís",
             "dir": "Cra 23 #7-30, Puerto Asís", "barrio": "Centro",
             "lat": 0.5055, "lon": -76.4975, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Puerto Asís",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 0.5060, "lon": -76.4980, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CHOCÓ
        # ══════════════════════════════════════════════════════════════════════
        "quibdó": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Quibdó",
             "dir": "Cra 2 #24-30, Quibdó", "barrio": "Centro",
             "lat": 5.6940, "lon": -76.6570, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Quibdó. Denuncias y medidas de protección.",
             "transporte": "Bus local · Taxi · Lancha fluvial"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Francisco de Asís Quibdó",
             "dir": "Calle 23 #3-50, Quibdó", "barrio": "Centro",
             "lat": 5.6945, "lon": -76.6575, "color": "#059669",
             "href": "tel:6727000", "phone": "672-7000", "horario": "24/7 Urgencias",
             "desc": "Hospital departamental. Urgencias y atención a víctimas.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Quibdó",
             "dir": "Cra 3 #22-40, Quibdó", "barrio": "Centro",
             "lat": 5.6935, "lon": -76.6565, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Quibdó",
             "dir": "Cra 4 #21-20, Quibdó", "barrio": "Centro",
             "lat": 5.6938, "lon": -76.6568, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Quibdó",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 5.6940, "lon": -76.6570, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # CAQUETÁ
        # ══════════════════════════════════════════════════════════════════════
        "florencia": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Florencia",
             "dir": "Cra 10 #14-30, Florencia", "barrio": "Centro",
             "lat": 1.6160, "lon": -75.6080, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Florencia. Denuncias y medidas de protección.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital María Inmaculada Florencia",
             "dir": "Calle 13 #11-50, Florencia", "barrio": "Centro",
             "lat": 1.6165, "lon": -75.6085, "color": "#059669",
             "href": "tel:4344444", "phone": "434-4444", "horario": "24/7 Urgencias",
             "desc": "Hospital de alta complejidad. Urgencias y medicina forense.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Fiscalía", "icon": "⚖️", "nom": "URI Fiscalía Florencia",
             "dir": "Cra 11 #12-40, Florencia", "barrio": "Centro",
             "lat": 1.6155, "lon": -75.6075, "color": "#7C3AED",
             "href": "tel:018000919748", "phone": "018000919748", "horario": "24/7 Sin cita",
             "desc": "Unidad de Reacción Inmediata. Denuncias penales urgentes.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Florencia",
             "dir": "Cra 12 #11-20, Florencia", "barrio": "Centro",
             "lat": 1.6158, "lon": -75.6078, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Bus local · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Florencia",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 1.6160, "lon": -75.6080, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # GUAVIARE
        # ══════════════════════════════════════════════════════════════════════
        "san josé del guaviare": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía San José del Guaviare",
             "dir": "Cra 18 #12-20, San José del Guaviare", "barrio": "Centro",
             "lat": 2.5670, "lon": -72.6390, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de San José del Guaviare. Denuncias y emergencias.",
             "transporte": "Taxi local · Transporte fluvial"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental San José del Guaviare",
             "dir": "Cra 19 #10-50, San José del Guaviare", "barrio": "Centro",
             "lat": 2.5675, "lon": -72.6395, "color": "#059669",
             "href": "tel:5840050", "phone": "584-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital departamental. Urgencias y atención a víctimas.",
             "transporte": "Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia San José del Guaviare",
             "dir": "Cra 20 #11-30, San José del Guaviare", "barrio": "Centro",
             "lat": 2.5665, "lon": -72.6385, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer San José del Guaviare",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 2.5670, "lon": -72.6390, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # AMAZONAS
        # ══════════════════════════════════════════════════════════════════════
        "leticia": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Leticia",
             "dir": "Cra 12 #8-20, Leticia", "barrio": "Centro",
             "lat": -4.2150, "lon": -69.9400, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Leticia. Denuncias y medidas de protección.",
             "transporte": "Mototaxi · Taxi local"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Rafael de Leticia",
             "dir": "Calle 9 #13-50, Leticia", "barrio": "Centro",
             "lat": -4.2145, "lon": -69.9405, "color": "#059669",
             "href": "tel:5927222", "phone": "592-7222", "horario": "24/7 Urgencias",
             "desc": "Hospital de Leticia. Urgencias y atención a víctimas.",
             "transporte": "Mototaxi · Taxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Leticia",
             "dir": "Cra 13 #7-30, Leticia", "barrio": "Centro",
             "lat": -4.2155, "lon": -69.9395, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Mototaxi · Taxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Leticia",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": -4.2150, "lon": -69.9400, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # SAN ANDRÉS
        # ══════════════════════════════════════════════════════════════════════
        "san andrés": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía San Andrés",
             "dir": "Av. Newball #3-20, San Andrés", "barrio": "Centro",
             "lat": 12.5440, "lon": -81.7200, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de San Andrés. Denuncias y medidas de protección.",
             "transporte": "Taxi · Bus local (rodadero)"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Timothy Britton San Andrés",
             "dir": "Av. Colombia, San Andrés", "barrio": "North End",
             "lat": 12.5455, "lon": -81.7215, "color": "#059669",
             "href": "tel:5123636", "phone": "512-3636", "horario": "24/7 Urgencias",
             "desc": "Hospital insular. Urgencias y atención a víctimas de violencia.",
             "transporte": "Taxi · Bus local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia San Andrés",
             "dir": "Cra 6 #10-20, San Andrés", "barrio": "Centro",
             "lat": 12.5445, "lon": -81.7205, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Taxi · Bus local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer San Andrés",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 12.5440, "lon": -81.7200, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # VICHADA
        # ══════════════════════════════════════════════════════════════════════
        "puerto carreño": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Puerto Carreño",
             "dir": "Cra 4 #6-20, Puerto Carreño", "barrio": "Centro",
             "lat": 6.1890, "lon": -67.4840, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Puerto Carreño. Denuncias y emergencias.",
             "transporte": "Taxi local · Transporte fluvial"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental de Puerto Carreño",
             "dir": "Calle 7 #5-50, Puerto Carreño", "barrio": "Centro",
             "lat": 6.1895, "lon": -67.4845, "color": "#059669",
             "href": "tel:5620050", "phone": "562-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital departamental. Urgencias y atención a víctimas.",
             "transporte": "Taxi local"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Puerto Carreño",
             "dir": "Cra 5 #5-30, Puerto Carreño", "barrio": "Centro",
             "lat": 6.1885, "lon": -67.4835, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Taxi local"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Puerto Carreño",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 6.1890, "lon": -67.4840, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # GUAINÍA
        # ══════════════════════════════════════════════════════════════════════
        "inírida": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Inírida",
             "dir": "Cra 10 #5-20, Inírida", "barrio": "Centro",
             "lat": 3.8650, "lon": -67.9240, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Inírida. Denuncias y emergencias.",
             "transporte": "Mototaxi · Transporte fluvial"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital Departamental de Inírida",
             "dir": "Cra 11 #4-50, Inírida", "barrio": "Centro",
             "lat": 3.8655, "lon": -67.9245, "color": "#059669",
             "href": "tel:5580050", "phone": "558-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital departamental. Urgencias y atención a víctimas.",
             "transporte": "Mototaxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Inírida",
             "dir": "Cra 12 #3-30, Inírida", "barrio": "Centro",
             "lat": 3.8645, "lon": -67.9235, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Mototaxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Inírida",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 3.8650, "lon": -67.9240, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],

        # ══════════════════════════════════════════════════════════════════════
        # VAUPÉS
        # ══════════════════════════════════════════════════════════════════════
        "mitú": [
            {"tipo": "Policía", "icon": "🚔", "nom": "Estación de Policía Mitú",
             "dir": "Cra 8 #4-20, Mitú", "barrio": "Centro",
             "lat": 1.2530, "lon": -70.2330, "color": "#1D4ED8",
             "href": "tel:123", "phone": "123", "horario": "24/7",
             "desc": "Estación de Policía de Mitú. Denuncias y medidas de protección.",
             "transporte": "Mototaxi · Transporte fluvial"},
            {"tipo": "Hospital", "icon": "🏥", "nom": "Hospital San Antonio Mitú",
             "dir": "Cra 9 #3-50, Mitú", "barrio": "Centro",
             "lat": 1.2535, "lon": -70.2335, "color": "#059669",
             "href": "tel:5640050", "phone": "564-0050", "horario": "24/7 Urgencias",
             "desc": "Hospital departamental. Urgencias y atención a víctimas.",
             "transporte": "Mototaxi"},
            {"tipo": "Comisaría", "icon": "🏛️", "nom": "Comisaría de Familia Mitú",
             "dir": "Cra 10 #2-30, Mitú", "barrio": "Centro",
             "lat": 1.2525, "lon": -70.2325, "color": "#0891B2",
             "href": "tel:123", "phone": "Presencial", "horario": "Lun–Vie 8am–5pm",
             "desc": "Medidas de protección por violencia intrafamiliar.",
             "transporte": "Mototaxi"},
            {"tipo": "Refugio", "icon": "🏠", "nom": "Casa Mujer Mitú",
             "dir": "Dirección confidencial — Llama al 155", "barrio": "Confidencial",
             "lat": 1.2530, "lon": -70.2330, "color": "#D97706",
             "href": "tel:155", "phone": "155", "horario": "24/7",
             "desc": "Alojamiento seguro coordinado con Línea 155.",
             "transporte": "Llama al 155"},
        ],
    }  # fin ENTIDADES_COL
    # ── Normalización robusta ─────────────────────────────────────────────────
    import unicodedata as _uc
    def _norm(s):
        s = s.lower().strip()
        s = _uc.normalize('NFD', s)
        s = ''.join(c for c in s if _uc.category(c) != 'Mn')
        return s

    _NORM_INDEX = {_norm(k): k for k in ENTIDADES_COL.keys()}

    def _find_city(texto):
        q = _norm(texto)
        if not q:
            return None
        if q in _NORM_INDEX:
            return _NORM_INDEX[q]
        for kn, k_orig in _NORM_INDEX.items():
            if kn in q or q in kn:
                return k_orig
        for kn, k_orig in _NORM_INDEX.items():
            if any(word in q for word in kn.split() if len(word) > 3):
                return k_orig
        if len(q) >= 4:
            for kn, k_orig in _NORM_INDEX.items():
                if kn.startswith(q[:4]) or q.startswith(kn[:4]):
                    return k_orig
        return None

    # Entidades nacionales siempre disponibles
    ENTIDADES_NACIONALES = [
        {"tipo":"Línea Nacional","icon":"📞","nom":"Línea 155 — Mujer","dir":"Línea gratuita nacional","barrio":"Nacional","lat":0,"lon":0,"color":"#EC4899","href":"tel:155","phone":"155","horario":"24/7 Gratuita","desc":"Línea de orientación y apoyo para mujeres víctimas de violencia. Gratuita desde cualquier teléfono en Colombia. Te orientan y activan recursos de protección.","transporte":"Llama al 155 desde cualquier teléfono — sin costo"},
        {"tipo":"Línea Nacional","icon":"🚨","nom":"Emergencias — 123","dir":"Línea gratuita nacional","barrio":"Nacional","lat":0,"lon":0,"color":"#DC2626","href":"tel:123","phone":"123","horario":"24/7 Gratuita","desc":"Línea de emergencias de la Policía Nacional. Para situaciones de peligro inmediato, acuden al lugar.","transporte":"Llama al 123 inmediatamente en caso de peligro"},
        {"tipo":"Línea Nacional","icon":"👨‍👩‍👧","nom":"ICBF — Línea 141","dir":"Línea gratuita nacional","barrio":"Nacional","lat":0,"lon":0,"color":"#059669","href":"tel:141","phone":"141","horario":"24/7 Gratuita","desc":"Instituto Colombiano de Bienestar Familiar. Protección familiar, menores en riesgo, orientación a mujeres.","transporte":"Llama al 141 desde cualquier teléfono — sin costo"},
    ]

    # ── Geolocalización automática vía query params ───────────────────────────
    # Leer coordenadas GPS inyectadas por el script de geolocalización
    qp = st.query_params
    if "geo_lat" in qp and "geo_lon" in qp and "geo_city" in qp:
        try:
            _glat = float(qp["geo_lat"])
            _glon = float(qp["geo_lon"])
            _gcity = qp["geo_city"]
            if st.session_state.get("gps_lat") != _glat:
                st.session_state["gps_lat"]  = _glat
                st.session_state["gps_lon"]  = _glon
                st.session_state["detected_city"] = _gcity
                st.session_state["geo_auto_done"] = True
        except Exception:
            pass

    st.components.v1.html("""
    <script>
    (function() {
        if (sessionStorage.getItem('_geo_done')) return;

        function norm(s) {
            return s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'').trim();
        }

        function send(lat, lon, city) {
            var url = new URL(window.parent.location.href);
            var prev = url.searchParams.get('geo_lat');
            var next = lat.toFixed(6);
            if (prev === next) return;
            sessionStorage.setItem('_geo_done', '1');
            url.searchParams.set('geo_lat', next);
            url.searchParams.set('geo_lon', lon.toFixed(6));
            url.searchParams.set('geo_city', norm(city));
            window.parent.history.replaceState({}, '', url.toString());
            setTimeout(function() {
                window.parent.location.href = url.toString();
            }, 300);
        }

        if (!navigator.geolocation) return;

        navigator.geolocation.getCurrentPosition(
            function(pos) {
                var lat = pos.coords.latitude;
                var lon = pos.coords.longitude;
                fetch('https://nominatim.openstreetmap.org/reverse?lat='+lat+'&lon='+lon+'&format=json&accept-language=es')
                    .then(function(r){ return r.json(); })
                    .then(function(d){
                        var a = d.address || {};
                        var city = a.city || a.town || a.municipality || a.county || 'colombia';
                        send(lat, lon, city);
                    })
                    .catch(function(){ send(lat, lon, 'colombia'); });
            },
            function(err){ console.warn('Geo error:', err.message); },
            {enableHighAccuracy: true, timeout: 12000, maximumAge: 60000}
        );
    })();
    </script>
    """, height=0)

    # ── Cabecera con botón de ubicación ──────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#EFF6FF;
            border:1px solid #BFDBFE;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#1D4ED8;font-weight:700;margin-bottom:12px;">🚔 AYUDA CERCANA</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 8px;letter-spacing:-0.5px;">Ayuda Cercana</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">
            Detecta tu ubicación automáticamente y encuentra la ayuda más cercana — policía, hospitales, fiscalía, refugios.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Alerta de emergencia siempre visible arriba
    st.markdown("""
    <div style="background:linear-gradient(135deg,#FEF2F2,#FEE2E2);border:2px solid #FCA5A5;border-radius:18px;
        padding:16px 22px;margin-bottom:20px;display:flex;align-items:center;gap:16px;">
        <div style="font-size:28px;">🚨</div>
        <div style="flex:1;">
            <div style="font-weight:900;color:#991B1B;font-size:14px;margin-bottom:4px;">¿Estás en peligro ahora mismo?</div>
            <div style="font-size:12px;color:#7F1D1D;line-height:1.6;">Llama <strong>123</strong> (Policía) o <strong>155</strong> (Línea Mujer) — ambas son gratuitas, 24 horas, desde cualquier celular.</div>
        </div>
        <div style="display:flex;gap:8px;">
            <a href="tel:123" style="text-decoration:none;">
                <div style="background:#DC2626;color:#fff;border-radius:12px;padding:10px 18px;font-size:15px;font-weight:900;text-align:center;
                    box-shadow:0 4px 14px rgba(220,38,38,0.4);">📞 123</div>
            </a>
            <a href="tel:155" style="text-decoration:none;">
                <div style="background:#7C3AED;color:#fff;border-radius:12px;padding:10px 18px;font-size:15px;font-weight:900;text-align:center;
                    box-shadow:0 4px 14px rgba(124,58,237,0.4);">💜 155</div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Índice normalizado para búsqueda rápida
    _NORM_INDEX = {_norm(k): k for k in ENTIDADES_COL.keys()}

    def _find_city(texto):
        q = _norm(texto)
        if not q:
            return None
        # 1) Exacta
        if q in _NORM_INDEX:
            return _NORM_INDEX[q]
        # 2) La clave contiene el query o viceversa
        for kn, k_orig in _NORM_INDEX.items():
            if kn in q or q in kn:
                return k_orig
        # 3) Por palabras individuales (>3 chars)
        for kn, k_orig in _NORM_INDEX.items():
            if any(word in q for word in kn.split() if len(word) > 3):
                return k_orig
        # 4) Por inicio (>=4 chars)
        if len(q) >= 4:
            for kn, k_orig in _NORM_INDEX.items():
                if kn.startswith(q[:4]) or q.startswith(kn[:4]):
                    return k_orig
        return None

    # ── Helper: distancia haversine en km ────────────────────────────────────
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    # ── Estado: coordenadas GPS reales (si se detectaron) ────────────────────
    gps_lat = st.session_state.get("gps_lat")
    gps_lon = st.session_state.get("gps_lon")
    geo_auto_done = st.session_state.get("geo_auto_done", False)

    # ── Selector de ciudad ────────────────────────────────────────────────────
    if geo_auto_done:
        _location_label = f"📡 Ubicación detectada automáticamente: **{st.session_state.get('detected_city','').title()}**"
        st.success(_location_label + "  — *Buscando entidades a 20 km a la redonda…*")

    city_col, clear_col = st.columns([3, 1])
    with city_col:
        city_input = st.text_input(
            "📍 O escribe tu ciudad (sin importar mayúsculas ni tildes):",
            value=st.session_state.get("detected_city", ""),
            key="ayuda_city",
            placeholder="Ej: bogota, medellín, CALI, barranquilla..."
        )
    with clear_col:
        st.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Borrar", use_container_width=True, key="clear_city"):
            st.session_state.pop("detected_city", None)
            st.session_state.pop("gps_lat", None)
            st.session_state.pop("gps_lon", None)
            st.session_state.pop("geo_auto_done", None)
            # Limpiar query params
            st.query_params.clear()
            st.rerun()
    
    # ── Buscar ciudad con normalización robusta ───────────────────────────────
    city_match = _find_city(city_input)

    # ── Seleccionar y ordenar entidades ───────────────────────────────────────
    # Si tenemos GPS exacto, buscar en TODAS las ciudades a ≤ 20 km
    RADIO_KM = 20.0
    entidades_ciudad = []
    usando_gps = False

    if gps_lat is not None and gps_lon is not None:
        # Recopilar todas las entidades de todas las ciudades y filtrar por distancia
        _todas = [e for lista in ENTIDADES_COL.values() for e in lista]
        entidades_cercanas = []
        for e in _todas:
            if e.get("lat") and e.get("lon") and e["lat"] != 0:
                dist = _haversine(gps_lat, gps_lon, e["lon"], e["lat"])  # nota: lat/lon en BD pueden estar invertidos
                # Intentar también con lat/lon en orden correcto
                dist2 = _haversine(gps_lat, gps_lon, e["lat"], e["lon"])
                d = min(dist, dist2)
                if d <= RADIO_KM:
                    entidades_cercanas.append((d, e))
        entidades_cercanas.sort(key=lambda x: x[0])
        entidades_ciudad = [e for _, e in entidades_cercanas]
        usando_gps = True
        if not entidades_ciudad:
            # Fallback a ciudad detectada si no hay nada a 20km
            entidades_ciudad = ENTIDADES_COL.get(city_match, [])
            usando_gps = False
    else:
        entidades_ciudad = ENTIDADES_COL.get(city_match, [])

    entidades = entidades_ciudad + ENTIDADES_NACIONALES

    if not entidades_ciudad:
        st.warning(
            f"📍 No tenemos entidades específicas para **{city_input.title()}** aún. "
            "Mostrando líneas nacionales disponibles para toda Colombia. "
            "Llama al **155** para que te orienten a la entidad más cercana.",
            icon="ℹ️"
        )
    else:
        _label = city_input.title() if city_input.strip() else "tu ubicación"
        _extra = f" · radio {RADIO_KM:.0f} km 📡" if usando_gps else ""
        st.markdown(
            f'<div style="font-size:12px;color:#6B7280;margin-bottom:16px;">'
            f'📍 Mostrando <strong style="color:#1E1B4B;">{len(entidades_ciudad)}</strong> entidades cerca de '
            f'<strong style="color:#1E1B4B;">{_label}</strong>{_extra} + líneas nacionales</div>',
            unsafe_allow_html=True
        )

    # ── Filtro por tipo ───────────────────────────────────────────────────────
    filter_tipo = st.selectbox(
        "🔍 Filtrar por tipo:",
        ["Todos","Policía","Hospital","Fiscalía","Comisaría","Refugio","Psicología","Línea Nacional"],
        key="ayuda_filter"
    )
    filtered = [e for e in entidades if filter_tipo == "Todos" or e["tipo"] == filter_tipo]

    # ── Grid de tarjetas ──────────────────────────────────────────────────────
    col_grid, col_det = st.columns([3, 2])

    with col_grid:
        gcols = st.columns(2)
        for i, e in enumerate(filtered):
            with gcols[i % 2]:
                is_sel = st.session_state.get("selected_entity") == e["nom"]
                border = f"2.5px solid {e['color']}" if is_sel else "1.5px solid #EDE9FE"
                shadow = f"0 8px 28px {e['color']}30" if is_sel else "0 2px 12px rgba(109,40,217,0.07)"
                bg = f"linear-gradient(135deg,{e['color']}08,#fff)" if is_sel else "#fff"
                maps_q = e['dir'].replace(' ', '+').replace('#', '%23')
                phone_badge = f'<div style="background:#FFF7ED;color:#D97706;font-size:9px;font-weight:700;padding:3px 9px;border-radius:20px;">📞 {e["phone"]}</div>' if e['phone'] != 'Presencial' else ''
                call_icon = '📞 Llamar' if e['href'].startswith('tel:') else '🌐 Web'
                html_card = f"""
                <div style="background:{bg};border:{border};border-radius:20px;
                    padding:18px;margin-bottom:12px;box-shadow:{shadow};cursor:pointer;transition:all 0.2s;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                        <div style="background:linear-gradient(135deg,{e['color']}22,{e['color']}10);border-radius:14px;
                            width:46px;height:46px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">{e['icon']}</div>
                        <div style="background:{e['color']}18;color:{e['color']};font-size:9px;font-weight:800;
                            padding:4px 10px;border-radius:20px;text-transform:uppercase;align-self:flex-start;">{e['tipo']}</div>
                    </div>
                    <div style="font-weight:800;font-size:13px;color:#1E1B4B;margin-bottom:4px;line-height:1.3;">{e['nom']}</div>
                    <div style="font-size:10px;color:#A78BFA;margin-bottom:4px;font-weight:600;">📍 {e['barrio']}</div>
                    <div style="font-size:10px;color:#6B7280;margin-bottom:10px;line-height:1.5;">{e['dir'][:55]}{'...' if len(e['dir'])>55 else ''}</div>
                    <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px;">
                        <div style="background:#ECFDF5;color:#059669;font-size:9px;font-weight:700;padding:3px 9px;border-radius:20px;">🕐 {e['horario']}</div>
                        {phone_badge}
                    </div>
                    <div style="display:flex;gap:6px;">
                        <a href="{e['href']}" style="text-decoration:none;flex:1;">
                            <div style="background:linear-gradient(135deg,{e['color']},{e['color']}CC);color:#fff;border-radius:10px;
                                padding:8px;text-align:center;font-size:11px;font-weight:800;">
                                {call_icon}
                            </div>
                        </a>
                        <a href="https://www.google.com/maps/search/?api=1&query={maps_q}" target="_blank" style="text-decoration:none;flex:1;">
                            <div style="background:#EFF6FF;color:#1D4ED8;border:1.5px solid #BFDBFE;border-radius:10px;
                                padding:8px;text-align:center;font-size:11px;font-weight:800;">🗺️ Maps</div>
                        </a>
                    </div>
                </div>"""
                st.markdown(html_card, unsafe_allow_html=True)
                btn_label = "✓ Ver menos" if is_sel else "ℹ️ Ver detalles"
                if st.button(btn_label, key=f"ent_{i}_{e['nom'][:12]}", use_container_width=True,
                             type="primary" if is_sel else "secondary"):
                    if is_sel:
                        st.session_state.pop("selected_entity", None)
                    else:
                        st.session_state["selected_entity"] = e["nom"]
                    st.rerun()

    # ── Panel de detalle ──────────────────────────────────────────────────────
    with col_det:
        sel_nom = st.session_state.get("selected_entity")
        sel_e = next((e for e in entidades if e["nom"] == sel_nom), None)

        if sel_e:
            st.markdown(f"""<div class="sh-card" style="position:sticky;top:16px;border-top:4px solid {sel_e['color']};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div style="background:linear-gradient(135deg,{sel_e['color']}22,{sel_e['color']}0E);border-radius:16px;
                        width:56px;height:56px;display:flex;align-items:center;justify-content:center;font-size:28px;flex-shrink:0;">{sel_e['icon']}</div>
                    <div style="background:{sel_e['color']}18;color:{sel_e['color']};font-size:9px;font-weight:800;
                        padding:5px 12px;border-radius:20px;text-transform:uppercase;">{sel_e['tipo']}</div>
                </div>
                <div style="font-size:17px;font-weight:900;color:#1E1B4B;margin-bottom:8px;line-height:1.3;">{sel_e['nom']}</div>
                <p style="font-size:12px;color:#374151;line-height:1.75;margin-bottom:16px;">{sel_e['desc']}</p>
            </div>""", unsafe_allow_html=True)

            # Ficha de datos
            datos = [
                ("📍 Dirección", sel_e['dir']),
                ("🏘️ Zona", sel_e['barrio']),
                ("🕐 Horario", sel_e['horario']),
                ("📞 Contacto", sel_e['phone']),
            ]
            for label, val in datos:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 12px;
                    background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:12px;margin-bottom:6px;">
                    <span style="font-size:11px;color:#7C3AED;font-weight:700;flex-shrink:0;">{label}</span>
                    <span style="font-size:11px;font-weight:800;color:#1E1B4B;text-align:right;margin-left:8px;line-height:1.4;">{val}</span>
                </div>""", unsafe_allow_html=True)

            # Cómo llegar
            if sel_e.get('transporte'):
                st.markdown(f"""<div style="background:linear-gradient(135deg,#EFF6FF,#DBEAFE);border-radius:14px;
                    padding:12px 14px;margin:10px 0;border:1px solid #BFDBFE;">
                    <div style="font-size:11px;font-weight:800;color:#1D4ED8;margin-bottom:5px;">🚌 Cómo llegar</div>
                    <div style="font-size:11px;color:#1E40AF;line-height:1.7;">{sel_e['transporte']}</div>
                </div>""", unsafe_allow_html=True)

            # Botones de acción principales
            st.markdown("<div style='margin-top:12px;display:flex;flex-direction:column;gap:8px;'>", unsafe_allow_html=True)

            call_label = f"📞 Llamar ahora: {sel_e['phone']}" if sel_e['href'].startswith('tel:') else "🌐 Visitar sitio web"
            st.markdown(f"""<a href="{sel_e['href']}" {'target="_blank"' if sel_e['href'].startswith('http') else ''} style="text-decoration:none;display:block;">
                <div style="width:100%;background:linear-gradient(135deg,{sel_e['color']},{sel_e['color']}BB);color:#fff;border-radius:14px;
                    padding:14px;text-align:center;font-size:13px;font-weight:900;
                    box-shadow:0 4px 16px {sel_e['color']}44;letter-spacing:0.3px;">{call_label}</div>
            </a>""", unsafe_allow_html=True)

            if sel_e['lat'] != 0:
                maps_url_dir = f"https://www.google.com/maps/dir/?api=1&destination={sel_e['lat']},{sel_e['lon']}&travelmode=transit"
                maps_url_walk = f"https://www.google.com/maps/dir/?api=1&destination={sel_e['lat']},{sel_e['lon']}&travelmode=walking"
                st.markdown(f"""
                <a href="{maps_url_dir}" target="_blank" style="text-decoration:none;display:block;margin-top:8px;">
                    <div style="width:100%;background:#fff;color:#1D4ED8;border:2px solid #BFDBFE;border-radius:14px;
                        padding:12px;text-align:center;font-size:12px;font-weight:800;">
                        🚌 Cómo llegar — Transporte público
                    </div>
                </a>
                <a href="{maps_url_walk}" target="_blank" style="text-decoration:none;display:block;margin-top:6px;">
                    <div style="width:100%;background:#F0FDF4;color:#059669;border:2px solid #BBF7D0;border-radius:14px;
                        padding:12px;text-align:center;font-size:12px;font-weight:800;">
                        🚶 Cómo llegar — A pie
                    </div>
                </a>
                """, unsafe_allow_html=True)
            else:
                maps_url_search = f"https://www.google.com/maps/search/?api=1&query={sel_e['nom'].replace(' ', '+')}"
                st.markdown(f"""<a href="{maps_url_search}" target="_blank" style="text-decoration:none;display:block;margin-top:8px;">
                    <div style="width:100%;background:#fff;color:#1D4ED8;border:2px solid #BFDBFE;border-radius:14px;
                        padding:12px;text-align:center;font-size:12px;font-weight:800;">🗺️ Ver en Google Maps</div>
                </a>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Instrucciones con IA
            if st.button("🤖 Instrucciones detalladas con IA", key="directions_btn", use_container_width=True):
                with st.spinner("Generando ruta personalizada..."):
                    prompt_ciudad = city_input if city_input else "Colombia"
                    dir_text = call_claude(
                        "Eres experta en transporte urbano colombiano. Da instrucciones claras con bullets y emojis. Incluye: TransMilenio/Metro/BRT según ciudad, taxi/Uber, a pie. Máx 120 palabras. Indica tiempo estimado y costo aproximado.",
                        f"¿Cómo llegar desde el centro de {prompt_ciudad} hasta {sel_e['nom']} ubicada en {sel_e['dir']}, barrio {sel_e.get('barrio','')}?"
                    )
                    st.session_state[f"dir_{sel_e['nom']}"] = dir_text

            dir_r = st.session_state.get(f"dir_{sel_e['nom']}", "")
            if dir_r:
                st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:14px;
                    padding:14px 16px;border:1px solid #C4B5FD;margin-top:10px;">
                    <div style="font-size:11px;font-weight:800;color:#5B21B6;margin-bottom:8px;">🧭 Ruta sugerida por SARA</div>
                    <div style="font-size:11px;color:#1E1B4B;line-height:1.85;white-space:pre-wrap;">{dir_r}</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:20px;
                border:2px dashed #C4B5FD;padding:40px 24px;text-align:center;position:sticky;top:16px;">
                <div style="font-size:48px;margin-bottom:14px;">📍</div>
                <div style="font-size:15px;font-weight:800;color:#5B21B6;margin-bottom:8px;">Selecciona una entidad</div>
                <div style="font-size:12px;color:#A78BFA;line-height:1.8;">
                    Haz clic en <strong>"Ver detalles"</strong> para ver la ficha completa:<br>
                    dirección, horario, cómo llegar y botón para llamar directamente.
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Líneas de emergencia siempre visibles abajo ───────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:800;color:#1E1B4B;font-size:14px;margin-bottom:14px;">📞 Líneas de emergencia — Todas gratuitas 24/7</div>', unsafe_allow_html=True)
    em_cols = st.columns(3)
    lineas = [
        ("🚨","123","Policía Nacional","Emergencias inmediatas","#DC2626","tel:123"),
        ("💜","155","Línea Mujer","Apoyo y orientación","#7C3AED","tel:155"),
        ("👨‍👩‍👧","141","ICBF","Protección familiar","#059669","tel:141"),
    ]
    for col, (icon, num, nombre, desc, color, href) in zip(em_cols, lineas):
        with col:
            st.markdown(f"""<a href="{href}" style="text-decoration:none;display:block;">
                <div style="background:linear-gradient(135deg,{color},{color}CC);color:#fff;border-radius:18px;
                    padding:18px;text-align:center;box-shadow:0 4px 18px {color}44;margin-bottom:8px;">
                    <div style="font-size:26px;margin-bottom:6px;">{icon}</div>
                    <div style="font-size:28px;font-weight:900;letter-spacing:1px;">{num}</div>
                    <div style="font-size:11px;font-weight:800;margin-top:4px;opacity:0.9;">{nombre}</div>
                    <div style="font-size:10px;opacity:0.8;margin-top:2px;">{desc}</div>
                </div>
            </a>""", unsafe_allow_html=True)
# ── ACERCA DE ─────────────────────────────────────────────────────────────────
elif "i️" in page:
    st.markdown("""
    <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin-bottom:28px;letter-spacing:-0.5px;">i️ Acerca de SafeHer Colombia</h1>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""<div class="sh-card">
            <div style="font-weight:800;color:#5B21B6;font-size:16px;margin-bottom:16px;">🎓 Proyecto Académico</div>
            <p style="font-size:13px;color:#1E1B4B;line-height:1.85;margin-bottom:20px;">
                Desarrollado como proyecto de <strong>Analítica y Machine Learning</strong>. Modelos entrenados con datos del
                Sistema de Información Estadístico de la <strong>Policía Nacional de Colombia</strong>.
                Plataforma integral de protección, prevención y apoyo para mujeres.
            </p>
            <div style="font-weight:800;color:#1E1B4B;margin-bottom:14px;font-size:14px;">👩‍💻 Equipo de Desarrollo:</div></div>""", unsafe_allow_html=True)

        for name in ["Laura Sofia Beltrán","Dana Yaray Vargas","Vanessa Mora"]:
            st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:14px;padding:13px 18px;
                display:flex;align-items:center;gap:14px;margin-bottom:10px;border:1px solid #C4B5FD;">
                <div style="width:38px;height:38px;background:linear-gradient(135deg,#4C1D95,#7C3AED);border-radius:50%;
                    display:flex;align-items:center;justify-content:center;font-size:18px;">👩‍🎓</div>
                <span style="font-size:13px;color:#1E1B4B;font-weight:700;">{name}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="sh-card">
            <div style="font-weight:800;color:#5B21B6;font-size:16px;margin-bottom:16px;">🤖 Modelos de Machine Learning</div></div>""", unsafe_allow_html=True)

        for ti, al, ta, co in [
            ("📊 Nivel de Gravedad","XGBoost + LightGBM","8 clases: MÍNIMO → CRÍTICO","#4C1D95"),
            ("🗺️ Zona de Riesgo","XGBoost + LightGBM","6 clases: MUY BAJO → MUY ALTO","#1D4ED8"),
            ("👥 Estimación de Víctimas","Ensemble de modelos","Valor numérico estimado","#059669"),
            ("🧠 IA de Apoyo (SARA)","Llama 3.3 70B (Groq)","Apoyo psicológico y legal","#7C3AED"),
        ]:
            st.markdown(f"""<div style="background:linear-gradient(135deg,{co}08,{co}04);border:1px solid {co}22;
                border-radius:16px;padding:14px 18px;margin-bottom:10px;">
                <div style="font-weight:800;color:#1E1B4B;font-size:13px;">{ti}</div>
                <div style="font-size:12px;color:{co};margin-top:3px;font-weight:600;">{al}</div>
                <div style="font-size:11px;color:#6B7280;margin-top:2px;">Target: {ta}</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div style="background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:1.5px solid #FCD34D;
            border-radius:20px;padding:20px;margin-bottom:18px;">
            <div style="font-weight:800;color:#92400E;font-size:14px;margin-bottom:10px;">⚠️ Limitaciones Importantes</div>
            <p style="font-size:12px;color:#78350F;line-height:1.8;">Plataforma <strong>académica prototipo</strong>.
            Las predicciones son aproximaciones estadísticas. Para emergencias reales llama al
            <strong>123</strong> o <strong>Línea 155</strong>.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-weight:800;color:#1E1B4B;font-size:14px;margin-bottom:16px;">🛠️ Stack Tecnológico</div>', unsafe_allow_html=True)
        for tech, pct, color in [
            ("Python + Streamlit","95%","#5B21B6"),("XGBoost","92%","#1D4ED8"),("LightGBM","90%","#059669"),
            ("Scikit-learn","88%","#D97706"),("Groq API (SARA)","100%","#7C3AED"),
            ("Pandas + NumPy","90%","#0891B2"),("Plotly + Folium","88%","#EC4899"),
        ]:
            st.markdown(f"""<div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;">
                    <span style="color:#1E1B4B;font-weight:600;">{tech}</span>
                    <span style="color:{color};font-weight:800;">{pct}</span>
                </div>
                <div style="background:#EDE9FE;border-radius:8px;height:9px;overflow:hidden;">
                    <div style="width:{pct};height:100%;background:linear-gradient(90deg,{color}80,{color});border-radius:8px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-weight:800;color:#5B21B6;font-size:14px;margin-bottom:14px;">📊 Cobertura del Sistema</div>', unsafe_allow_html=True)
        for v, k in [("Colombia completa","Cobertura"),("33","Departamentos"),("1.121","Municipios"),
                     ("6","Tipos de delito"),("Policía Nacional","Fuente de datos"),("2019–2027","Período de análisis")]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:9px 0;
                border-bottom:1px solid #EDE9FE;font-size:12px;">
                <span style="color:#6B7280;font-weight:500;">{k}</span>
                <span style="color:#1E1B4B;font-weight:800;">{v}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border:1.5px solid #C4B5FD;
            border-radius:18px;padding:18px;margin-top:16px;">
            <div style="font-weight:800;color:#5B21B6;font-size:13px;margin-bottom:10px;">🔑 Configuración de IA</div>
            <div style="font-size:12px;color:#1E1B4B;line-height:1.8;">
                Para activar todas las funciones de IA, crea el archivo:
            </div>
            <div style="background:#1E1B4B;border-radius:10px;padding:12px;margin:10px 0;font-family:monospace;font-size:11px;color:#A7F3D0;">
                .streamlit/secrets.toml<br>
                GROQ_API_KEY = "gsk_tu-clave-aqui"
            </div>
            <div style="font-size:11px;color:#7C3AED;font-weight:600;">O define la variable de entorno GROQ_API_KEY</div>
        </div>""", unsafe_allow_html=True)
