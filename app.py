import streamlit as st
import random
import math
import os
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
from groq import Groq
import plotly.graph_objects as go

# ─── CARGAR MODELOS PKL ───────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    import pathlib
    BASE_DIR = pathlib.Path("/mount/src/safeher-colombia")  
    models = {}
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
        fpath = BASE_DIR / fname
        try:
            with open(fpath, "rb") as f:
                models[key] = pickle.load(f)
        except Exception as e:
            models[key] = None
    return models
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
    +'section[data-testid="stSidebara"] label[data-baseweb="radio"]{'
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

# ─── PREDICCIÓN ML (CORREGIDA) ────────────────────────────────────────────────

def calc_prediction(dep, mun, delito, sexo, etario, año):
    MODELS = load_models()
    # ── Fallback sintético (siempre se calcula primero como respaldo) ─────────
    base       = CRIME_DATA.get(dep, {"score": 3.0, "zona": "MEDIO-BAJO", "gravedad": "BAJO", "municipios": 10})
    año_factor = 1.05 if año >= 2024 else (1.0 if año >= 2020 else 0.9)
    adjusted   = base["score"] * DELIT_FACTOR.get(delito, 1.0) * año_factor
    zonas      = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
    gravedades = ["MÍNIMO","MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO","CRÍTICO"]
    zona_idx   = min(max(round(adjusted) - 1, 0), 5)
    grav_idx   = min(max(round(adjusted),     0), 7)
    zona       = zonas[zona_idx]
    gravedad   = gravedades[grav_idx]
    victimas   = round(adjusted * 18 + random.random() * 10)

    used_pkl   = False
    pkl_errors = []

    # ── MAPEO DE VALORES AL FORMATO DEL MODELO ────────────────────────────────
    DELITO_MAP = {
        "VIOLENCIA INTRAFAMILIAR": "VIOLENCIA INTRAFAMILIAR",
        "VIOLENCIA SEXUAL":        "DELITOS SEXUALES",
        "LESIONES PERSONALES":     "LESIONES PERSONALES",
        "AMENAZAS":                "AMENAZAS",
        "HURTO":                   "LESIONES PERSONALES",  # fallback
        "HOMICIDIO":               "HOMICIDIO DOLOSO",
    }
    ETARIO_MAP = {
        "DE 0 A 17 AÑOS":   "DE 14 A 17 A",
        "DE 18 A 26 AÑOS":  "DE 18 A 26 A",
        "DE 27 A 59 AÑOS":  "DE 27 A 59 A",
        "DE 60 Y MÁS":      "MAYOR DE 60 A",
    }
    DEP_MAP = {
        "BOGOTÁ D.C.":           "BOGOTÁ, D. C.",
        "ANTIOQUIA":              "ANTIOQUIA",
        "VALLE DEL CAUCA":        "VALLE DEL CAUCA",
        "CUNDINAMARCA":           "CUNDINAMARCA",
        "ATLÁNTICO":              "ATLÁNTICO",
        "SANTANDER":              "SANTANDER",
        "NARIÑO":                 "NARIÑO",
        "CÓRDOBA":                "CÓRDOBA",
        "BOLÍVAR":                "BOLÍVAR",
        "TOLIMA":                 "TOLIMA",
        "HUILA":                  "HUILA",
        "CAUCA":                  "CAUCA",
        "META":                   "META",
        "CESAR":                  "CESAR",
        "MAGDALENA":              "MAGDALENA",
        "BOYACÁ":                 "BOYACÁ",
        "CALDAS":                 "CALDAS",
        "RISARALDA":              "RISARALDA",
        "QUINDÍO":                "QUINDÍO",
        "NORTE DE SANTANDER":     "NORTE DE SANTANDER",
        "SUCRE":                  "SUCRE",
        "LA GUAJIRA":             "LA GUAJIRA",
        "CAQUETÁ":                "CAQUETÁ",
        "ARAUCA":                 "ARAUCA",
        "CASANARE":               "CASANARE",
        "VICHADA":                "VICHADA",
        "GUAINÍA":                "GUAINÍA",
        "GUAVIARE":               "GUAVIARE",
        "VAUPÉS":                 "VAUPÉS",
        "AMAZONAS":               "AMAZONAS",
        "PUTUMAYO":               "PUTUMAYO",
        "CHOCÓ":                  "CHOCÓ",
        "SAN ANDRÉS":             "ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA",
    }

    delito_pkl  = DELITO_MAP.get(delito, "LESIONES PERSONALES")
    etario_pkl  = ETARIO_MAP.get(etario, "DE 27 A 59 A")
    dep_pkl     = DEP_MAP.get(dep, dep)

    FEATURE_COLS = ["DEPARTAMENTO_HECHO", "MUNICIPIO_HECHO", "GRUPO_DELITO", "SEXO", "GRUPO_ETARIO", "AÑO"]

    # ── PREDICCIÓN ZONA con XGBoost ───────────────────────────────────────────
    try:
        xgb_z = MODELS.get("xgb_zona")
        enc_z = MODELS.get("encoders_zona")
        le_z  = MODELS.get("le_zona")

        if xgb_z is None:
            raise ValueError("xgb_zona.pkl no cargado o no encontrado en el directorio")
        if enc_z is None:
            raise ValueError("encoders_zona.pkl no cargado o no encontrado en el directorio")
        if le_z is None:
            raise ValueError("le_target_zona.pkl no cargado o no encontrado en el directorio")

        row = pd.DataFrame([{
            "DEPARTAMENTO_HECHO": dep_pkl,
            "MUNICIPIO_HECHO":    mun,
            "GRUPO_DELITO":       delito_pkl,
            "SEXO":               sexo,
            "GRUPO_ETARIO":       etario_pkl,
            "AÑO":                int(año),
        }])[FEATURE_COLS]

        col_map = {
            "DEPARTAMENTO_HECHO": "DEPARTAMENTO_HECHO",
            "MUNICIPIO_HECHO":    "MUNICIPIO_HECHO",
            "GRUPO_DELITO":       "GRUPO_DELITO",
            "SEXO":               "SEXO",
            "GRUPO_ETARIO":       "GRUPO_ETARIO",
        }
        for col, enc_key in col_map.items():
            if enc_key not in enc_z:
                raise ValueError(f"Encoder '{enc_key}' no encontrado en encoders_zona.pkl")
            le_col = enc_z[enc_key]
            val    = str(row[col].iloc[0])
            if val not in le_col.classes_:
                fallback = le_col.classes_[0]
                pkl_errors.append(f"⚠️ Zona — '{val}' no visto en {col}. Fallback: '{fallback}'")
                row[col] = le_col.transform([fallback])[0]
            else:
                row[col] = le_col.transform([val])[0]

        row["AÑO"] = int(año)
        zona_pred_num = xgb_z.predict(row)[0]
        zona          = le_z.inverse_transform([int(zona_pred_num)])[0]
        used_pkl      = True

    except Exception as e:
        pkl_errors.append(f"❌ Predicción ZONA (XGBoost) falló: {e}")

    # ── PREDICCIÓN GRAVEDAD con XGBoost ──────────────────────────────────────
    try:
        xgb_g = MODELS.get("xgb_gravedad")
        le_g  = MODELS.get("le_gravedad")
        pre_g = MODELS.get("preprocessor_grav")
        scl_g = MODELS.get("scaler_gravedad")

        if xgb_g is None:
            raise ValueError("xgb_gravedad.pkl no cargado o no encontrado en el directorio")
        if le_g is None:
            raise ValueError("le_target_gravedad.pkl no cargado o no encontrado en el directorio")

        row2 = pd.DataFrame([{
            "DEPARTAMENTO_HECHO": dep_pkl,
            "MUNICIPIO_HECHO":    mun,
            "GRUPO_DELITO":       delito_pkl,
            "SEXO":               sexo,
            "GRUPO_ETARIO":       etario_pkl,
            "AÑO":                int(año),
        }])[FEATURE_COLS]

        if pre_g is not None:
            row2_transformed = pre_g.transform(row2)
        elif scl_g is not None:
            enc_z2 = MODELS.get("encoders_zona")
            if enc_z2 is not None:
                for col, enc_key in col_map.items():
                    if enc_key in enc_z2:
                        le_col2 = enc_z2[enc_key]
                        val2    = str(row2[col].iloc[0])
                        closest = val2 if val2 in le_col2.classes_ else le_col2.classes_[0]
                        if val2 not in le_col2.classes_:
                            pkl_errors.append(f"⚠️ Gravedad — '{val2}' no visto en {col}. Fallback: '{closest}'")
                        row2[col] = le_col2.transform([closest])[0]
            row2["AÑO"]      = int(año)
            row2_transformed = scl_g.transform(row2)
        else:
            row2_transformed = row2

        grav_pred_num = xgb_g.predict(row2_transformed)[0]
        gravedad      = le_g.inverse_transform([int(grav_pred_num)])[0]
        used_pkl      = True

    except Exception as e:
        pkl_errors.append(f"❌ Predicción GRAVEDAD (XGBoost) falló: {e}")
    # ── NOTA: lgbm_zona y pipe_lgbm_gravedad están cargados pero no se usan.
    # Si quieres activarlos como ensemble, descomenta el bloque de abajo:
    #
    # try:
    #     lgbm_z = MODELS.get("lgbm_zona")
    #     if lgbm_z is not None and used_pkl:
    #         zona_lgbm = lgbm_z.predict(row)[0]
    #         zona_lgbm_label = le_z.inverse_transform([int(zona_lgbm)])[0]
    #         # Ensemble simple: mayoría de votos entre xgb y lgbm
    #         # zona = zona si xgb == lgbm, si no usa el de mayor confianza
    # except Exception as e:
    #     pkl_errors.append(f"⚠️ LGBM zona: {e}")

    # ── Cálculos auxiliares ───────────────────────────────────────────────────
    probs_zona = {}
    for i, z in enumerate(zonas):
        dist = abs(i - zona_idx)
        probs_zona[z] = max(2, 100 - dist * 28 + (random.random() * 6 - 3))
    total_z    = sum(probs_zona.values())
    probs_zona = {k: round(v / total_z * 100, 1) for k, v in probs_zona.items()}

    trend = []
    for y in [2019,2020,2021,2022,2023,2024,2025,2026,2027]:
        yf    = 1.05 if y >= 2024 else (1.0 if y >= 2020 else 0.9)
        noise = random.random() * 0.3 - 0.15
        s     = base["score"] * DELIT_FACTOR.get(delito, 1.0) * yf * (1 + (y - 2020) * 0.025) + noise
        trend.append({"year": y, "score": round(min(max(s, 0.5), 6.0), 2), "projected": y >= 2025})

    comparativa = []
    for d in DELITOS:
        df = DELIT_FACTOR.get(d, 1.0)
        sc = base["score"] * df * año_factor
        zi = min(max(round(sc) - 1, 0), 5)
        comparativa.append({"label": d, "value": round(sc, 1), "risk": zonas[zi]})
    comparativa.sort(key=lambda x: x["value"], reverse=True)

    months   = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    seasonal = [0.85,0.8,0.9,0.95,1.0,1.05,1.1,1.15,1.0,0.95,1.1,1.3]
    monthly  = [
        {"month": m,
         "value": round(adjusted * seasonal[i] * (1 + random.random()*0.1-0.05), 2),
         "cases": round(victimas/12 * seasonal[i] * (1+random.random()*0.2-0.1))}
        for i, m in enumerate(months)
    ]

    return {
        "zona":        zona,
        "gravedad":    gravedad,
        "victimas":    victimas,
        "probs_zona":  probs_zona,
        "trend":       trend,
        "comparativa": comparativa,
        "score":       round(adjusted, 1),
        "zona_idx":    zona_idx,
        "monthly":     monthly,
        "used_pkl":    used_pkl,
        "pkl_errors":  pkl_errors,   # ← NUEVO: errores/advertencias visibles
    }

# ── Datos de municipios ───────────────────────────────────────────────────────
@st.cache_data
def build_municipio_data():
    data  = {}
    zonas = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
    for dep_name, dep_info in CRIME_DATA.items():
        muns     = get_municipios(dep_name)
        base     = dep_info["score"]
        mun_list = []
        for mun in muns:
            seed      = int(hashlib.md5(f"{dep_name}{mun}".encode()).hexdigest(), 16) % 1000
            variation = (seed / 1000.0 - 0.5) * 1.4
            score     = round(min(max(base + variation, 0.8), 5.9), 2)
            zona_idx  = min(max(round(score) - 1, 0), 5)
            zona      = zonas[zona_idx]
            mun_list.append({"name": mun, "score": score, "zona": zona, "dep": dep_name})
        data[dep_name] = mun_list
    return data

MUNICIPIO_DATA = build_municipio_data()

# ── GeoJSON Colombia ──────────────────────────────────────────────────────────
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
        form   = st.session_state.get("pred_form", {})

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

            # ── Badge PKL con diagnóstico detallado (CORREGIDO) ───────────────
            if result.get("used_pkl"):
                st.markdown('''<div style="background:#ECFDF5;border:1px solid #6EE7B740;border-radius:12px;
                    padding:10px 16px;margin:12px 0;display:inline-block;font-size:12px;font-weight:700;color:#059669;">
                    ✅ Modelos PKL reales activos — predicciones desde archivos entrenados</div>''',
                    unsafe_allow_html=True)
            else:
                st.markdown('''<div style="background:#FFFBEB;border:1px solid #FCD34D40;border-radius:12px;
                    padding:10px 16px;margin:12px 0;display:inline-block;font-size:12px;font-weight:700;color:#D97706;">
                    ⚙️ Modo simulación — los modelos PKL no pudieron cargarse o tuvieron errores</div>''',
                    unsafe_allow_html=True)

            if result.get("pkl_errors"):
                with st.expander(f"🔍 Diagnóstico PKL — {len(result['pkl_errors'])} mensaje(s)", expanded=not result.get("used_pkl")):
                    for err in result["pkl_errors"]:
                        color_err = "#DC2626" if err.startswith("❌") else "#D97706"
                        st.markdown(
                            f'<div style="font-size:12px;color:{color_err};padding:5px 0;'
                            f'font-family:monospace;border-bottom:1px solid #EDE9FE;">{err}</div>',
                            unsafe_allow_html=True
                        )

            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Tendencia Histórica", "🎯 Distribución de Probabilidad", "📅 Variación Mensual", "🕸️ Radar de Riesgo"])

            with tab1:
                solid_x = [t["year"] for t in result["trend"] if not t["projected"]]
                solid_y = [t["score"] for t in result["trend"] if not t["projected"]]
                proj_x  = [solid_x[-1]] + [t["year"] for t in result["trend"] if t["projected"]]
                proj_y  = [solid_y[-1]] + [t["score"] for t in result["trend"] if t["projected"]]
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
                radar_vals     = [round(min(v, 1.0), 2) for v in radar_vals]
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
                cfg    = RISK_LEVELS.get(item["risk"], {"color": "#888"})
                is_sel = item["label"] == form.get("delito", delito)
                bg     = cfg["color"] + "12" if is_sel else "#FAFAFA"
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
            score   = dep_data["score"]
            color   = get_risk_color(score)
            is_sel  = dep_name == sel_dep
            pct     = int(score / 6 * 100)
            zona    = dep_data["zona"]
            gravedad= dep_data["gravedad"]
            muns    = dep_data["municipios"]
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
            weight   = 3 if is_sel else 1.5
            fill_op  = 0.88 if is_sel else 0.72
            stroke_c = "#1E1B4B" if is_sel else "#ffffff"
            folium.GeoJson(feature,
                style_function=lambda x, c=color, w=weight, fo=fill_op, sc=stroke_c: {
                    "fillColor": c, "color": sc, "weight": w, "fillOpacity": fo},
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=240),
                highlight_function=lambda x, c=color: {"fillColor": c, "fillOpacity": 0.95, "weight": 3, "color": "#1E1B4B"},
            ).add_to(m)
            coords    = feature["geometry"]["coordinates"][0]
            lons      = [p[0] for p in coords]; lats = [p[1] for p in coords]
            cx        = sum(lons)/len(lons); cy = sum(lats)/len(lats)
            short_name= dep_name.split()[0][:8] if len(dep_name) > 12 else dep_name[:10]
            folium.Marker(location=[cy, cx],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:8px;font-weight:800;color:white;text-shadow:0 1px 3px rgba(0,0,0,0.7);white-space:nowrap;text-align:center;line-height:1.2;"><div>{short_name}</div><div style="font-size:9px;">{score:.1f}</div></div>',
                    icon_size=(70, 28), icon_anchor=(35, 14)),
            ).add_to(m)
        _add_legend(m)
        return m._repr_html_()

    @st.cache_data
    def build_folium_map_mun(dep_name_sel, mun_data_json):
        mun_data   = json.loads(mun_data_json)
        dep_info   = CRIME_DATA.get(dep_name_sel, {})
        dep_feature= next((f for f in COLOMBIA_GEO["features"] if f["properties"]["DPTO"] == dep_name_sel), None)
        if dep_feature:
            coords      = dep_feature["geometry"]["coordinates"][0]
            center_lat  = sum(p[1] for p in coords) / len(coords)
            center_lon  = sum(p[0] for p in coords) / len(coords)
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
            score    = mun["score"]
            zona     = mun["zona"]
            color    = get_risk_color(score)
            pct      = int(score / 6 * 100)
            if mun_name in MUN_COORDS:
                lat, lon = MUN_COORDS[mun_name]
            else:
                seed    = int(hashlib.md5(f"{dep_name_sel}{mun_name}".encode()).hexdigest(), 16)
                lat_off = ((seed % 1000) / 1000.0 - 0.5) * 1.5
                lon_off = (((seed // 1000) % 1000) / 1000.0 - 0.5) * 1.5
                lat     = center_lat + lat_off
                lon     = center_lon + lon_off
            tooltip_html = f"""
            <div style="font-family:'Segoe UI',sans-serif;min-width:180px;padding:2px;">
                <div style="font-weight:800;font-size:13px;color:#1E1B4B;border-bottom:2px solid {color};padding-bottom:4px;margin-bottom:8px;">📍 {mun_name}</div>
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
                <div style="background:{color};color:white;padding:8px 12px;border-radius:8px 8px 0 0;font-weight:800;font-size:13px;">📍 {mun_name}</div>
                <div style="padding:10px 12px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;">
                    <div style="font-size:10px;color:#6B7280;margin-bottom:6px;">{dep_name_sel}</div>
                    <div style="font-size:26px;font-weight:900;color:{color};margin-bottom:4px;">{score:.1f}<span style="font-size:12px;color:#9CA3AF;">/ 6.0</span></div>
                    <div style="background:#F3F4F6;border-radius:4px;height:7px;margin-bottom:8px;">
                        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
                    </div>
                    <div style="font-size:11px;font-weight:700;color:{color};">Zona: {zona}</div>
                </div>
            </div>"""
            radius = 8 + score * 3
            folium.CircleMarker(location=[lat, lon], radius=radius,
                color="#1E1B4B", weight=1.5, fill=True, fill_color=color, fill_opacity=0.85,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(m)
            folium.Marker(location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:7px;font-weight:800;color:#1E1B4B;text-shadow:0 0 3px white,0 0 3px white;white-space:nowrap;text-align:center;margin-top:{int(radius)+6}px;">{mun_name[:12]}</div>',
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
            map_html      = build_folium_map_mun(dep_muni_sel, mun_data_json)
            components.html(map_html, height=540, scrolling=False)
            st.markdown(f'<div style="font-size:11px;color:#6B7280;text-align:center;margin-top:4px;">🖱️ Zoom · Clic en círculo para detalles · Tamaño proporcional al score · Depto: <strong>{dep_muni_sel}</strong></div>', unsafe_allow_html=True)
        else:
            sel_dep_map = st.session_state.get("selected_dep", "")
            map_html    = build_folium_map_dep(filter_zone, sel_dep_map)
            components.html(map_html, height=540, scrolling=False)
            st.markdown('<div style="font-size:11px;color:#6B7280;text-align:center;margin-top:4px;">🖱️ Zoom con scroll · Clic en departamento para detalles · Pasa cursor para info rápida</div>', unsafe_allow_html=True)

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

        if map_view == "Por Municipio":
            mun_data_list = MUNICIPIO_DATA.get(dep_muni_sel, [])
            mun_sorted    = sorted(mun_data_list, key=lambda x: x["score"], reverse=True)
            st.markdown(f'<div style="font-size:13px;font-weight:700;color:#1E1B4B;margin:18px 0 12px;">📊 Ranking de Municipios — {dep_muni_sel}</div>', unsafe_allow_html=True)
            max_mun_score = mun_sorted[0]["score"] if mun_sorted else 1
            for i, mun in enumerate(mun_sorted):
                color = get_risk_color(mun["score"])
                pct   = mun["score"] / max_mun_score * 100
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
                color     = get_risk_color(d["score"])
                pct       = d["score"] / 6 * 100
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
            mun_data_list  = MUNICIPIO_DATA.get(dep_muni_sel, [])
            mun_sorted_d   = sorted(mun_data_list, key=lambda x: x["score"], reverse=True)
            dep_info       = CRIME_DATA.get(dep_muni_sel, {})
            dep_color      = get_risk_color(dep_info.get("score", 3.0))
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
                    top_muns  = ", ".join([f"{m['name']} ({m['score']:.1f})" for m in mun_sorted_d[:3]])
                    safe_muns = ", ".join([f"{m['name']} ({m['score']:.1f})" for m in mun_sorted_d[-3:][::-1]])
                    ai_mun    = call_claude(
                        "Eres experto en seguridad pública colombiana. Análisis breve (máx 130 palabras) con bullets y emojis sobre distribución de riesgo entre municipios de un departamento.",
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
                sel   = {"name": sel_name, **CRIME_DATA[sel_name]}
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
                    sc  = sel["score"] * fac
                    z   = zonas_list[min(max(round(sc) - 1, 0), 5)]
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
                            "Eres experto en seguridad pública colombiana. Análisis breve (máx 130 palabras) con bullets y emojis.",
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
                    <div style="font-size:12px;color:#A78BFA;line-height:1.7;">Haz clic en cualquier departamento de la lista para ver el análisis detallado.</div>
                </div>""", unsafe_allow_html=True)

# ── VIAJE SEGURO ──────────────────────────────────────────────────────────────
elif "✈️" in page:
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#ECFDF5;
            border:1px solid #A7F3D0;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#059669;font-weight:700;margin-bottom:12px;">✈️ PLANIFICACIÓN DE VIAJE SEGURO</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 6px;letter-spacing:-0.5px;">Viaje Seguro</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">Consulta el nivel de seguridad de cualquier departamento antes de viajar.</p>
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        dep_viaje = st.selectbox("🗺️ Selecciona el departamento de destino:", DEPARTAMENTOS, key="dep_viaje")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analizar_btn = st.button("🔍 Analizar Destino", type="primary", use_container_width=True, key="viaje_btn")

    if analizar_btn or st.session_state.get("viaje_result"):
        if analizar_btn:
            data = CRIME_DATA.get(dep_viaje, {"score": 2.8, "zona": "BAJO", "gravedad": "BAJO", "municipios": 10})
            muns = get_municipios(dep_viaje)
            st.session_state["viaje_result"] = {"dep": dep_viaje, "data": data, "muns": muns}
            st.session_state.pop("viaje_tips", None)

        vr = st.session_state.get("viaje_result")
        if vr:
            score = vr["data"]["score"]
            if score <= 2.0:
                safety = {"label": "Seguro", "color": "#059669", "bg": "linear-gradient(135deg,#ECFDF5,#D1FAE5)", "icon": "🟢", "stars": 5}
            elif score <= 3.0:
                safety = {"label": "Precaución", "color": "#F59E0B", "bg": "linear-gradient(135deg,#FFFBEB,#FEF3C7)", "icon": "🟡", "stars": 3}
            elif score <= 4.0:
                safety = {"label": "Riesgo Medio", "color": "#EF4444", "bg": "linear-gradient(135deg,#FEF2F2,#FEE2E2)", "icon": "🟠", "stars": 2}
            else:
                safety = {"label": "Alto Riesgo", "color": "#DC2626", "bg": "linear-gradient(135deg,#FEF2F2,#FECDD3)", "icon": "🔴", "stars": 1}

            stars_html = "".join([f'<span style="font-size:20px;color:{"#F59E0B" if i<safety["stars"] else "#E2E8F0"};">★</span>' for i in range(5)])
            st.markdown(f"""<div style="background:{safety['bg']};border:2px solid {safety['color']}30;
                border-radius:24px;padding:28px 34px;margin-bottom:24px;display:flex;align-items:center;gap:24px;">
                <div style="font-size:56px;">{safety['icon']}</div>
                <div style="flex:1;">
                    <div style="font-size:24px;font-weight:900;color:#1E1B4B;margin-bottom:4px;">{vr['dep']}</div>
                    <div style="font-size:17px;font-weight:700;color:{safety['color']};margin-bottom:10px;">{safety['label']}</div>
                    <div>{stars_html}<span style="font-size:12px;color:#6B7280;margin-left:8px;">índice de seguridad</span></div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:52px;font-weight:900;color:{safety['color']};font-family:Georgia,serif;line-height:1;">{score:.1f}</div>
                    <div style="font-size:12px;color:#6B7280;font-weight:600;">Score / 6.0</div>
                </div>
            </div>""", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div style="font-weight:800;font-size:13px;color:#1E1B4B;margin-bottom:14px;">🏙️ Municipios del Departamento</div>', unsafe_allow_html=True)
                chips = "".join([f'<span style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);color:#5B21B6;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;margin:3px;border:1px solid #C4B5FD;">{m}</span>' for m in vr["muns"]])
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{chips}</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div style="font-weight:800;font-size:13px;color:#1E1B4B;margin-bottom:14px;">⚠️ Riesgo por Tipo de Delito</div>', unsafe_allow_html=True)
                zonas_l       = ["MUY BAJO","BAJO","MEDIO-BAJO","MEDIO-ALTO","ALTO","MUY ALTO"]
                delito_scores = sorted(
                    [{"d": d, "sc": round(vr["data"]["score"] * DELIT_FACTOR.get(d, 1.0), 1)} for d in DELITOS],
                    key=lambda x: x["sc"], reverse=True
                )
                max_sc = delito_scores[0]["sc"] if delito_scores else 1
                for item in delito_scores:
                    z   = zonas_l[min(max(round(item["sc"]) - 1, 0), 5)]
                    cfg = RISK_LEVELS.get(z, {"color": "#888"})
                    st.markdown(f"""<div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <span style="font-size:11px;color:#1E1B4B;font-weight:600;">{item['d']}</span>
                            {risk_badge(z, small=True)}
                        </div>
                        <div style="background:#EDE9FE;border-radius:6px;height:7px;overflow:hidden;">
                            <div style="width:{item['sc']/max_sc*100:.0f}%;height:100%;background:{cfg['color']};border-radius:6px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="font-size:15px;font-weight:800;color:#1E1B4B;margin-bottom:6px;">🤖 Consejos Personalizados con IA para {vr["dep"]}</div>', unsafe_allow_html=True)
            if "viaje_tips" not in st.session_state:
                with st.spinner("✨ Preparando consejos personalizados..."):
                    tips = call_claude(
                        "Eres experta en seguridad para mujeres viajeras en Colombia. Responde en español con bullets y emojis. Secciones: 🛡️ Recomendaciones de seguridad, 🏠 Mejores zonas para alojarse, 🕐 Horarios seguros, 🚗 Transporte recomendado, 📞 Números de emergencia locales. Máx 220 palabras.",
                        f"Consejos para mujer viajando a {vr['dep']}, Colombia. Score de riesgo: {score:.1f}/6.0 (zona: {vr['data']['zona']})."
                    )
                    st.session_state["viaje_tips"] = tips
            tips_text = st.session_state.get("viaje_tips", "")
            st.markdown(f'<div style="font-size:13px;color:#1E1B4B;line-height:1.85;white-space:pre-wrap;background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:14px;padding:18px;border:1px solid #C4B5FD;">{tips_text}</div>', unsafe_allow_html=True)

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
                text-align:center;cursor:pointer;min-height:136px;margin-bottom:16px;box-shadow:0 3px 16px rgba(0,0,0,0.06);">
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
        step   = st.session_state.get("denuncia_step", 1)
        steps  = ["Clasificación", "Descripción", "Opciones y Envío"]
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
            anon   = st.toggle("🔒 Denuncia Anónima (Recomendado)", value=st.session_state.get("d_anon", True), key="d_anon_toggle")
            st.session_state["d_anon"] = anon
            anon_bg    = "linear-gradient(135deg,#ECFDF5,#D1FAE5)" if anon else "linear-gradient(135deg,#F5F3FF,#EDE9FE)"
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
                    d_dep    = st.selectbox("🗺️ Departamento", DEPARTAMENTOS, key="d_dep")
                c3, c4 = st.columns(2)
                with c3:
                    d_fecha = st.date_input("📅 Fecha aproximada", key="d_fecha", value=None)
                with c4:
                    d_hora  = st.time_input("🕐 Hora aproximada", key="d_hora", value=None)
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
            razones  = ["✅ Protege a otras mujeres","✅ Genera registros estadísticos","✅ Activa medidas de protección",
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
                content   = msg['content'].replace('\n', '<br>')
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
                        reply   = call_claude(SARA_SYSTEM, "", history=history)
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
                reply   = call_claude(SARA_SYSTEM, "", history=history)
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
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex;align-items:center;gap:6px;background:#EFF6FF;
            border:1px solid #BFDBFE;border-radius:24px;padding:5px 16px;font-size:11px;
            color:#1D4ED8;font-weight:700;margin-bottom:12px;">🚔 AYUDA CERCANA</div>
        <h1 style="font-size:30px;font-weight:900;color:#1E1B4B;margin:0 0 8px;letter-spacing:-0.5px;">Ayuda Cercana</h1>
        <p style="color:#6B7280;font-size:14px;margin:0;">Encuentra entidades de apoyo con información detallada, cómo llegar y contacto directo.</p>
    </div>
    """, unsafe_allow_html=True)

    city_input = st.text_input("📍 Tu ciudad o barrio:", value="Medellín", key="ayuda_city",
                                placeholder="Ej: Medellín, Bogotá, Cali, Barranquilla...")
    entidades = [
        {"tipo":"Policía","icon":"🚔","nom":"CAI Centro","dir":"Carrera 45 #54-20, Medellín","barrio":"Centro","dist":"0.4 km","color":"#1D4ED8","href":"tel:123","phone":"123","horario":"24/7","desc":"Centro de Atención Inmediata. Atención permanente para denuncias y emergencias policiales.","transporte":"Metro: Prado (5 min caminando) · Bus: múltiples rutas por Cra 45"},
        {"tipo":"Hospital","icon":"🏥","nom":"Hospital General de Medellín","dir":"Calle 24 #29-6, Medellín","barrio":"Bomboná","dist":"1.2 km","color":"#059669","href":"tel:4411227","phone":"4411227","horario":"24/7 Urgencias","desc":"Hospital público con urgencias completas, medicina forense y apoyo psicológico para víctimas de violencia.","transporte":"Bus: rutas por Cll 24 · Metro: Industriales (12 min caminando)"},
        {"tipo":"Fiscalía","icon":"⚖️","nom":"Fiscalía Seccional Medellín","dir":"Calle 44 #52-165, Medellín","barrio":"El Centro","dist":"0.8 km","color":"#7C3AED","href":"https://www.fiscalia.gov.co","phone":"01-8000-919-748","horario":"Lun–Vie 7am–5pm","desc":"Recepción de denuncias penales, medidas de protección y seguimiento a casos de violencia.","transporte":"Metro: Alpujarra (8 min caminando) · Bus: Av. Regional"},
        {"tipo":"Refugio","icon":"🏠","nom":"Casa Refugio Luz y Esperanza","dir":"Dirección confidencial — llama al 155","barrio":"Confidencial","dist":"2.1 km","color":"#D97706","href":"tel:155","phone":"155","horario":"24/7 Disponible","desc":"Alojamiento temporal gratuito y seguro para mujeres víctimas de violencia y sus hijos.","transporte":"Llama al 155 para coordinación de transporte seguro y discreto"},
        {"tipo":"Psicología","icon":"🧠","nom":"CAIVAS - Atención Integral","dir":"Calle 50 #40-20, Medellín","barrio":"Prado","dist":"1.5 km","color":"#8B5CF6","href":"tel:137","phone":"137","horario":"Lun–Sáb 8am–8pm","desc":"Centro de Atención Integral a Víctimas. Psicología gratuita, terapia individual y grupos de apoyo.","transporte":"Bus: Cll 50 (múltiples rutas) · Metro: Prado (10 min caminando)"},
        {"tipo":"Policía","icon":"🚔","nom":"Estación Policía Laureles","dir":"Carrera 81 #30-05, Medellín","barrio":"Laureles","dist":"3.2 km","color":"#1D4ED8","href":"tel:123","phone":"123","horario":"24/7","desc":"Estación de Policía del barrio Laureles. Denuncia y apoyo policial inmediato.","transporte":"Bus: Cra 80 (múltiples rutas) · Metro: Universidad (15 min caminando)"},
        {"tipo":"Hospital","icon":"🏥","nom":"Clínica Las Américas","dir":"Diagonal 75B #2A-80, Medellín","barrio":"Estadio","dist":"2.8 km","color":"#059669","href":"tel:4456600","phone":"4456600","horario":"24/7 Urgencias","desc":"Urgencias completas y medicina forense. Atención prioritaria a víctimas de violencia.","transporte":"Metro: Estadio (8 min caminando) · Bus: Cll 73 (múltiples rutas)"},
        {"tipo":"Fiscalía","icon":"⚖️","nom":"URI Fiscalía 24 Horas","dir":"Calle 57 #45-129, Medellín","barrio":"Niquitao","dist":"0.9 km","color":"#7C3AED","href":"https://www.fiscalia.gov.co","phone":"01-8000-919-748","horario":"24/7 Sin cita previa","desc":"Unidad de Reacción Inmediata. Denuncias penales urgentes las 24 horas, sin necesidad de cita.","transporte":"Metro: Hospital (10 min caminando) · Bus: Cll 57"},
        {"tipo":"Psicología","icon":"🧠","nom":"Comisaría de Familia N°1","dir":"Carrera 52 #48-10, Medellín","barrio":"El Centro","dist":"1.1 km","color":"#8B5CF6","href":"tel:123","phone":"Presencial","horario":"Lun–Vie 8am–5pm","desc":"Medidas de protección familiar, conciliación y apoyo psicosocial integral. Sin costo.","transporte":"Metro: Alpujarra (12 min caminando) · Bus: Cra 52"},
        {"tipo":"Refugio","icon":"🏠","nom":"Casa de Acogida ICBF","dir":"Dirección confidencial — Línea 141","barrio":"Confidencial","dist":"3.5 km","color":"#D97706","href":"tel:141","phone":"141 ICBF","horario":"24/7","desc":"Casa de acogida para mujeres y niños en situación de violencia intrafamiliar.","transporte":"Llama al 141 (ICBF) para información de acceso seguro"},
    ]

    filter_tipo = st.selectbox("🔍 Filtrar por tipo de entidad:", ["Todos","Policía","Hospitales","Fiscalía","Refugios","Psicología"], key="ayuda_filter")
    filter_map  = {"Todos":"Todos","Policía":"Policía","Hospitales":"Hospital","Fiscalía":"Fiscalía","Refugios":"Refugio","Psicología":"Psicología"}
    filtered    = [e for e in entidades if filter_tipo == "Todos" or e["tipo"] == filter_map[filter_tipo]]

    st.markdown(f'<div style="font-size:12px;color:#6B7280;margin-bottom:16px;"><strong style="color:#1E1B4B;">{len(filtered)}</strong> lugares de apoyo cerca de <strong style="color:#1E1B4B;">{city_input}</strong></div>', unsafe_allow_html=True)

    col_grid, col_ent = st.columns([2, 1])
    with col_grid:
        gcols = st.columns(3)
        for i, e in enumerate(filtered):
            with gcols[i % 3]:
                is_sel = st.session_state.get("selected_entity") == e["nom"]
                border = f"2px solid {e['color']}" if is_sel else "1.5px solid #EDE9FE"
                shadow = f"0 6px 24px {e['color']}28" if is_sel else "0 2px 12px rgba(109,40,217,0.07)"
                st.markdown(f"""<div style="background:#fff;border:{border};border-radius:20px;
                    padding:20px;margin-bottom:14px;box-shadow:{shadow};">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                        <div style="background:linear-gradient(135deg,{e['color']}18,{e['color']}0D);border-radius:14px;
                            width:48px;height:48px;display:flex;align-items:center;justify-content:center;font-size:24px;">{e['icon']}</div>
                        <div style="background:{e['color']}14;color:{e['color']};font-size:9px;font-weight:800;
                            padding:4px 10px;border-radius:20px;text-transform:uppercase;">{e['tipo']}</div>
                    </div>
                    <div style="font-weight:800;font-size:13px;color:#1E1B4B;margin-bottom:5px;">{e['nom']}</div>
                    <div style="font-size:10px;color:#A78BFA;margin-bottom:4px;">📍 {e['barrio']}</div>
                    <div style="font-size:11px;color:#6B7280;margin-bottom:10px;line-height:1.5;">{e['dir']}</div>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;">
                        <div style="background:#ECFDF5;color:#059669;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;">🚶 {e['dist']}</div>
                        <div style="background:#EFF6FF;color:#1D4ED8;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;">🕐 {e['horario']}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"{'✓ Seleccionado' if is_sel else 'Ver detalles'}", key=f"ent_{e['nom']}", use_container_width=True,
                             type="primary" if is_sel else "secondary"):
                    if is_sel:
                        st.session_state.pop("selected_entity", None)
                    else:
                        st.session_state["selected_entity"] = e["nom"]
                    st.rerun()

    with col_ent:
        sel_nom = st.session_state.get("selected_entity")
        sel_e   = next((e for e in entidades if e["nom"] == sel_nom), None)
        if sel_e:
            st.markdown(f"""<div class="sh-card" style="position:sticky;top:20px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;">
                    <div style="background:linear-gradient(135deg,{sel_e['color']}18,{sel_e['color']}0D);border-radius:16px;
                        width:56px;height:56px;display:flex;align-items:center;justify-content:center;font-size:28px;">{sel_e['icon']}</div>
                    <div style="background:{sel_e['color']}14;color:{sel_e['color']};font-size:9px;font-weight:800;
                        padding:5px 12px;border-radius:20px;text-transform:uppercase;">{sel_e['tipo']}</div>
                </div>
                <div style="font-size:19px;font-weight:900;color:#1E1B4B;margin-bottom:6px;">{sel_e['nom']}</div>
                <p style="font-size:12px;color:#6B7280;line-height:1.7;margin-bottom:18px;">{sel_e['desc']}</p>""",
                unsafe_allow_html=True)
            for label, val in [("📍 Dirección", sel_e['dir']),("🏘️ Barrio", sel_e['barrio']),("🕐 Horario", sel_e['horario']),("📞 Contacto", sel_e['phone'])]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:9px 12px;
                    background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:12px;margin-bottom:8px;">
                    <span style="font-size:11px;color:#A78BFA;font-weight:600;">{label}</span>
                    <span style="font-size:11px;font-weight:800;color:#1E1B4B;text-align:right;max-width:55%;">{val}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:linear-gradient(135deg,#EFF6FF,#DBEAFE);border-radius:14px;
                padding:12px 14px;margin-bottom:14px;border:1px solid #BFDBFE;">
                <div style="font-size:11px;font-weight:800;color:#1D4ED8;margin-bottom:5px;">🚌 Cómo llegar desde {city_input}</div>
                <div style="font-size:11px;color:#1E40AF;line-height:1.7;">{sel_e['transporte']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<a href="{sel_e["href"]}" target="{"_blank" if sel_e["href"].startswith("http") else "_self"}" style="text-decoration:none;display:block;margin-bottom:8px;">'
                       f'<div style="width:100%;background:linear-gradient(135deg,{sel_e["color"]},{sel_e["color"]}CC);color:#fff;border-radius:14px;'
                       f'padding:13px;text-align:center;font-size:13px;font-weight:800;'
                       f'box-shadow:0 4px 14px {sel_e["color"]}44;">{"📞 Llamar: "+sel_e["phone"] if sel_e["href"].startswith("tel:") else "🌐 Visitar sitio web"}</div></a>',
                       unsafe_allow_html=True)
            maps_url = f"https://www.google.com/maps/dir/?api=1&destination={sel_e['dir'].replace(' ', '+')}&travelmode=transit"
            st.markdown(f'<a href="{maps_url}" target="_blank" style="text-decoration:none;display:block;margin-bottom:8px;">'
                       f'<div style="width:100%;background:#fff;color:#1D4ED8;border:1.5px solid #BFDBFE;border-radius:14px;'
                       f'padding:11px;text-align:center;font-size:12px;font-weight:700;">🗺️ Ruta en transporte público</div></a>',
                       unsafe_allow_html=True)
            if st.button(f"🤖 Instrucciones detalladas con IA", key="directions_btn", use_container_width=True):
                with st.spinner("Calculando mejor ruta..."):
                    dir_text = call_claude(
                        "Experto en transporte urbano de Colombia. Instrucciones claras con bullets y emojis. Incluye: TransMilenio/Metro/BRT, taxi/app, a pie. Máx 120 palabras.",
                        f"¿Cómo llegar desde el centro de {city_input} hasta {sel_e['nom']} en {sel_e['dir']}, barrio {sel_e.get('barrio','')}, distancia ~{sel_e['dist']}?"
                    )
                    st.session_state[f"dir_{sel_e['nom']}"] = dir_text
            dir_r = st.session_state.get(f"dir_{sel_e['nom']}", "")
            if dir_r:
                st.markdown(f"""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:14px;
                    padding:14px 16px;border:1px solid #C4B5FD;margin-top:4px;">
                    <div style="font-size:11px;font-weight:800;color:#5B21B6;margin-bottom:8px;">🧭 Instrucciones de SARA</div>
                    <div style="font-size:11px;color:#1E1B4B;line-height:1.8;white-space:pre-wrap;">{dir_r}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:linear-gradient(135deg,#F5F3FF,#EDE9FE);border-radius:20px;
                border:2px dashed #C4B5FD;padding:48px 24px;text-align:center;">
                <div style="font-size:48px;margin-bottom:14px;">🚔</div>
                <div style="font-size:15px;font-weight:700;color:#5B21B6;margin-bottom:6px;">Selecciona una entidad</div>
                <div style="font-size:12px;color:#A78BFA;line-height:1.7;">Haz clic en "Ver detalles" para ver información completa, cómo llegar e instrucciones con IA.</div>
            </div>""", unsafe_allow_html=True)

    tips_ayuda = ["📱 Google Maps: busca 'Comisaría de Familia + tu ciudad'",
                  "📞 Línea 155 te orienta al refugio más cercano 24/7",
                  "🚔 Estación de Policía más cercana: denuncia inmediata",
                  "🏥 Todo hospital debe atenderte en urgencias sin costo",
                  "📋 Fiscalía URI: denuncias urgentes 24h sin cita previa",
                  "🏠 ICBF Línea 141: protección familiar y menores"]
    t_cols = st.columns(2)
    for i, tip in enumerate(tips_ayuda):
        with t_cols[i % 2]:
            st.markdown(f"""<div style="font-size:12px;color:#1E1B4B;line-height:1.7;padding:11px 14px;background:#fff;
                border-radius:14px;border:1px solid #EDE9FE;margin-bottom:8px;
                box-shadow:0 2px 8px rgba(109,40,217,0.05);">{tip}</div>""", unsafe_allow_html=True)

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










