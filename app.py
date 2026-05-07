"""
HIKARI · HIMARI · MACHI — Complaint Driver Classification Pipeline
Hugging Face Spaces · Streamlit Inference App

MACHI  — Masking Algorithm for Contextual & Heuristic Identification
HIMARI — Heuristic Input Mapping & Array Reduction Interface
HIKARI — High-dimensional Inference & Key-driver Analysis Routing Intelligence
"""

import os
import re
import pickle
import joblib
import numpy as np
import nltk
import streamlit as st
from typing import Set

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="HIKARI · Complaint Driver Classifier",
    page_icon="✦",
    layout="centered",
)

# ── Download NLTK data (cached at container level) ───────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)

download_nltk()
from nltk.corpus import stopwords as nltk_stopwords

# ── Artifact loading ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """
    Load all three pipeline artefacts once and hold them in memory.
    Expected to be co-located with app.py in the Space root.
    """
    from tensorflow.keras.models import load_model
    from tensorflow.keras.backend import int_shape

    hikari_model = load_model("keras-pkl/hikari_v2.keras", compile=False)
    crf_model    = joblib.load("keras-pkl/machi_v2.pkl")

    with open("keras-pkl/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_seq_len = int_shape(hikari_model.input)[1]   # auto-detects 196
    return hikari_model, crf_model, tokenizer, max_seq_len

hikari_model, crf_model, tokenizer, MAX_SEQUENCE_LENGTH = load_artifacts()

# ── Stop words ───────────────────────────────────────────────────────────────
STOP_WORDS: Set[str] = set(nltk_stopwords.words("english")).union({"water"})

# ── Complaint driver labels ───────────────────────────────────────────────────
TOPICS = [
    "Customer Service",
    "Digital",
    "Online Experience",
    "Outages and Faults",
    "Process",
    "Reputation",
    "Sustainability",
    "Trust",
    "Value for Money",
    "Vulnerability & FDV",
]

# ── MACHI helper functions (CRF feature engineering) ─────────────────────────

def tokenize_with_offsets(text: str):
    pattern = r"\w+|[^\w\s]"
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text)]


def word2features(tokens, i):
    w = tokens[i]
    feats = {
        "bias": 1.0,
        "word.lower": w.lower(),
        "word[-3:]": w[-3:],
        "word.isupper": w.isupper(),
        "word.istitle": w.istitle(),
        "word.isdigit": w.isdigit(),
    }
    if i > 0:
        p = tokens[i - 1]
        feats.update({
            "-1:word.lower": p.lower(),
            "-1:word.istitle": p.istitle(),
            "-1:word.isupper": p.isupper(),
        })
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        n = tokens[i + 1]
        feats.update({
            "+1:word.lower": n.lower(),
            "+1:word.istitle": n.istitle(),
            "+1:word.isupper": n.isupper(),
        })
    else:
        feats["EOS"] = True
    return feats


def sent2features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]


# Australia Post street types — used by the address regex pre-pass
_STREET_TYPES = [
    "ALLY","ALLEY","ALWY","ALLEYWAY","AMBL","AMBLE","ANCG","ANCHORAGE","APP","APPROACH",
    "ARC","ARCADE","ART","ARTERY","AVE","AVENUE","BASN","BASIN","BCH","BEACH","BEND",
    "BLK","BLOCK","BVD","BOULEVARD","BRCE","BRACE","BRAE","BRK","BREAK","BDGE","BRIDGE",
    "BDWY","BROADWAY","BROW","BYPA","BYPASS","BYWY","BYWAY","CAUS","CAUSEWAY","CTR","CENTRE",
    "CNWY","CENTREWAY","CH","CHASE","CIR","CIRCLE","CLT","CIRCLET","CCT","CIRCUIT",
    "CRCS","CIRCUS","CL","CLOSE","CLDE","COLONNADE","CMMN","COMMON","CON","CONCOURSE",
    "CPS","COPSE","CNR","CORNER","CSO","CORSO","CT","COURT","CTYD","COURTYARD","COVE",
    "CRES","CRESCENT","CRST","CREST","CRSS","CROSS","CRSG","CROSSING","CRD","CROSSROAD",
    "COWY","CROSSWAY","CUWY","CRUISEWAY","CDS","CUL-DE-SAC","CTTG","CUTTING","DALE",
    "DELL","DEVN","DEVIATION","DIP","DSTR","DISTRIBUTOR","DR","DRIVE","DRWY","DRIVEWAY",
    "EDGE","ELB","ELBOW","END","ENT","ENTRANCE","ESP","ESPLANADE","EST","ESTATE",
    "EXP","EXPRESSWAY","EXTN","EXTENSION","FAWY","FAIRWAY","FTRK","FIRE","FITR","FIRETRAIL",
    "FLAT","FOLW","FOLLOW","FTWY","FOOTWAY","FSHR","FORESHORE","FORM","FORMATION",
    "FWY","FREEWAY","FRNT","FRONT","FRTG","FRONTAGE","GAP","GDN","GARDEN","GTE","GATE",
    "GDNS","GARDENS","GTES","GATES","GLD","GLADE","GLEN","GRA","GRANGE","GRN","GREEN",
    "GRND","GROUND","GR","GROVE","GLY","GULLY","HTS","HEIGHTS","HRD","HIGHROAD",
    "HWY","HIGHWAY","HILL","INTG","INTERCHANGE","INTN","INTERSECTION","JNC","JUNCTION",
    "KEY","LDG","LANDING","LANE","LNWY","LANEWAY","LEES","LINE","LINK","LT","LITTLE",
    "LKT","LOOKOUT","LOOP","LWR","LOWER","MALL","MNDR","MEANDER","MEW","MEWS","MWY","MOTORWAY",
    "MT","MOUNT","NOOK","OTLK","OUTLOOK","PDE","PARADE","PARK","PKLD","PARKLANDS",
    "PKWY","PARKWAY","PART","PASS","PATH","PHWY","PATHWAY","PIAZ","PIAZZA","PL","PLACE",
    "PLAT","PLATEAU","PLZA","PLAZA","PKT","POCKET","PNT","POINT","PORT","PROM","PROMENADE",
    "QUAD","QDGL","QUADRANGLE","QDRT","QUADRANT","QY","QUAY","QYS","QUAYS","RMBL","RAMBLE",
    "RAMP","RNGE","RANGE","RCH","REACH","RES","RESERVE","REST","RTT","RETREAT","RIDE",
    "RDGE","RIDGE","RGWY","RIDGEWAY","RING","RISE","RVR","RIVER","RVWY","RIVERWAY",
    "RVRA","RIVIERA","RD","ROAD","RDS","ROADS","RDSD","ROADSIDE","RDWY","ROADWAY",
    "RNDE","RONDE","RSBL","ROSEBOWL","RTY","ROTARY","RND","ROUND","RTE","ROUTE",
    "ROW","RUE","RUN","SWY","SERVICE","SDNG","SIDING","SLPE","SLOPE","SND","SOUND",
    "SPUR","SQ","SQUARE","STRS","STAIRS","SHWY","STATE","STPS","STEPS","STRA","STRAND",
    "ST","STREET","STRP","STRIP","SBWY","SUBWAY","TARN","TCE","TERRACE","THOR","THOROUGHFARE",
    "TLWY","TOLLWAY","TOP","TOR","TWRS","TOWERS","TRK","TRACK","TRL","TRAIL","TRLR","TRAILER",
    "TRI","TRIANGLE","TKWY","TRUNKWAY","TURN","UPAS","UNDERPASS","UPR","UPPER","VALE",
    "VDCT","VIADUCT","VIEW","VLLS","VILLAS","VSTA","VISTA","WADE","WALK","WKWY","WALKWAY",
    "WAY","WHRF","WHARF","WYND","YARD",
]

STREET_PATTERN = re.compile(
    r"\b\d+\s+[A-Za-z0-9\s]+\s+(?:" + "|".join(_STREET_TYPES) + r")\b",
    flags=re.I,
)


def redact_crf(text: str) -> str:
    """
    MACHI two-pass PII redaction:
    1. Regex pre-pass — deterministic patterns (emails, phones, AU addresses, IDs)
    2. CRF sequential pass — BIO-tagged NER for names and residual entities
    """
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[redacted]", text)
    # Mobile numbers (4-3-3 format)
    text = re.sub(r"\b\d{4}\s\d{3}\s\d{3}\b", "[redacted]", text)
    # Street addresses
    text = STREET_PATTERN.sub("[redacted]", text)
    # City/state/postcode
    text = re.sub(
        r"\b[A-Za-z ]+\s+(VIC|NSW|QLD|SA|WA|TAS|NT|ACT)\s*\d{4}\b",
        "[redacted]", text, flags=re.I,
    )
    # 7-digit IDs
    text = re.sub(r"\b\d{7}\b", "[redacted]", text)
    # Medicare/concession/license formats
    for pat in [r"\b\d{4}\s\d{5}\s\d\b", r"\b\d{3}\s\d{3}\s\d{3}[A-Za-z]\b", r"\b\d{9}\b"]:
        text = re.sub(pat, "[redacted]", text)

    # CRF sequential pass
    toks  = tokenize_with_offsets(text)
    words = [w for w, _, _ in toks]
    feats = sent2features(words)
    tags  = crf_model.predict_single(feats)

    out, i = [], 0
    while i < len(words):
        if tags[i].startswith("B-"):
            while i < len(words) and tags[i].startswith(("B-", "I-")):
                i += 1
            out.append("[redacted]")
        else:
            out.append(words[i])
            i += 1

    return " ".join(out)


def himari_preprocess(text: str, case_origin: str, stop_words: Set[str]) -> str:
    """
    HIMARI channel-aware preprocessing:
    1. Channel extraction  →  2. Normalise  →  3. MACHI PII redaction
    →  4. Stopword removal  →  5. Truncate  →  6. Wrap <start>/<end>
    """
    if case_origin == "Email":
        text = re.sub(r"^(?:From|Sent|To|Subject):.*\r?\n", "", text, flags=re.I | re.M)
        m = re.search(r"(^On .+ wrote:)", text, flags=re.I | re.M)
        if m:
            text = text[: m.start()]
        text = re.split(r"-----Original Message-----", text, flags=re.I)[0]

    elif case_origin == "Web":
        m = re.search(r"Case Description:(.*?)(?=First Name:)", text, flags=re.S | re.I)
        if m:
            text = m.group(1)

    text = text.lower().strip()
    text = redact_crf(text)

    toks     = text.split()
    filtered = [t for t in toks if t not in stop_words]

    max_body = MAX_SEQUENCE_LENGTH - 2
    if len(filtered) > max_body:
        filtered = filtered[:max_body]

    return "<start> " + " ".join(filtered) + " <end>"


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(raw_comment: str, case_origin: str = "Other", threshold: float = 0.5):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned  = himari_preprocess(raw_comment, case_origin, STOP_WORDS)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    probs    = hikari_model.predict(padded, verbose=0)[0]

    results = [(TOPICS[i], float(p)) for i, p in enumerate(probs)]
    return results, cleaned


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    body, .stApp { background-color: #F5F0E8; color: #1A1A1A; }
    h1, h2, h3 { font-family: 'Arial Black', sans-serif; text-transform: uppercase; letter-spacing: 0.04em; }
    .block-container { max-width: 780px; }
    .driver-row { display: flex; align-items: center; margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✦ HIKARI")
st.markdown(
    "**High-dimensional Inference & Key-driver Analysis Routing Intelligence**  \n"
    "Multi-label complaint driver classification for utility sector CX analytics.  \n"
    "*Preprocessing by HIMARI · PII redaction by MACHI*"
)
st.divider()

with st.form("inference_form"):
    verbatim = st.text_area(
        "Customer verbatim",
        placeholder="Paste raw complaint text here — email body, web form submission, or phone-log transcript.",
        height=160,
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        case_origin = st.selectbox(
            "Case origin",
            ["Other", "Email", "Web"],
            help="Tells HIMARI which channel-specific extraction rules to apply before classification.",
        )
    with col2:
        threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Drivers above this sigmoid score are returned as active labels.",
        )
    submitted = st.form_submit_button("✦ Classify", use_container_width=True)

if submitted:
    if not verbatim.strip():
        st.warning("Please enter a verbatim before classifying.")
    else:
        with st.spinner("Running HIMARI → MACHI → HIKARI…"):
            results, cleaned = predict(verbatim, case_origin, threshold)

        # ── Active drivers ──────────────────────────────────────────
        active = [(t, p) for t, p in results if p >= threshold]
        active.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Predicted Drivers")
        if active:
            for topic, prob in active:
                pct = int(prob * 100)
                st.markdown(f"**{topic}**")
                st.progress(prob, text=f"{pct}%")
        else:
            st.info("No drivers exceeded the confidence threshold. Try lowering the slider.")

        # ── All scores ──────────────────────────────────────────────
        with st.expander("All driver scores"):
            all_sorted = sorted(results, key=lambda x: x[1], reverse=True)
            for topic, prob in all_sorted:
                marker = "●" if prob >= threshold else "○"
                st.markdown(f"`{marker}` **{topic}** — {prob:.3f}")

        # ── Cleaned text ────────────────────────────────────────────
        with st.expander("Preprocessed text (post-HIMARI + MACHI redaction)"):
            st.code(cleaned, language=None)

st.divider()
st.caption(
    "MACHI · HIMARI · HIKARI — Complaint Driver Classification Pipeline  \n"
    "Models trained on utility sector CX data. AU-specific PII patterns. "
    "Not intended for use with sensitive production data in this demo environment."
)
