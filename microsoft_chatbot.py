import streamlit as st
import re
import json
import pandas as pd
import numpy as np
import altair as alt

# --- Optional: free LLM via Hugging Face (flan-t5-small) ---
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# --- Spell checker ---
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    _HAS_SPELL = True
except Exception:
    _HAS_SPELL = False


def correct_spelling(text: str) -> str:
    if not _HAS_SPELL:
        return text
    corrected = []
    for word in text.split():
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return " ".join(corrected)


@st.cache_resource(show_spinner=False)
def get_llm():
    if not _HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception:
        return None


LLM = get_llm()

# ----------------- Financial Data -----------------
financial_data = [
    {"Company": "Microsoft", "Year": 2022, "Total Revenue": 1.98E11, "Net Income": 72738000000, "Total Assets": 3.65E11,
     "Total Liabilities": 1.98E11, "Cash Flow": 89035000000},
    {"Company": "Microsoft", "Year": 2023, "Total Revenue": 2.12E11, "Net Income": 72361000000, "Total Assets": 4.12E11,
     "Total Liabilities": 2.06E11, "Cash Flow": 87582000000},
    {"Company": "Microsoft", "Year": 2024, "Total Revenue": 2.45E11, "Net Income": 88136000000, "Total Assets": 5.12E11,
     "Total Liabilities": 2.44E11, "Cash Flow": 1.19E11},
    {"Company": "Tesla", "Year": 2022, "Total Revenue": 81462000000, "Net Income": 12587000000, "Total Assets": 82338000000,
     "Total Liabilities": 36440000000, "Cash Flow": 14724000000},
    {"Company": "Tesla", "Year": 2023, "Total Revenue": 96773000000, "Net Income": 14973000000, "Total Assets": 1.07E11,
     "Total Liabilities": 43009000000, "Cash Flow": 13256000000},
    {"Company": "Tesla", "Year": 2024, "Total Revenue": 97690000000, "Net Income": 7153000000, "Total Assets": 1.22E11,
     "Total Liabilities": 48390000000, "Cash Flow": 14923000000},
    {"Company": "Apple", "Year": 2022, "Total Revenue": 3.94E11, "Net Income": 99803000000, "Total Assets": 3.53E11,
     "Total Liabilities": 3.02E11, "Cash Flow": 1.22E11},
    {"Company": "Apple", "Year": 2023, "Total Revenue": 3.83E11, "Net Income": 96995000000, "Total Assets": 3.53E11,
     "Total Liabilities": 2.90E11, "Cash Flow": 1.11E11},
    {"Company": "Apple", "Year": 2024, "Total Revenue": 3.91E11, "Net Income": 93736000000, "Total Assets": 3.65E11,
     "Total Liabilities": 3.08E11, "Cash Flow": 1.18E11},
]

VALID_COMPANIES = {"Microsoft", "Tesla", "Apple"}
VALID_COMPANIES_LOWER = {c.lower() for c in VALID_COMPANIES}
VALID_METRICS = [
    "Total Revenue",
    "Net Income",
    "Total Assets",
    "Total Liabilities",
    "Cash Flow",
]

SYNONYMS = {
    "Total Revenue": {"revenue", "sales", "turnover", "income from sales"},
    "Net Income": {"net income", "profit", "earnings", "net profit"},
    "Total Assets": {"assets", "total assets", "asset"},
    "Total Liabilities": {"liabilities", "debt", "debts", "obligations"},
    "Cash Flow": {"cash flow", "cashflow", "operating cash flow", "cash"},
}

# build metric phrase list for regex checks
_metric_phrases = set()
for k, syns in SYNONYMS.items():
    _metric_phrases.add(k.lower())
    for s in syns:
        _metric_phrases.add(s.lower())
metric_phrase_list = sorted(_metric_phrases, key=lambda x: -len(x))
metric_regex = r"(?:" + r"|".join(re.escape(p) for p in metric_phrase_list) + r")"

# ----------------- Basic parsing -----------------
def extract_companies_basic(query: str):
    found = []
    low = query.lower()
    for c in VALID_COMPANIES:
        if c.lower() in low:
            found.append(c)
    return list(dict.fromkeys(found))

# (other parsing + forecast functions remain unchanged)

# ----------------- Chatbot core -----------------
def respond(query: str):
    comps, yrs, mets, do_forecast, horizon, error_msg = parse_query(query)

    if error_msg:
        return error_msg, None, None

    if not mets:
        return "⚠️ I can provide: revenue, net income, assets, liabilities, or cash flow.", None, None
    if not comps:
        return "⚠️ Please mention at least one company (Microsoft, Tesla, or Apple).", None, None

    # (forecast + multi-company logic stays the same)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Financial Chatbot — LLM + Forecast", page_icon="💬", layout="wide")
st.title("💬 Financial Data Chatbot — LLM + Forecasting")

if "history" not in st.session_state:
    st.session_state.history = []

for q, r in st.session_state.history:
    st.markdown(f"**🧑 You:** {q}")
    if isinstance(r, tuple):
        df, chart, note = r
        if isinstance(df, pd.DataFrame):
            st.dataframe(df.reset_index(drop=True), use_container_width=True)  # ✅ fixed index issue
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        if note:
            st.info(note)
    else:
        st.warning(r)

query = st.chat_input("💡 Ask your question here…")
if query:
    answer = respond(query)
    st.session_state.history.append((query, answer))
    st.rerun()
