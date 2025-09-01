import streamlit as st
import re
import json
import pandas as pd
import numpy as np
import altair as alt
from spellchecker import SpellChecker


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

# ----------------- Basic parsing -----------------
def extract_companies_basic(query: str):
    found = []
    low = query.lower()
    for c in VALID_COMPANIES:
        if c.lower() in low:
            found.append(c)
    return list(dict.fromkeys(found))


def extract_years_basic(query: str):
    years = re.findall(r"\b(20\d{2})\b", query)
    return [int(y) for y in years]


def extract_year_range(query: str):
    query = query.lower()
    patterns = [
        r"from\s+(20\d{2})\s+(?:to|through|-)\s+(20\d{2})",
        r"between\s+(20\d{2})\s+and\s+(20\d{2})",
        r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b",
    ]
    for p in patterns:
        m = re.search(p, query)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            return list(range(a, b + 1))
    return None


# ----------------- LLM-powered parsing -----------------
def parse_with_llm(query: str):
    if LLM is None:
        return None
    try:
        out = LLM(query, max_new_tokens=128)[0]["generated_text"].strip()
        return json.loads(out)
    except Exception:
        return None


# ----------------- Data utilities -----------------
def to_millions(value):
    if value is None:
        return None
    try:
        return float(value) / 1e6
    except Exception:
        return None


def get_company_year_df(companies, years, metrics):
    rows = [r for r in financial_data if r["Company"] in companies and (not years or r["Year"] in years)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["Company", "Year"] + metrics
    df = df[keep]
    for m in metrics:
        df[m] = df[m].map(to_millions)
    return df.sort_values(["Company", "Year"]).reset_index(drop=True)


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.reset_index(drop=True).copy()
    df2.insert(0, "S.No", range(1, len(df2) + 1))
    return df2


# ----------------- Company & Year Validation -----------------
def detect_unsupported_companies(query: str, comps_found: list):
    low = query.lower()
    for word in re.findall(r"\b[a-zA-Z]+\b", low):
        if word.lower() not in VALID_COMPANIES_LOWER and word.lower().title() not in comps_found:
            if word.lower() in {"amazon", "google", "meta"}:  # obvious company names
                return f"⚠️ Sorry — I only have data for Microsoft, Tesla, and Apple. You asked about: {word.title()}."
    return None


# ----------------- Chatbot core -----------------
def parse_query(query: str):
    clean = correct_spelling(query)
    yrs_range = extract_year_range(clean)
    yrs = yrs_range or extract_years_basic(clean)

    # Company & metric extraction
    comps = extract_companies_basic(clean)
    mets = [m for m in VALID_METRICS if m.lower() in clean.lower()]

    # Validation
    unsupported_msg = detect_unsupported_companies(clean, comps)
    if unsupported_msg:
        return [], [], [], False, 0, unsupported_msg

    # Year > 2034 check
    if yrs and max(yrs) > 2034:
        return [], [], [], False, 0, "⚠️ I can only provide forecasts up to 2034."

    forecast_flag = any(y > max(r["Year"] for r in financial_data) for y in yrs) if yrs else False
    horizon = (max(yrs) - max(r["Year"] for r in financial_data)) if forecast_flag else 0

    return comps, yrs, mets, forecast_flag, horizon, None


def respond(query: str):
    comps, yrs, mets, do_forecast, horizon, error_msg = parse_query(query)

    if error_msg:
        return error_msg, None, None
    if not mets:
        return "⚠️ I can provide: revenue, net income, assets, liabilities, or cash flow.", None, None
    if not comps:
        return "⚠️ Please mention at least one company (Microsoft, Tesla, or Apple).", None, None

    df = get_company_year_df(comps, yrs, mets)
    if df.empty:
        return "No data found for your filters.", None, None

    df_out = add_serial_column(df)
    melt = df.melt(id_vars=["Company", "Year"], value_vars=mets, var_name="Metric", value_name="Value")
    chart = (
        alt.Chart(melt)
        .mark_bar()
        .encode(
            x="Year:O",
            y="Value:Q",
            color="Metric:N",
            tooltip=["Year", "Metric", alt.Tooltip("Value:Q", title="Value (mn)")],
        )
    )
    return df_out, chart, None


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
            st.dataframe(df, use_container_width=True, hide_index=True)  # ✅ hide default 0 index
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning(r)

query = st.chat_input("💡 Ask your question here…")
if query:
    answer = respond(query)
    st.session_state.history.append((query, answer))
    st.rerun()
