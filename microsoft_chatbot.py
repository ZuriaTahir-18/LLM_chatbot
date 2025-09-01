import streamlit as st
import re
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
VALID_METRICS = ["Total Revenue", "Net Income", "Total Assets", "Total Liabilities", "Cash Flow"]

SYNONYMS = {
    "Total Revenue": {"revenue", "sales", "turnover", "income from sales"},
    "Net Income": {"net income", "profit", "earnings", "net profit"},
    "Total Assets": {"assets", "total assets", "asset"},
    "Total Liabilities": {"liabilities", "debt", "debts", "obligations"},
    "Cash Flow": {"cash flow", "cashflow", "operating cash flow", "cash"},
}

# ----------------- Parsing helpers -----------------
def extract_companies_basic(query: str):
    return [c for c in VALID_COMPANIES if c.lower() in query.lower()]

def extract_years_basic(query: str):
    years = re.findall(r"\b(20\d{2})\b", query)
    return [int(y) for y in years]

def extract_horizon(query: str):
    m = re.search(r"next\s+(\d+)\s+year", query.lower())
    return int(m.group(1)) if m else None

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
            if a > b: a, b = b, a
            return list(range(a, b + 1))
    return None

def extract_metrics_basic(query: str):
    low = query.lower()
    chosen = []
    for metric, keys in SYNONYMS.items():
        if any(k in low for k in keys) or metric.lower() in low:
            chosen.append(metric)
    return list(dict.fromkeys(chosen))

# ----------------- Data utilities -----------------
def to_millions(value):
    return float(value) / 1e6 if value else None

def get_company_year_df(companies, years, metrics):
    rows = [r for r in financial_data if r["Company"] in companies and (not years or r["Year"] in years)]
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["Company", "Year"] + metrics
    df = df[keep]
    for m in metrics:
        df[m] = df[m].map(to_millions)
    return df.sort_values(["Company", "Year"]).reset_index(drop=True)

def add_serial_column(df: pd.DataFrame, reorder_for_forecast: bool = False) -> pd.DataFrame:
    df2 = df.reset_index(drop=True).copy()
    df2.insert(0, "S.No", range(1, len(df2) + 1))  # start from 1
    if reorder_for_forecast:
        desired = ["S.No", "Year", "Metric", "Value", "Company", "Type"]
        existing = [c for c in desired if c in df2.columns]
        df2 = df2[existing]
    return df2

# ----------------- Forecasting -----------------
from sklearn.linear_model import LinearRegression

def forecast_linear(df: pd.DataFrame, company: str, metric: str, horizon: int = 2, years_requested: list = None):
    if df.empty: return None, None
    full_hist = df[df["Company"] == company][["Year", metric]].dropna().copy()
    if len(full_hist) < 2: return None, None

    X, y = full_hist[["Year"]].values, full_hist[metric].values
    model = LinearRegression().fit(X, y)
    last_year = int(full_hist["Year"].max())

    hist_years = sorted(full_hist["Year"].tolist())

    if years_requested:
        future_years = sorted([y for y in years_requested if y > last_year])
    else:
        future_years = list(range(last_year + 1, last_year + 1 + horizon))

    future_years = [y for y in future_years if y <= 2034]

    y_pred = model.predict(np.array(future_years).reshape(-1, 1)) if future_years else []

    hist_list = [{"Company": company, "Year": int(y), "Metric": metric,
                  "Value": full_hist.set_index("Year")[metric][y], "Type": "Actual"}
                 for y in hist_years]

    fut_list = [{"Company": company, "Year": int(y), "Metric": metric,
                 "Value": float(pred), "Type": "Forecast"}
                for y, pred in zip(future_years, y_pred)]

    combo = pd.DataFrame(hist_list + fut_list)

    chart = (
        alt.Chart(combo)
        .mark_line(point=True)
        .encode(
            x="Year:O", y="Value:Q", color="Type:N",
            tooltip=["Company", "Metric", "Year", alt.Tooltip("Value:Q", title="Value (mn)")]
        )
        .properties(title=f"{company} — {metric} (mn): Actual vs Forecast")
    )
    return combo, chart

# ----------------- Chatbot -----------------
def parse_query(query: str):
    clean = correct_spelling(query)
    clean_low = clean.lower()

    yrs_range = extract_year_range(clean_low)
    yrs_text = yrs_range if yrs_range else extract_years_basic(clean_low)
    horizon_text = extract_horizon(clean_low)
    comps_text = extract_companies_basic(clean)
    mets_text = extract_metrics_basic(clean)

    max_data_year = 2024
    MAX_FORECAST_YEAR = 2034

    # Handle years beyond available forecast
    if yrs_text:
        if max(yrs_text) > MAX_FORECAST_YEAR:
            return [], [], [], False, 0, f"⚠️ Sorry, I can only forecast up to {MAX_FORECAST_YEAR}."
        if min(yrs_text) < 2022:
            return [], [], [], False, 0, "⚠️ Sorry, I only have data from 2022 onwards."

    forecast_flag = "forecast" in clean_low or "predict" in clean_low or "next" in clean_low
    if yrs_text and max(yrs_text) > max_data_year:
        forecast_flag = True
        horizon_text = max(yrs_text) - max_data_year

    comps_text = extract_companies_basic(clean)
    if not comps_text:
        return [], [], [], False, 0, "⚠️ Sorry, I only have data for Microsoft, Tesla, and Apple."

    horizon = horizon_text or 2 if forecast_flag else 0
    return comps_text, yrs_text, mets_text, forecast_flag, horizon, None

def respond(query: str):
    comps, yrs, mets, do_forecast, horizon, error_msg = parse_query(query)
    if error_msg: return error_msg, None, None
    if not mets: return "⚠️ Please specify revenue, net income, assets, liabilities, or cash flow.", None, None
    if not comps: return "⚠️ Please mention Microsoft, Tesla, or Apple.", None, None

    if do_forecast:
        charts, results = [], []
        for comp in comps:
            for metric in mets:
                df_all = get_company_year_df([comp], None, [metric])
                combo, fchart = forecast_linear(df_all, comp, metric, horizon, years_requested=yrs)
                if combo is not None and not combo.empty:
                    charts.append(fchart)
                    results.append(combo)
        if results:
            final_df = add_serial_column(pd.concat(results, ignore_index=True), reorder_for_forecast=True)
            return final_df, alt.vconcat(*charts), None
        return "No forecast generated.", None, None

    df = get_company_year_df(comps, yrs, mets)
    if df.empty: return "No data found.", None, None
    df_out = add_serial_column(df, reorder_for_forecast=False)

    melt = df.melt(id_vars=["Company", "Year"], value_vars=mets, var_name="Metric", value_name="Value")
    chart = alt.Chart(melt).mark_bar().encode(
        x="Year:O", y="Value:Q", color="Company:N" if len(comps) > 1 else "Metric:N",
        tooltip=["Company", "Metric", "Year", alt.Tooltip("Value:Q", title="Value (mn)")]
    ).properties(title=f"{', '.join(comps)} — {', '.join(mets)}")
    return df_out, chart, None

# ----------------- UI -----------------
st.set_page_config(page_title="💬 Financial Chatbot", page_icon="💹", layout="wide")
st.title("💹 AI-Powered Financial Data Chatbot")

st.markdown(
    """
    🚀 **Your Smart Finance Assistant**  
    Ask me about **Revenue, Net Income, Assets, Liabilities, or Cash Flow** for  
    **Microsoft, Tesla, and Apple (2022–2024)** 📊  

    🔥 Features:  
    - **Compare companies instantly** → *"Compare Tesla and Apple revenue"*  
    - **Forecast up to 2034** → *"Predict Apple assets in 2030"*  
    - **Custom ranges** → *"Apple net income from 2023 to 2026"*  
    - **Multi-metrics** → *"Tesla revenue and liabilities next 3 years"*  
    - **Spelling correction** → *"aape revnue 2024"* ✅  

    ⚡ All values are in **millions**. Future years are predicted using **AI regression**.
    """
)

if "history" not in st.session_state: st.session_state.history = []

for q, r in st.session_state.history:
    st.markdown(f"**🧑 You:** {q}")
    if isinstance(r, tuple):
        df, chart, note = r
        if isinstance(df, pd.DataFrame): st.dataframe(df, use_container_width=True)
        if chart is not None: st.altair_chart(chart, use_container_width=True)
        if note: st.info(note)
    else: st.warning(r)

query = st.chat_input("💡 Ask your question here…")
if query:
    answer = respond(query)
    st.session_state.history.append((query, answer))
    st.rerun()
