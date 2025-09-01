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
    if len(years) == 2:
        a, b = int(years[0]), int(years[1])
        return list(range(min(a, b), max(a, b) + 1))
    return [int(y) for y in years]

def extract_metrics_basic(query: str):
    low = query.lower()
    chosen = []
    for metric, keys in SYNONYMS.items():
        if any(k in low for k in keys):
            chosen.append(metric)
    for m in VALID_METRICS:
        if m.lower() in low and m not in chosen:
            chosen.append(m)
    return list(dict.fromkeys(chosen))

# ----------------- Forecasting -----------------
from sklearn.linear_model import LinearRegression

def forecast_linear(df: pd.DataFrame, company: str, metric: str, horizon: int = 2):
    sub = df[df["Company"] == company][["Year", metric]].dropna()
    if len(sub) < 2:
        return None
    X, y = sub[["Year"]].values, sub[metric].values
    model = LinearRegression().fit(X, y)
    last_year = int(sub["Year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)
    y_pred = model.predict(future_years.reshape(-1, 1))

    hist = sub.rename(columns={metric: "Value"}).assign(Type="Actual", Metric=metric, Company=company)
    fut = pd.DataFrame({
        "Company": company,
        "Year": future_years,
        "Value": y_pred,
        "Type": "Forecast",
        "Metric": metric,
    })
    return pd.concat([hist, fut], ignore_index=True)

# ----------------- Respond -----------------
def respond(query: str):
    comps, yrs, mets, forecast, horizon = extract_companies_basic(query), extract_years_basic(query), extract_metrics_basic(query), ("forecast" in query.lower()), 2
    if not comps or not mets:
        return "⚠️ Please specify at least one company and one metric.", None, None

    results, charts = [], []
    if forecast:
        for comp in comps:
            for metric in mets:
                df = pd.DataFrame([r for r in financial_data if r["Company"] == comp])
                if df.empty:
                    continue
                combo = forecast_linear(df, comp, metric, horizon)
                if combo is not None:
                    results.append(combo)
                    chart = alt.Chart(combo).mark_line(point=True).encode(
                        x="Year:O", y="Value:Q", color="Type:N", tooltip=["Company", "Metric", "Year", "Value"]
                    ).properties(title=f"{comp} — {metric}")
                    charts.append(chart)
        if results:
            return pd.concat(results, ignore_index=True), alt.vconcat(*charts), None
    return "⚠️ No forecast generated.", None, None

# ----------------- UI -----------------
st.set_page_config(page_title="Financial Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Data Chatbot — Forecasting")

query = st.chat_input("Ask your question…")
if query:
    df, chart, note = respond(query)
    if isinstance(df, pd.DataFrame):
        st.dataframe(df, use_container_width=True)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    if note:
        st.info(note)
