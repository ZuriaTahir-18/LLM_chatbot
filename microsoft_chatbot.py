import streamlit as st
import re
import json
import pandas as pd
import numpy as np
import altair as alt

# --- Optional: free LLM via Hugging Face (flan-t5-small) ---
# This adds light natural-language understanding: synonyms, spelling fixes, JSON parsing.
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
        # Small + free model; downloads on first run (HF Spaces is fine)
        return pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception:
        return None

LLM = get_llm()

# ----------------- Financial Data (same as your original) -----------------
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
VALID_METRICS = [
    "Total Revenue",
    "Net Income",
    "Total Assets",
    "Total Liabilities",
    "Cash Flow",
]

SYNONYMS = {
    # metric → set of synonyms (lowercase)
    "Total Revenue": {"revenue", "sales", "turnover", "income from sales"},
    "Net Income": {"net income", "profit", "earnings", "net profit"},
    "Total Assets": {"assets", "total assets", "asset"},
    "Total Liabilities": {"liabilities", "debt", "debts", "obligations"},
    "Cash Flow": {"cash flow", "cashflow", "operating cash flow", "cash"},
}

# ----------------- Helper: basic (regex/keyword) parsing -----------------
def extract_companies_basic(query: str):
    found = []
    low = query.lower()
    for c in VALID_COMPANIES:
        if c.lower() in low:
            found.append(c)
    return list(dict.fromkeys(found))  # unique, keep order


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
    # If explicitly asking for "metrics" words like revenue/assets/net income etc.
    for m in VALID_METRICS:
        if m.lower() in low and m not in chosen:
            chosen.append(m)
    return list(dict.fromkeys(chosen))


# ----------------- Helper: LLM-powered parsing -----------------
def parse_with_llm(query: str):
    """Try to normalize user query into {companies, years, metrics, forecast} using a tiny free LLM.
    Returns dict or None if parsing fails.
    """
    if LLM is None:
        return None
    instruction = (
        "You extract structured info from a finance question. "
        "Recognize company names among [Microsoft, Apple, Tesla]; "
        "map synonyms to metrics among [Total Revenue, Net Income, Total Assets, Total Liabilities, Cash Flow]; "
        "identify 4-digit years; detect if forecasting is requested (words like forecast/predict/next). "
        "Return STRICT JSON with keys: companies (list), years (list), metrics (list), forecast (bool), horizon (int). "
        "If horizon not given, use 2."
    )
    prompt = f"{instruction}\nQuestion: {query}\nJSON:"
    try:
        out = LLM(prompt, max_new_tokens=128)[0]["generated_text"].strip()
        # Allow cases where model adds text before/after JSON — try to locate JSON
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            out = out[start:end + 1]
        data = json.loads(out)
        # Keep only valid values
        companies = [c for c in data.get("companies", []) if c in VALID_COMPANIES]
        years = [int(y) for y in data.get("years", []) if str(y).isdigit()]
        metrics = [m for m in data.get("metrics", []) if m in VALID_METRICS]
        forecast = bool(data.get("forecast", False))
        horizon = int(data.get("horizon", 2))
        return {
            "companies": companies,
            "years": years,
            "metrics": metrics,
            "forecast": forecast,
            "horizon": max(1, min(horizon, 5)),  # cap to 5 years
        }
    except Exception:
        return None


# ----------------- Data utilities -----------------
def to_millions(value):
    if value is None:
        return None
    try:
        v = float(value)
        return v / 1e6
    except Exception:
        return None


def get_company_year_df(companies, years, metrics):
    # filter by companies/years
    rows = [r for r in financial_data if r["Company"] in companies and (not years or r["Year"] in years)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Keep only needed columns
    keep = ["Company", "Year"] + metrics
    df = df[keep]
    # Convert values to millions
    for m in metrics:
        df[m] = df[m].map(to_millions)
    return df.sort_values(["Company", "Year"]).reset_index(drop=True)


# ----------------- Forecasting (simple linear regression) -----------------
# Note: with only 3 points per company, keep it simple & transparent
try:
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def forecast_linear(df: pd.DataFrame, company: str, metric: str, horizon: int = 2):
    if not _HAS_SKLEARN or df.empty:
        return None, None
    sub = df[df["Company"] == company][["Year", metric]].dropna()
    if len(sub) < 2:
        return None, None
    X = sub[["Year"]].values
    y = sub[metric].values
    model = LinearRegression().fit(X, y)
    last_year = int(sub["Year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)
    y_pred = model.predict(future_years.reshape(-1, 1))

    hist = sub.rename(columns={metric: "Value"}).assign(Type="Actual", Metric=metric)
    fut = pd.DataFrame({
        "Company": company,
        "Year": future_years,
        "Value": y_pred,
        "Type": "Forecast",
        "Metric": metric,
    })
    combo = pd.concat([hist, fut], ignore_index=True)

    chart = (
        alt.Chart(combo)
        .mark_line(point=True)
        .encode(
            x="Year:O",
            y="Value:Q",
            color="Type:N",
            tooltip=["Company", "Metric", "Year", alt.Tooltip("Value:Q", title="Value (mn)")],
        )
        .properties(title=f"{company} — {metric} (mn): Actual vs Forecast")
    )
    return combo, chart


# ----------------- Chatbot core -----------------

def parse_query(query: str):
    """Combine LLM + basic rules for robust parsing."""
    parsed = parse_with_llm(query)
    if parsed:
        comps = parsed["companies"] or extract_companies_basic(query)
        yrs = parsed["years"] or extract_years_basic(query)
        mets = parsed["metrics"] or extract_metrics_basic(query)
        forecast = parsed["forecast"] or ("forecast" in query.lower() or "predict" in query.lower())
        horizon = parsed["horizon"]
    else:
        comps = extract_companies_basic(query)
        yrs = extract_years_basic(query)
        mets = extract_metrics_basic(query)
        forecast = ("forecast" in query.lower() or "predict" in query.lower())
        horizon = 2
    return comps, yrs, mets, forecast, horizon


def respond(query: str):
    comps, yrs, mets, do_forecast, horizon = parse_query(query)

    if not mets:
        return "⚠️ I can provide: revenue, net income, assets, liabilities, or cash flow.", None, None
    if not comps:
        return "⚠️ Please mention at least one company (Microsoft, Tesla, or Apple).", None, None

    # Multi-company comparison
    if len(comps) >= 2:
        df = get_company_year_df(comps, yrs, mets)
        if df.empty:
            return "No data found for your filters.", None, None

        # Melt to long format for multi-metric facet
        melt = df.melt(id_vars=["Company", "Year"], value_vars=mets, var_name="Metric", value_name="Value")
        chart = (
            alt.Chart(melt)
            .mark_bar()
            .encode(
                x="Year:O",
                y="Value:Q",
                color="Company:N",
                column=alt.Column("Metric:N", header=alt.Header(title="Metric")),
                tooltip=["Company", "Metric", "Year", alt.Tooltip("Value:Q", title="Value (mn)")],
            )
        )
        return df, chart, None

    # Single company (can show multiple metrics + optional forecast)
    comp = comps[0]
    df = get_company_year_df([comp], yrs, mets)
    if df.empty:
        return "No data found for your filters.", None, None

    melt = df.melt(id_vars=["Company", "Year"], value_vars=mets, var_name="Metric", value_name="Value")
    base_chart = (
        alt.Chart(melt)
        .mark_bar()
        .encode(
            x="Year:O",
            y="Value:Q",
            color="Metric:N",
            tooltip=["Year", "Metric", alt.Tooltip("Value:Q", title="Value (mn)")],
        )
        .properties(title=f"{comp}: {', '.join(mets)} (mn)")
    )

    # Forecast: if a single metric is requested, add forecast line
    if do_forecast and len(mets) == 1:
        metric = mets[0]
        combo, fchart = forecast_linear(df, comp, metric, horizon)
        if fchart is not None:
            return df, base_chart | fchart, None  # side-by-side charts

    return df, base_chart, None


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Financial Chatbot — LLM + Forecast", page_icon="💬", layout="wide")
st.title("💬 Financial Data Chatbot — LLM + Forecasting")

st.markdown(
    """
    Ask about **Revenue, Net Income, Assets, Liabilities, or Cash Flow** for **Microsoft, Tesla, Apple** (2022–2024). 
    Try natural questions like *"Apple 2023 profit"*, *"Compare Tesla vs Microsoft sales"*, or *"Forecast Apple revenue next 2 years"*.
    
    **Note:** All values shown in **millions**. If spelling is off, the LLM will try to normalize it.
    """
)

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

for q, r in st.session_state.history:
    st.markdown(f"**🧑 You:** {q}")
    if isinstance(r, tuple):
        df, chart, note = r
        if isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True)
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

"
