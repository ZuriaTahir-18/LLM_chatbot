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
    years = re.findall(r"\\b(20\\d{2})\\b", query)
    if len(years) == 2:
        a, b = int(years[0]), int(years[1])
        return list(range(min(a, b), max(a, b) + 1))
    return [int(y) for y in years]

# 🔹 detect expressions like "next 3 years"
def extract_horizon(query: str):
    m = re.search(r"next (\\d+) year", query.lower())
    if m:
        return int(m.group(1))
    return None

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

# ----------------- LLM-powered parsing -----------------
def parse_with_llm(query: str):
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
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            out = out[start:end + 1]
        data = json.loads(out)
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
            "horizon": max(1, min(horizon, 5)),
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
    rows = [r for r in financial_data if r["Company"] in companies and (not years or r["Year"] in years)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["Company", "Year"] + metrics
    df = df[keep]
    for m in metrics:
        df[m] = df[m].map(to_millions)
    return df.sort_values(["Company", "Year"]).reset_index(drop=True)

# ----------------- Forecasting -----------------
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

    hist = sub.rename(columns={metric: "Value"}).assign(Type="Actual", Metric=metric, Company=company)
    fut = pd.DataFrame({
        "Company": company,
        "Year": future_years,
        "Metric": metric,
        "Value": y_pred,
        "Type": "Forecast",
    })
    combo = pd.concat([hist, fut], ignore_index=True)

    # ✅ Reorder columns: Serial, Year, Metric, Value, Company, Type
    combo.reset_index(inplace=True)
    combo.rename(columns={"index": "S.No"}, inplace=True)
    combo = combo[["S.No", "Year", "Metric", "Value", "Company", "Type"]]

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
# ----------------- Chatbot core (REPLACE parse_query with this) -----------------
def parse_query(query: str):
    # 1) Basic normalization + spelling
    clean = correct_spelling(query)
    clean_low = clean.lower()

    # 2) Extract obvious things from text first
    yrs_text = extract_years_basic(clean_low)         # explicit 4-digit years in query
    horizon_text = extract_horizon(clean_low)        # "next N years" if present
    comps_text = extract_companies_basic(clean)      # companies by substring
    mets_text = extract_metrics_basic(clean)         # metrics by substring/synonyms

    # 3) Try LLM parsing as a fallback/refinement (do not blindly trust its horizon)
    parsed = parse_with_llm(clean)  # may be None
    if parsed:
        # companies & metrics: prefer LLM if it gave them, else keep text-extracted ones
        comps = parsed.get("companies") or comps_text
        mets = parsed.get("metrics") or mets_text

        # years: prefer LLM years if it returned any, else use text years
        parsed_years = parsed.get("years") or []
        yrs = parsed_years if parsed_years else yrs_text

        # forecast detection: combine LLM signal with keyword detection
        forecast = bool(parsed.get("forecast", False)) or ("forecast" in clean_low or "predict" in clean_low)
        # horizon: only accept parsed horizon if LLM actually returned it (i.e., non-null)
        horizon = parsed.get("horizon", None)
    else:
        comps = comps_text
        mets = mets_text
        yrs = yrs_text
        forecast = ("forecast" in clean_low or "predict" in clean_low or "next" in clean_low)
        horizon = None

    # 4) If horizon not set from LLM, prefer explicit "next N years" found in text
    if horizon is None and horizon_text:
        horizon = horizon_text

    # 5) If explicit future years were requested (e.g., 2030), convert to horizon
    if yrs:
        max_year = max(r["Year"] for r in financial_data)
        future_years = [y for y in yrs if y > max_year]
        if future_years:
            forecast = True
            # horizon is distance from max available year (e.g., 2030 -> 6 if max_year=2024)
            horizon = max(y - max_year for y in future_years)
            # remove those future years from yrs, since we treat them as forecast horizon
            yrs = [y for y in yrs if y <= max_year]

    # 6) Final fallback: if forecast requested but horizon still None -> default 2
    if forecast and (horizon is None or horizon == 0):
        horizon = 2

    # 7) If forecast not requested, keep horizon 0 (no forecasting)
    if not forecast:
        horizon = 0

    return comps, yrs, mets, forecast, horizon


def respond(query: str):
    comps, yrs, mets, do_forecast, horizon = parse_query(query)

    if not mets:
        return "⚠️ I can provide: revenue, net income, assets, liabilities, or cash flow.", None, None
    if not comps:
        return "⚠️ Please mention at least one company (Microsoft, Tesla, or Apple).", None, None

    # 🔹 Multi-company & multi-metric with forecast
    if do_forecast:
        charts = []
        results = []
        for comp in comps:
            df = get_company_year_df([comp], yrs, mets)
            if df.empty:
                continue
            for metric in mets:
                combo, fchart = forecast_linear(df, comp, metric, horizon)
                if fchart is not None:
                    charts.append(fchart)
                    results.append(combo)
        if charts:
            final_chart = alt.vconcat(*charts)
            return pd.concat(results, ignore_index=True), final_chart, None
        else:
            return "No forecast could be generated.", None, None

    # Multi-company comparison without forecast
    if len(comps) >= 2:
        df = get_company_year_df(comps, yrs, mets)
        if df.empty:
            return "No data found for your filters.", None, None

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

    # Single company without forecast
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

    return df, base_chart, None

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Financial Chatbot — LLM + Forecast", page_icon="💬", layout="wide")
st.title("💬 Financial Data Chatbot — LLM + Forecasting")

st.markdown(
    """
    🚀 **Welcome to the AI-Powered Financial Chatbot!** 💬  

    Curious about the financial future of **Microsoft, Tesla, or Apple**?  
    This chatbot helps you **explore, compare, and forecast** key financial metrics with ease.  

    🔍 **What you can do here:**  
    - Ask about **Revenue, Net Income, Assets, Liabilities, or Cash Flow** 📊  
    - Get **forecasts for the next 1–5 years** using smart predictive models 📈  
    - Compare **multiple companies side by side** for quick insights 🤝  
    - Enjoy **auto spell-correction** so typos won’t stop you ✨  

    💡 **Try asking:**  
    - *"Compare Microsoft and Tesla profit in 2023"*  
    - *"Forecast Apple revenue for the next 4 years"*  
    - *"Show Tesla assets and liabilities between 2022–2024"*  

    👉 All results are shown in **millions**, with clean tables & interactive charts.  
    """
)


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


