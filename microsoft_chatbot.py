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

# helper to make a set of metric-related keywords
METRIC_KEYWORDS = set()
for m, syns in SYNONYMS.items():
    METRIC_KEYWORDS.update(w.lower() for w in m.split())
    for s in syns:
        METRIC_KEYWORDS.update(t.lower() for t in s.split())

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

# 🔹 detect expressions like "next 3 years"
def extract_horizon(query: str):
    m = re.search(r"next (\d+) year", query.lower())
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
            "horizon": max(1, min(horizon, 10)),
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

# helper to format dataframe for display (index starts at 1)
def format_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df.index = df.index + 1
    return df

# helper to detect unsupported company mentions
COMMON_IGNORE = {
    'next','year','years','predict','forecast','in','for','and','or','show','compare','between','vs','versus',
    'please','me','my','the','a','an','of','to','on','with','by','is','are','i','you','how','what','which','give',
    'show','display','get','list','all','company','companies'
}

def detect_unsupported_companies(query: str, comps_found: list):
    """Return error message string if user mentioned a company not in VALID_COMPANIES, else None."""
    low = query.lower()
    tokens = re.findall(r"\b[a-z]{2,}\b", low)
    candidates = []
    for i, t in enumerate(tokens):
        if t in COMMON_IGNORE:
            continue
        if t in METRIC_KEYWORDS:
            continue
        if t.isdigit():
            continue
        if t in VALID_COMPANIES_LOWER:
            continue
        # consider it a company candidate only if it sits next to a metric word (e.g. 'amazon assets')
        left = tokens[i-1] if i-1 >= 0 else ''
        right = tokens[i+1] if i+1 < len(tokens) else ''
        if left in METRIC_KEYWORDS or right in METRIC_KEYWORDS:
            candidates.append(t)
    # also consider direct "company metric in YEAR" patterns where company may be first token
    # e.g., 'amazon assets in 2023' already caught; keep candidates unique
    candidates = [c for c in dict.fromkeys(candidates) if c not in VALID_COMPANIES_LOWER]
    if candidates:
        names = ', '.join(c.title() for c in candidates)
        return f"⚠️ Sorry — I only have data for Microsoft, Tesla, and Apple. I detected: {names}."
    return None

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

    # 🔹 FIX: include Company & Metric in historical part
    hist = pd.DataFrame({
        "Company": company,
        "Year": sub["Year"].values,
        "Metric": metric,
        "Value": sub[metric].values,
        "Type": "Actual",
    })
    fut = pd.DataFrame({
        "Company": company,
        "Year": future_years,
        "Metric": metric,
        "Value": y_pred,
        "Type": "Forecast",
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
    # normalize + spelling
    clean = correct_spelling(query)
    clean_low = clean.lower()

    # text-based extraction
    yrs_text = extract_years_basic(clean_low)
    horizon_text = extract_horizon(clean_low)
    comps_text = extract_companies_basic(clean)
    mets_text = extract_metrics_basic(clean)

    # LLM parsing fallback (only used to improve extraction if available)
    parsed = parse_with_llm(clean)
    if parsed:
        comps = parsed.get("companies") or comps_text
        mets = parsed.get("metrics") or mets_text
        parsed_years = parsed.get("years") or []
        yrs = parsed_years if parsed_years else yrs_text
        forecast_flag = bool(parsed.get("forecast", False)) or ("forecast" in clean_low or "predict" in clean_low)
        horizon = parsed.get("horizon", None)
    else:
        comps = comps_text
        mets = mets_text
        yrs = yrs_text
        forecast_flag = ("forecast" in clean_low or "predict" in clean_low or "next" in clean_low)
        horizon = None

    # prefer explicit 'next N years' in text if LLM didn't give horizon
    if horizon is None and horizon_text:
        horizon = horizon_text

    # Detect unsupported companies mentioned explicitly (e.g., 'amazon assets')
    unsupported_msg = detect_unsupported_companies(clean, comps)
    if unsupported_msg:
        return [], [], [], False, 0, unsupported_msg

    # Auto-detect future years (beyond dataset) and convert to horizon
    max_year = max(r["Year"] for r in financial_data)
    if yrs:
        future_years = [y for y in yrs if y > max_year]
        if future_years:
            # limit to 10 years ahead
            gap = max(future_years) - max_year
            if gap > 10:
                return [], [], [], False, 0, f"⚠️ I can only forecast up to 10 years ahead (till {max_year + 10})."
            forecast_flag = True
            horizon = min(gap, 10)
            # remove future years from yrs (we treat them as forecast horizon)
            yrs = [y for y in yrs if y <= max_year]

    # If forecast requested but still no horizon, default to 2
    if forecast_flag and (horizon is None or horizon == 0):
        horizon = 2

    # If no forecast requested, set horizon to 0
    if not forecast_flag:
        horizon = 0

    return comps, yrs, mets, forecast_flag, horizon, None


def respond(query: str):
    comps, yrs, mets, do_forecast, horizon, error_msg = parse_query(query)

    if error_msg:
        return error_msg, None, None

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
            final_df = pd.concat(results, ignore_index=True)
            final_df = format_df_for_display(final_df)
            return final_df, final_chart, None
        else:
            return "No forecast could be generated.", None, None

    # Multi-company comparison without forecast
    if len(comps) >= 2:
        df = get_company_year_df(comps, yrs, mets)
        if df.empty:
            return "No data found for your filters.", None, None

        df = format_df_for_display(df)
        melt = df.melt(id_vars=[df.columns[0], df.columns[1]], value_vars=mets, var_name="Metric", value_name="Value")
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

    df = format_df_for_display(df)
    melt = df.melt(id_vars=[df.columns[0], df.columns[1]], value_vars=mets, var_name="Metric", value_name="Value")
    base_chart = (
        alt.Chart(melt)
        .mark_bar()
        .encode(
            x="Year:O",
            y="Value:Q",
            color="Metric:N",
            tooltip=["Year", "Metric", alt.Tooltip("Value:Q", title="Value (mn)" )],
        )
        .properties(title=f"{comp}: {', '.join(mets)} (mn)")
    )

    return df, base_chart, None

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Financial Chatbot — LLM + Forecast", page_icon="💬", layout="wide")
st.title("💬 Financial Data Chatbot — LLM + Forecasting")

st.markdown(
    """
    🔮 **Your Personal Finance Data Analyst**  
    Ask anything about **Revenue, Net Income, Assets, Liabilities, or Cash Flow** for **Microsoft, Tesla, and Apple** (2022–2024).  
    
    ✅ Compare companies instantly (*"Compare Tesla and Apple profit"*)  
    ✅ Forecast up to **10 years ahead** (*"Tesla assets in 2030"*)  
    ✅ Handle multiple metrics together (*"Apple revenue and liabilities next 3 years"*)  
    ✅ Spelling auto-corrected — just type naturally!  
    
    ⚡ Data shown in **millions**. Future years beyond dataset are forecasted using **AI regression models**.  
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
