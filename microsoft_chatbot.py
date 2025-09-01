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
# flatten to common short phrases (e.g., 'assets', 'revenue', 'net income')
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


def extract_years_basic(query: str):
    years = re.findall(r"\b(20\d{2})\b", query)
    return [int(y) for y in years]


# 🔹 detect expressions like "next 3 years"
def extract_horizon(query: str):
    m = re.search(r"next\s+(\d+)\s+year", query.lower())
    if m:
        return int(m.group(1))
    return None


# 🔹 detect ranges like "from 2023 to 2026" or "between 2023 and 2026" or "2023-2026"
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


def extract_metrics_basic(query: str):
    low = query.lower()
    chosen = []
    for metric, keys in SYNONYMS.items():
        if any(k in low for k in keys) or metric.lower() in low:
            chosen.append(metric)
    # ensure order stable
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


# helper: add 'S.No' starting at 1 and optionally reorder for forecast view
def add_serial_column(df: pd.DataFrame, reorder_for_forecast: bool = False) -> pd.DataFrame:
    df2 = df.reset_index(drop=True).copy()
    df2.insert(0, "S.No", range(1, len(df2) + 1))
    if reorder_for_forecast:
        # desired order for forecast outputs
        desired = ["S.No", "Year", "Metric", "Value", "Company", "Type"]
        existing = [c for c in desired if c in df2.columns]
        df2 = df2[existing]
    return df2


# improved unsupported-company detector
COMMON_IGNORE = {
    'next','year','years','predict','forecast','in','for','and','or','show','compare','between','vs','versus',
    'please','me','my','the','a','an','of','to','on','with','by','is','are','i','you','how','what','which','give',
    'show','display','get','list','all','company','companies'
}


def detect_unsupported_companies(query: str, comps_found: list):
    """Return error message string if user mentioned a company not in VALID_COMPANIES, else None."""
    low = query.lower()
    # 1) quick regex: company followed by metric phrase: 'amazon assets', 'amazon revenue'
    m = re.search(r"\b([a-z][a-z0-9&\.\-]{1,})\s+" + metric_regex, low)
    if m:
        candidate = m.group(1).lower()
        if candidate not in VALID_COMPANIES_LOWER:
            return f"⚠️ Sorry — I only have data for Microsoft, Tesla, and Apple. I detected: {candidate.title()}."
    # 2) token-based heuristics when regex doesn't match
    tokens = re.findall(r"\b[a-z]{2,}\b", low)
    candidates = []
    for i, t in enumerate(tokens):
        if t in COMMON_IGNORE:
            continue
        if t in VALID_COMPANIES_LOWER:
            continue
        if t in metric_phrase_list:
            continue
        # if token sits next to metric word, treat as candidate
        left = tokens[i-1] if i-1 >= 0 else ''
        right = tokens[i+1] if i+1 < len(tokens) else ''
        if left in metric_phrase_list or right in metric_phrase_list:
            candidates.append(t)
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


def forecast_linear(df: pd.DataFrame, company: str, metric: str, horizon: int = 2, years_requested: list = None):
    """
    df: a DataFrame that contains at least the historical rows for the company and metric
    company: company name
    metric: metric column name (already in millions)
    horizon: fallback horizon if years_requested not provided
    years_requested: optional list of years the user explicitly asked for (may include future years)
    Returns: combo_df (Company, Year, Metric, Value, Type), chart
    """
    if not _HAS_SKLEARN or df.empty:
        return None, None

    # Use ALL available historical data to train the model (don't filter by years_requested)
    full_hist = df[df["Company"] == company][["Year", metric]].dropna().copy()
    if len(full_hist) < 2:
        return None, None

    X = full_hist[["Year"]].values
    y = full_hist[metric].values
    model = LinearRegression().fit(X, y)

    last_year = int(full_hist["Year"].max())

    # determine which future years we need to predict
    if years_requested:
        future_years = sorted([y for y in years_requested if y > last_year])
    else:
        future_years = list(range(last_year + 1, last_year + 1 + horizon))

    # cap predictions to 10 years ahead of dataset max (safety)
    dataset_max = max(r["Year"] for r in financial_data)
    future_years = [y for y in future_years if y <= dataset_max + 10]

    y_pred = model.predict(np.array(future_years).reshape(-1, 1)) if future_years else np.array([])

    # Historical rows to display: if years_requested provided, show only requested years within historical range
    if years_requested:
        hist_years = sorted([y for y in years_requested if y <= last_year])
    else:
        hist_years = sorted(full_hist["Year"].tolist())

    full_hist_map = full_hist.set_index("Year")[metric].to_dict()
    hist_list = []
    for y in hist_years:
        if y in full_hist_map:
            hist_list.append({
                "Company": company,
                "Year": int(y),
                "Metric": metric,
                "Value": full_hist_map[y],
                "Type": "Actual",
            })

    fut_list = []
    for y, pred in zip(future_years, y_pred):
        fut_list.append({
            "Company": company,
            "Year": int(y),
            "Metric": metric,
            "Value": float(pred),
            "Type": "Forecast",
        })

    combo = pd.DataFrame(hist_list + fut_list)

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

    # text-based extraction (range detection first)
    yrs_range = extract_year_range(clean_low)
    yrs_text = extract_years_basic(clean_low) if yrs_range is None else yrs_range
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

    # prefer explicit 'next N years' in text if no LLM horizon
    if horizon is None and horizon_text:
        horizon = horizon_text

    # Detect unsupported companies mentioned explicitly
    unsupported_msg = detect_unsupported_companies(clean, comps)
    if unsupported_msg:
        return [], [], [], False, 0, unsupported_msg

    # Auto-detect future years (beyond dataset) and convert to horizon
    dataset_max = max(r["Year"] for r in financial_data)
    if yrs:
        future_years = [y for y in yrs if y > dataset_max]
        if future_years:
            gap = max(future_years) - dataset_max
            if gap > 10:
                return [], [], [], False, 0, f"⚠️ I can only forecast up to 10 years ahead (till {dataset_max + 10})."
            forecast_flag = True
            # For range queries we want the exact future years user asked for — compute horizon accordingly
            horizon = min(gap, 10)
            # keep yrs as-is (we will use yrs to decide which historical years to display and which future years to forecast)

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
            # For training use all available historical data for this company & metric
            for metric in mets:
                df_all = get_company_year_df([comp], None, [metric])
                if df_all.empty:
                    continue
                # pass years_requested so forecast_linear shows only requested historical years (if any)
                combo, fchart = forecast_linear(df_all, comp, metric, horizon, years_requested=yrs)
                if fchart is not None and combo is not None and not combo.empty:
                    charts.append(fchart)
                    results.append(combo)
        if charts:
            final_chart = alt.vconcat(*charts)
            final_df = pd.concat(results, ignore_index=True)
            # add serial column and reorder for forecast view
            final_df = add_serial_column(final_df, reorder_for_forecast=True)
            return final_df, final_chart, None
        else:
            return "No forecast could be generated.", None, None

    # Multi-company comparison without forecast
    if len(comps) >= 2:
        df = get_company_year_df(comps, yrs, mets)
        if df.empty:
            return "No data found for your filters.", None, None

        # add serial column (keep columns intact)
        df_out = add_serial_column(df, reorder_for_forecast=False)
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
        return df_out, chart, None

    # Single company without forecast
    comp = comps[0]
    df = get_company_year_df([comp], yrs, mets)
    if df.empty:
        return "No data found for your filters.", None, None

    df_out = add_serial_column(df, reorder_for_forecast=False)
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

    return df_out, base_chart, None


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
