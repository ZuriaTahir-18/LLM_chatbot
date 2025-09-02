import streamlit as st
import re
import pandas as pd
import numpy as np
import altair as alt
from transformers import pipeline

# Load the LLM once

llm = pipeline("text2text-generation", model="google/flan-t5-small")

# --- Optional: Spell checker ---
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
        w = re.sub(r'[^a-zA-Z\-]', '', word)
        if not w:
            corrected.append(word)
            continue
        correction = spell.correction(w)
        # keep original casing when possible
        if correction and w.islower():
            fixed = correction
        else:
            fixed = correction.capitalize() if w[:1].isupper() else correction
        corrected.append(word.replace(w, fixed if correction else w))
    return " ".join(corrected)

# --- Optional: LLM via Hugging Face ---
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
    {"Company": "Microsoft", "Year": 2022, "Total Revenue": 1.98E11, "Net Income": 72738000000, "Total Assets": 3.65E11, "Total Liabilities": 1.98E11, "Cash Flow": 89035000000},
    {"Company": "Microsoft", "Year": 2023, "Total Revenue": 2.12E11, "Net Income": 72361000000, "Total Assets": 4.12E11, "Total Liabilities": 2.06E11, "Cash Flow": 87582000000},
    {"Company": "Microsoft", "Year": 2024, "Total Revenue": 2.45E11, "Net Income": 88136000000, "Total Assets": 5.12E11, "Total Liabilities": 2.44E11, "Cash Flow": 1.19E11},
    {"Company": "Tesla", "Year": 2022, "Total Revenue": 81462000000, "Net Income": 12587000000, "Total Assets": 82338000000, "Total Liabilities": 36440000000, "Cash Flow": 14724000000},
    {"Company": "Tesla", "Year": 2023, "Total Revenue": 96773000000, "Net Income": 14973000000, "Total Assets": 1.07E11, "Total Liabilities": 43009000000, "Cash Flow": 13256000000},
    {"Company": "Tesla", "Year": 2024, "Total Revenue": 97690000000, "Net Income": 7153000000, "Total Assets": 1.22E11, "Total Liabilities": 48390000000, "Cash Flow": 14923000000},
    {"Company": "Apple", "Year": 2022, "Total Revenue": 3.94E11, "Net Income": 99803000000, "Total Assets": 3.53E11, "Total Liabilities": 3.02E11, "Cash Flow": 1.22E11},
    {"Company": "Apple", "Year": 2023, "Total Revenue": 3.83E11, "Net Income": 96995000000, "Total Assets": 3.53E11, "Total Liabilities": 2.90E11, "Cash Flow": 1.11E11},
    {"Company": "Apple", "Year": 2024, "Total Revenue": 3.91E11, "Net Income": 93736000000, "Total Assets": 3.65E11, "Total Liabilities": 3.08E11, "Cash Flow": 1.18E11},
]

VALID_COMPANIES = {"Microsoft", "Tesla", "Apple"}
SYNONYMS = {
    "Total Revenue": {"revenue", "sales", "turnover", "income from sales"},
    "Net Income": {"net income", "profit", "earnings", "net profit"},
    "Total Assets": {"assets", "total assets", "asset"},
    "Total Liabilities": {"liabilities", "debt", "debts", "obligations"},
    "Cash Flow": {"cash flow", "cashflow", "operating cash flow", "cash"},
}

VALID_COMPANIES = {"Microsoft", "Tesla", "Apple"}

COMPANY_SYNONYMS = {
    "Apple": {"apple", "appl", "appe", "app"},
    "Tesla": {"tesla", "tesl", "tesa", "tessla"},
    "Microsoft": {"microsoft", "micro", "msft", "microsftv", "micosoft"},
}


# ----------------- Utilities -----------------
def to_millions(value):
    return float(value)/1e6 if value is not None else None

def get_company_year_df(companies, years, metrics):
    rows = [r for r in financial_data if r["Company"] in companies and (not years or r["Year"] in years)]
    if not rows:
        return pd.DataFrame()
    cols = ["Company", "Year"] + metrics
    df = pd.DataFrame(rows)[cols]
    for m in metrics:
        df[m] = df[m].map(to_millions)
    return df.sort_values(["Company","Year"]).reset_index(drop=True)

def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.reset_index(drop=True).copy()
    if "S.No" in df2.columns:
        df2 = df2.drop(columns=["S.No"])
    df2.insert(0, "S.No", range(1, len(df2) + 1))
    return df2

# ----------------- Forecasting -----------------
from sklearn.linear_model import LinearRegression

def forecast_linear(df_hist_company_metric: pd.DataFrame, company: str, metric: str, years_requested: list=None):
    # df_hist_company_metric must have columns: Company, Year, metric
    hist = df_hist_company_metric[df_hist_company_metric["Company"] == company][["Year", metric]].dropna().copy()
    if hist.empty or len(hist) < 2:
        return None
    X = hist[["Year"]].values
    y = hist[metric].values
    model = LinearRegression().fit(X, y)
    last_year = int(hist["Year"].max())
    future_years = sorted([y for y in (years_requested or []) if y > last_year and y <= 2034])
    if not future_years:
        return None
    preds = model.predict(np.array(future_years).reshape(-1, 1))
    out = pd.DataFrame({
        "Company": company,
        "Year": future_years,
        "Metric": metric,
        "Value": preds.astype(float),
        "Type": "Forecast"
    })
    return out

# ----------------- Chatbot Parsing -----------------
def extract_metrics_basic(query:str):
    low = query.lower()
    chosen = []
    for metric, keys in SYNONYMS.items():
        if any(k in low for k in keys) or metric.lower() in low:
            chosen.append(metric)
    return list(dict.fromkeys(chosen))

def extract_companies_basic(query:str):
    return [c for c in VALID_COMPANIES if c.lower() in query.lower()]

def extract_years_basic(query:str):
    # explicit years like 2025, 2030 ...
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]
    return years

def extract_next_n_years(query: str):
    low = query.lower()
    # patterns: "next 3 years", "next year", "coming 2 years", "upcoming 5 years"
    m = re.search(r"\b(next|coming|upcoming)\s+(\d+)\s+year", low)
    if m:
        return int(m.group(2))
    # "next year" singular
    if re.search(r"\b(next|coming|upcoming)\s+year\b", low):
        return 1
    return 0

def parse_query(query:str):
    q_fixed = correct_spelling(query)
    comps = extract_companies_basic(q_fixed)
    metrics = extract_metrics_basic(q_fixed)
    years = extract_years_basic(q_fixed)
    next_n = extract_next_n_years(q_fixed)
    return comps, metrics, years, next_n, None if (comps and metrics) else (
        None, None, None, None,
        "⚠️ Sorry, I only have data for Microsoft, Tesla, and Apple." if not comps
        else "⚠️ Please specify revenue, net income, assets, liabilities, or cash flow."
    )


def normalize_company(name: str) -> str | None:
    name_lower = name.lower()
    for canonical, synonyms in COMPANY_SYNONYMS.items():
        if name_lower == canonical.lower() or name_lower in synonyms:
            return canonical
    return None  # if not found

    companies_found = []
    for token in tokens:  # however you’re splitting/extracting
        comp = normalize_company(token)
        if comp:
            companies_found.append(comp)

# def generate_summary(df, llm=LLM):
#     if llm is None or df is None or not isinstance(df, pd.DataFrame) or df.empty:
#         return None
#     # small, clean table to text
#     try:
#         sample = df.head(50).to_string(index=False)
#         prompt = f"Summarize this financial data in 2 sentences. Focus on trends and any forecast vs actual:\n{sample}"
#         result = llm(prompt, max_length=150, do_sample=False)[0]['generated_text']
#         return result
#     except Exception:
#         return None



# ----------------- Respond -----------------


def preprocess_query(query: str) -> str:
    prompt = f"Correct spelling mistakes and simplify this financial query: {query}"
    response = llm(prompt, max_length=64, num_return_sequences=1)
    return response[0]["generated_text"]

def respond(query: str):

    query = preprocess_query(query)
    
    comps, mets, yrs, next_n, error = parse_query(query)
    if error:
        return error, None

    # Dataset year bounds
    earliest_year, latest_year, max_forecast_year = 2022, 2024, 2034

    # ---------------- Year Validation ----------------
    if yrs:
        for year in yrs:
            if year < earliest_year or year > max_forecast_year:
                return f"⚠️ Sorry, data is only available between **{earliest_year} and {max_forecast_year}**. You entered {year}.", None, None

    yrs = yrs or []

    # ---------------- Handle "Past N years" ----------------
    if next_n and "past" in query.lower():
        end_year = latest_year
        start_year = max(end_year - next_n + 1, earliest_year)
        yrs = list(range(start_year, end_year + 1))

    # ---------------- Handle "Next N years" ----------------
    elif next_n and "next" in query.lower():
        start_year = latest_year + 1
        yrs = list(range(start_year, min(start_year + next_n - 1, max_forecast_year) + 1))

    # ---------------- Validate again ----------------
    invalid_years = [y for y in yrs if y < earliest_year or y > max_forecast_year]
    if invalid_years:
        return f"⚠️ Please enter years between {earliest_year} and {max_forecast_year}. Invalid: {invalid_years}", None, None

    # ---------------- Split Years ----------------
    hist_years_req = [y for y in yrs if y <= latest_year]
    fut_years_req = [y for y in yrs if y > latest_year]

    # Always include previous year for comparison
    all_hist_years = set(hist_years_req)
    for y in hist_years_req:
        if y - 1 >= earliest_year:
            all_hist_years.add(y - 1)
    all_hist_years = sorted(all_hist_years)

    # If user asked only “next N years”, show recent actuals (2023, 2024) for context
    if not hist_years_req and fut_years_req:
        all_hist_years = [max(latest_year - 1, earliest_year), latest_year]

    # ---------------- Historical DF ----------------
    df_hist_wide = get_company_year_df(comps, all_hist_years, mets)

    if not df_hist_wide.empty:
        df_hist_long = df_hist_wide.melt(
            id_vars=["Company", "Year"],
            value_vars=mets,
            var_name="Metric",
            value_name="Value"
        )
        df_hist_long["Type"] = "Actual"
    else:
        df_hist_long = pd.DataFrame(columns=["Company","Year","Metric","Value","Type"])

    # ---------------- Forecast DF ----------------
    forecast_frames = []
    if fut_years_req:
        for comp in comps:
            for metric in mets:
                fc = forecast_linear(df_hist_wide[["Company","Year", *mets]], comp, metric, fut_years_req)
                if fc is not None and not fc.empty:
                    forecast_frames.append(fc)

    df_forecast_long = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame(columns=["Company","Year","Metric","Value","Type"])

    # ---------------- Combine ----------------
    df_all_long = pd.concat([df_hist_long, df_forecast_long], ignore_index=True)
    df_out = add_serial_column(df_all_long)

    # ---------------- Chart ----------------
    base = alt.Chart(df_all_long).encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("Value:Q", title="Value (mn)"),
        color=alt.Color("Company:N", title="Company")
    )

    bars = base.transform_filter(
        alt.datum.Type == "Actual"
    ).mark_bar().encode(
        tooltip=["Company","Metric","Year","Type",alt.Tooltip("Value:Q", title="Value (mn)")]
    )

    line = base.transform_filter(
        alt.datum.Type == "Forecast"
    ).mark_line(strokeDash=[6,4]).encode(
        tooltip=["Company","Metric","Year","Type",alt.Tooltip("Value:Q", title="Value (mn)")]
    )

    points = base.transform_filter(
        alt.datum.Type == "Forecast"
    ).mark_point().encode(
        tooltip=["Company","Metric","Year","Type",alt.Tooltip("Value:Q", title="Value (mn)")]
    )

    chart = alt.layer(bars, line, points).facet(
        column=alt.Column("Metric:N", header=alt.Header(title=""))
    ).resolve_scale(
        y='independent'
    ).properties(
        title=f"{', '.join(comps)} — {', '.join(mets)} (mn)"
    )

    # ---------------- Summary ----------------
    # summary = generate_summary(df_all_long)

    return df_out, chart

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="💬 Financial Chatbot", page_icon="💹", layout="wide")
st.title("💹 AI-Powered Financial Data Chatbot")

# --- LLM Test Button (kept) ---
if st.button("Test LLM"):
    if LLM is None:
        st.error("❌ LLM is not loaded. Check transformers/torch installation.")
    else:
        prompt = "Summarize: Microsoft revenue increased in 2023."
        try:
            output = LLM(prompt, max_length=50, do_sample=False)[0]['generated_text']
            st.success("✅ LLM is working!")
            # st.caption(f"LLM output: {output}")
        except Exception as e:
            st.error(f"❌ LLM failed to generate output: {e}")

st.markdown("""
🚀 **Your Smart Finance Assistant**  
Ask about **Revenue, Net Income, Assets, Liabilities, or Cash Flow** for  
**Microsoft, Tesla, and Apple (2022–2024)** 📊  
⚡ Values in **millions**. Forecast up to **2034** using **Linear Regression**.  

""")

# Chat history (kept)
if "history" not in st.session_state:
    st.session_state.history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Show chat bubbles/history
for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        if isinstance(r, tuple):
            df, chart = r
            if isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True, hide_index=True)  # 🔒 hide index column
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            # Show summary inside expander (no blue info box)
            # if summary:
            #     with st.expander("AI Summary"):
            #         st.write(summary)
        else:
            st.warning(r)

# Input (kept as chat_input to avoid removing feature)
query = st.chat_input("💡 Ask your question here…")
if query:
    st.session_state.last_query = query  # remember last text
    with st.chat_message("user"):
        st.write(query)
    with st.status("Processing…", expanded=False) as status:
        answer = respond(query)
        st.session_state.history.append((query, answer))
        status.update(label="Done", state="complete", expanded=False)
    # rerun to render the new response but chat_input will be blank (Streamlit behavior)
    st.rerun()

# Show last query (so user sees what they asked even if chat_input clears)
if st.session_state.last_query:
    st.caption(f"Last query: *{st.session_state.last_query}*")
