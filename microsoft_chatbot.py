import streamlit as st
import re
import pandas as pd
import altair as alt

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

valid_companies = {"Microsoft", "Tesla", "Apple"}

# ----------------- Helper Functions -----------------
def extract_companies(query):
    companies = []
    for item in financial_data:
        if item["Company"].lower() in query.lower():
            companies.append(item["Company"])
    return list(set(companies))

def extract_years(query):
    years = re.findall(r'\b(20\d{2})\b', query)
    if len(years) == 2:
        start_year, end_year = int(years[0]), int(years[1])
        years_in_range = list(range(start_year, end_year + 1))
        return years_in_range
    return [int(year) for year in years] if years else []

def convert_to_million(value):
    """Convert numbers to millions for better readability."""
    if value >= 1E6:
        return value / 1E6  # Return as numeric value for graph
    return value

# ----------------- Chatbot Logic -----------------
def financial_chatbot(query):
    query_lower = query.lower()
    companies = extract_companies(query)
    years = extract_years(query)

    # Handle multiple metrics (assets, revenue, etc.)
    metrics = []
    if "revenue" in query_lower:
        metrics.append("Total Revenue")
    if "assets" in query_lower:
        metrics.append("Total Assets")
    if "net income" in query_lower:
        metrics.append("Net Income")
    if "liabilities" in query_lower:
        metrics.append("Total Liabilities")
    if "cash flow" in query_lower:
        metrics.append("Cash Flow")

    # If no valid metric is found, return error message
    if not metrics:
        return "⚠️ Sorry, I can only provide info on revenue, net income, assets, liabilities, or cash flow."

    # Check if any unknown companies were mentioned in the query
    notify_msg = ""
    if companies:
        unknown_companies = [company for company in companies if company not in valid_companies]
        if unknown_companies:
            notify_msg = f"⚠️ Sorry, I only have data for Microsoft, Tesla, and Apple. Did you mean one of them? Please recheck your spelling if it was a typo."

    if not companies:
        return notify_msg or "⚠️ Please mention at least one company (Microsoft, Tesla, or Apple)."

    # comparison (multiple companies)
    if len(companies) >= 2:
        data_companies = {
            comp: [item for item in financial_data if item["Company"] == comp and (not years or item["Year"] in years)]
            for comp in companies
        }
        if not any(data_companies.values()):
            return notify_msg or "No data found."
        all_years = sorted(set([d["Year"] for data in data_companies.values() for d in data]))
        df = pd.DataFrame({"Year": all_years})

        # Collect data for all requested metrics
        for metric in metrics:
            for comp in companies:
                df[f"{comp} - {metric} (in millions)"] = [next((d[metric] for d in data_companies[comp] if d["Year"] == y), None) for y in all_years]
        
        # Convert the values to millions (numerically) before plotting
        for col in df.columns[1:]:
            df[col] = df[col].apply(lambda x: convert_to_million(x) if x is not None else x)

        # Plot the comparison chart
        df_melt = df.melt("Year", var_name="Company and Metric", value_name="Value")
        chart = alt.Chart(df_melt).mark_bar().encode(
            x="Year:O", 
            y="Value:Q", 
            color=alt.Color("Year:N", scale=alt.Scale(scheme='category20')),
            tooltip=["Year", "Company and Metric", alt.Tooltip("Value:Q", title="Value (in millions)")]
        )
        return (df, chart, notify_msg)


    # single company
    else:
        comp = companies[0]
        data = [item for item in financial_data if item["Company"] == comp and (not years or item["Year"] in years)]
        if not data:
            return notify_msg or "No data found."
        df = pd.DataFrame({
            "Year": [d["Year"] for d in data],
        })
        
        # Collect data for all requested metrics
        for metric in metrics:
            df[f"{metric} ({comp}) (in millions)"] = [d[metric] for d in data]

        # Convert the values to millions (numerically) before plotting
        for col in df.columns[1:]:
            df[col] = df[col].apply(lambda x: convert_to_million(x) if x is not None else x)

        # Plot the individual company chart
        chart = alt.Chart(df).mark_bar().encode(
            x="Year:O", y=f"{metric} ({comp}) (in millions):Q", color="Year:N", tooltip=["Year", f"{metric} ({comp}) (in millions)"]
        )
        return (df, chart, notify_msg)

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="Financial Data Chatbot", page_icon="💬", layout="wide")

st.title("💬 Financial Data Chatbot")

# ----------------- Description Section -----------------
st.markdown("""
### 📊 About This Chatbot
This chatbot is designed to help you explore the **financial performance** of three major companies from the years 2022, 2023, and 2024:
- **Microsoft**
- **Tesla**
- **Apple**

The data includes the following metrics:
- **Total Revenue** – Company’s total income from sales.
- **Net Income** – Profit after all expenses are deducted.
- **Total Assets** – Everything the company owns (e.g., buildings, cash, equipment).
- **Total Liabilities** – Everything the company owes (e.g., loans, debt).
- **Cash Flow** – The money coming in and out of the business.

### 💡 How to Use
You can ask questions like:
- `What is Apple's net income in 2023?`
- `Compare Microsoft and Tesla revenue`
- `Show Tesla cash flow over the years`
- `Give me Apple’s total assets`

👉 You must **specify a company name** in your query.  
👉 You can also compare multiple companies in one query  
👉 Please check for spelling errors for better accuracy!

---

**Note:** Values are displayed in millions.
""")

st.markdown("Ask me about **Revenue, Net Income, Assets, Liabilities, or Cash Flow** for **Microsoft, Tesla, and Apple**.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for q, r in st.session_state.history:
    st.markdown(f"**🧑 You:** {q}")
    if isinstance(r, tuple):  # response with df + chart + notify
        df, chart, notify_msg = r
        if isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True)

            # Display the chart
            st.altair_chart(chart, use_container_width=True)

        if notify_msg:
            st.info(notify_msg)
    else:
        st.warning(r)

# ✅ Input always at the bottom
query = st.chat_input("💡 Ask your question here...")

if query:
    response = financial_chatbot(query)
    st.session_state.history.append((query, response))
    st.rerun()  # refresh to show new message at bottom


