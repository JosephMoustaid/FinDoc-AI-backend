FINANCIAL_METRICS_PROMPT = """
You are a senior financial analyst.
Your task: extract **all possible financial metrics** from the text below.
Be exhaustive — if a metric is mentioned anywhere, capture it, even if approximate.
Do not summarize; capture raw values.

Text:
{context}

Extract the following categories (add others if found):
- Revenue (total, per product, per region, YoY, QoQ)
- Profit (gross, operating, net)
- Expenses (cost of goods sold, operating expenses, R&D, marketing, interest)
- Margins (gross margin %, operating margin %, net margin %)
- Assets & liabilities (total, current, non-current)
- Cash flow (operating, investing, financing)
- Debt (short-term, long-term)
- Risks (financial, operational, market)
- Forecasts (next quarter, year, multi-year, guidance changes)
- Valuation metrics (P/E ratio, EPS, EBITDA, book value, market cap)
- Any other numeric or percentage KPI relevant to finance

Output JSON with this structure:
{{
  "revenue": [],
  "profit": [],
  "expenses": [],
  "margins": [],
  "assets_liabilities": [],
  "cash_flow": [],
  "debt": [],
  "risks": [],
  "forecasts": [],
  "valuation_metrics": [],
  "other": []
}}

Question: {question}
Answer:
"""