from groq import Groq

client = Groq()

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

def ask_groq_llm(context: str, question: str = "Extract all financial metrics") -> str:
    prompt = FINANCIAL_METRICS_PROMPT.format(context=context, question=question)

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_completion_tokens=1500,
        top_p=1,
        reasoning_effort="medium",
        stream=False
    )

    if hasattr(completion, 'choices'):
        return completion.choices[0].message.content.strip()
    else:
        return str(completion).strip()
