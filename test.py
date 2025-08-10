from flask import Flask, request, jsonify, send_file
from huggingface_hub import InferenceClient
from flask_cors import CORS
import fitz  # PyMuPDF
import os
import json
import re
from dotenv import load_dotenv
import yfinance as yf
from yahooquery import search
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
import numpy as np

# Load environment variables
load_dotenv()

# Get API key from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
  raise ValueError("HF_API_KEY environment variable not set")

app = Flask(__name__)
CORS(app)

client = InferenceClient(api_key=HF_API_KEY)

UPLOAD_FOLDER = "uploads"
FINANCIAL_DATA_FOLDER = "financial_data"
REPORTS_FOLDER = "reports"
CHARTS_FOLDER = "charts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FINANCIAL_DATA_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# Global storage for current document's financial data
current_financial_data = {}
current_pdf_text = ""


def safe_json_loads(raw_text):
  """Extract JSON from raw text and parse it safely."""
  if not raw_text or not raw_text.strip():
    return {}

  # Try to find JSON block inside text
  match = re.search(r'\{.*\}', raw_text, re.DOTALL)
  if match:
    try:
      return json.loads(match.group())
    except json.JSONDecodeError:
      pass

  return {}


def extract_financial_data(text):
  """Extract comprehensive financial data from the full document text."""
  try:
    extraction_instruction = f"""
        Analyze this financial document and extract ALL available financial information.
        Return a comprehensive JSON with the following structure:
        {{
          "company_info": {{
            "name": "company name if found",
            "sector": "industry/sector if mentioned",
            "fiscal_year": "fiscal year period"
          }},
          "revenue_data": {{
            "total_revenue": "current period revenue",
            "revenue_growth": "growth rate or change",
            "revenue_breakdown": "any segment breakdown"
          }},
          "profitability": {{
            "gross_profit": "gross profit amount",
            "operating_profit": "operating profit/EBIT",
            "net_income": "net income/profit",
            "profit_margins": "any margin percentages"
          }},
          "financial_position": {{
            "total_assets": "total assets value",
            "total_liabilities": "total liabilities",
            "shareholders_equity": "equity amount",
            "cash_position": "cash and equivalents"
          }},
          "cash_flow": {{
            "operating_cash_flow": "cash from operations",
            "free_cash_flow": "free cash flow",
            "capex": "capital expenditures"
          }},
          "key_metrics": {{
            "eps": "earnings per share",
            "pe_ratio": "price to earnings if mentioned",
            "debt_to_equity": "debt ratios",
            "roe": "return on equity"
          }},
          "risks_and_outlook": {{
            "key_risks": "main risk factors mentioned",
            "guidance": "forward guidance or outlook",
            "market_conditions": "market commentary"
          }}
        }}

        If any field is not available in the document, set it to null.
        Extract specific numbers, percentages, and monetary values.

        Document text: {text}
        """

    extraction_messages = [
      {"role": "system",
       "content": "You are an expert financial analyst. Extract all financial information comprehensively and accurately from documents."},
      {"role": "user", "content": extraction_instruction}
    ]

    extraction_response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-3B-Instruct",
      messages=extraction_messages,
      max_tokens=1024,
      temperature=0.1
    )

    raw_content = extraction_response.choices[0].message.content
    extracted_data = safe_json_loads(raw_content)

    return {
      "financial_data": extracted_data,
      "extraction_success": bool(extracted_data),
      "raw_model_output": raw_content
    }

  except Exception as e:
    return {
      "financial_data": {},
      "extraction_success": False,
      "error": str(e),
      "raw_model_output": ""
    }


def company_name_to_symbol(company_name):
  """Convert company name to stock symbol using Yahoo Query."""
  try:
    results = search(company_name)
    symbols = [quote['symbol'] for quote in results.get('quotes', [])]
    return symbols if symbols else []
  except Exception as e:
    print(f"Error searching for company: {e}")
    return []


def get_company_info_for_symbols(symbols):
  """Get detailed company information from Yahoo Finance."""
  info_list = []
  for symbol in symbols:
    try:
      ticker = yf.Ticker(symbol)
      info = ticker.info
      company_data = {
        'symbol': symbol,
        'longName': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'website': info.get('website', 'N/A'),
        'description': info.get('longBusinessSummary', 'N/A'),
        'marketCap': info.get('marketCap', 'N/A'),
        'regularMarketPrice': info.get('regularMarketPrice', 'N/A'),
        'regularMarketChangePercent': info.get('regularMarketChangePercent', 'N/A'),
        'trailingPE': info.get('trailingPE', 'N/A'),
        'forwardPE': info.get('forwardPE', 'N/A'),
        'beta': info.get('beta', 'N/A'),
        'dividendYield': info.get('dividendYield', 'N/A'),
        'profitMargins': info.get('profitMargins', 'N/A'),
        'revenueGrowth': info.get('revenueGrowth', 'N/A'),
        'earningsGrowth': info.get('earningsGrowth', 'N/A'),
        'debtToEquity': info.get('debtToEquity', 'N/A'),
        'returnOnEquity': info.get('returnOnEquity', 'N/A'),
        'currentRatio': info.get('currentRatio', 'N/A'),
        'quickRatio': info.get('quickRatio', 'N/A'),
        'totalRevenue': info.get('totalRevenue', 'N/A'),
        'grossMargins': info.get('grossMargins', 'N/A'),
        'operatingMargins': info.get('operatingMargins', 'N/A'),
        'totalDebt': info.get('totalDebt', 'N/A'),
        'totalCash': info.get('totalCash', 'N/A'),
        'priceToBook': info.get('priceToBook', 'N/A')
      }
      info_list.append(company_data)
    except Exception as e:
      print(f"Error getting info for {symbol}: {e}")
      continue
  return info_list


def format_number(value):
  """Format large numbers for better readability."""
  if value == 'N/A' or value is None or value == '':
    return 'N/A'

  try:
    num = float(value)
    if num >= 1e12:
      return f"${num / 1e12:.2f}T"
    elif num >= 1e9:
      return f"${num / 1e9:.2f}B"
    elif num >= 1e6:
      return f"${num / 1e6:.2f}M"
    elif num >= 1e3:
      return f"${num / 1e3:.2f}K"
    else:
      return f"${num:.2f}"
  except:
    return str(value)


def format_percentage(value):
  """Format percentage values."""
  if value == 'N/A' or value is None or value == '':
    return 'N/A'
  try:
    num_val = float(value)
    if abs(num_val) <= 1:
      return f"{num_val * 100:.2f}%"
    else:
      return f"{num_val:.2f}%"
  except:
    return str(value)


def create_financial_charts(yahoo_data, company_name):
  """Create beautiful financial charts for the PDF report."""
  charts = []

  # Set up the style
  plt.style.use('default')
  colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

  try:
    # Chart 1: Financial Health Pie Chart
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Create financial health metrics
    metrics = []
    values = []
    colors_used = []

    if yahoo_data.get('profitMargins') and yahoo_data.get('profitMargins') != 'N/A':
      profit_margin = float(yahoo_data.get('profitMargins', 0)) * 100
      metrics.append(f'Profit Margin\n{profit_margin:.1f}%')
      values.append(abs(profit_margin) if profit_margin > 0 else 1)
      colors_used.append(colors_palette[0])

    if yahoo_data.get('returnOnEquity') and yahoo_data.get('returnOnEquity') != 'N/A':
      roe = float(yahoo_data.get('returnOnEquity', 0)) * 100
      metrics.append(f'ROE\n{roe:.1f}%')
      values.append(abs(roe) if roe > 0 else 1)
      colors_used.append(colors_palette[1])

    if yahoo_data.get('currentRatio') and yahoo_data.get('currentRatio') != 'N/A':
      current_ratio = float(yahoo_data.get('currentRatio', 0))
      metrics.append(f'Current Ratio\n{current_ratio:.2f}')
      values.append(current_ratio * 10)  # Scale for visualization
      colors_used.append(colors_palette[2])

    if values:
      wedges, texts, autotexts = ax1.pie(values, labels=metrics, colors=colors_used,
                                         autopct='', startangle=90, textprops={'fontsize': 10})
      ax1.set_title(f'{company_name} - Financial Health Metrics', fontsize=14, fontweight='bold', pad=20)
    else:
      ax1.text(0.5, 0.5, 'Financial data not available', ha='center', va='center', transform=ax1.transAxes)
      ax1.set_title(f'{company_name} - Financial Health Metrics', fontsize=14, fontweight='bold')

    plt.tight_layout()
    chart1_path = os.path.join(CHARTS_FOLDER, f'{company_name.replace(" ", "_")}_financial_health.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    charts.append(chart1_path)

    # Chart 2: Valuation Metrics Bar Chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    valuation_metrics = []
    valuation_values = []

    if yahoo_data.get('trailingPE') and yahoo_data.get('trailingPE') != 'N/A':
      pe_ratio = float(yahoo_data.get('trailingPE', 0))
      if pe_ratio > 0 and pe_ratio < 100:  # Filter unrealistic values
        valuation_metrics.append('P/E Ratio')
        valuation_values.append(pe_ratio)

    if yahoo_data.get('priceToBook') and yahoo_data.get('priceToBook') != 'N/A':
      pb_ratio = float(yahoo_data.get('priceToBook', 0))
      if pb_ratio > 0 and pb_ratio < 50:  # Filter unrealistic values
        valuation_metrics.append('P/B Ratio')
        valuation_values.append(pb_ratio)

    if yahoo_data.get('beta') and yahoo_data.get('beta') != 'N/A':
      beta = float(yahoo_data.get('beta', 0))
      if abs(beta) < 10:  # Filter unrealistic values
        valuation_metrics.append('Beta')
        valuation_values.append(abs(beta))

    if valuation_values:
      bars = ax2.bar(valuation_metrics, valuation_values, color=colors_palette[:len(valuation_values)])
      ax2.set_title(f'{company_name} - Valuation Metrics', fontsize=14, fontweight='bold')
      ax2.set_ylabel('Value', fontsize=12)

      # Add value labels on bars
      for bar, value in zip(bars, valuation_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(valuation_values) * 0.01,
                 f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
      ax2.text(0.5, 0.5, 'Valuation data not available', ha='center', va='center', transform=ax2.transAxes)
      ax2.set_title(f'{company_name} - Valuation Metrics', fontsize=14, fontweight='bold')

    plt.tight_layout()
    chart2_path = os.path.join(CHARTS_FOLDER, f'{company_name.replace(" ", "_")}_valuation.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    charts.append(chart2_path)

  except Exception as e:
    print(f"Error creating charts: {e}")

  return charts


def generate_ai_analysis(pdf_data, yahoo_data, company_name):
  """Generate AI-powered financial analysis for the PDF report."""
  try:
    analysis_prompt = f"""
        Create a comprehensive financial analysis for {company_name} based on the available data.
        Write in a professional, analytical tone suitable for investors and financial professionals.
        Focus on key insights, trends, and investment implications.

        PDF Document Data: {json.dumps(pdf_data) if pdf_data else 'No document data available'}
        Market Data: {json.dumps(yahoo_data) if yahoo_data else 'No market data available'}

        Provide a detailed analysis covering:
        1. Financial performance overview
        2. Profitability and efficiency analysis
        3. Financial position and liquidity
        4. Valuation assessment
        5. Key strengths and concerns
        6. Investment outlook

        Keep the analysis between 500-800 words and make it actionable for investors.
        """

    messages = [
      {"role": "system", "content": "You are a senior financial analyst providing detailed investment analysis."},
      {"role": "user", "content": analysis_prompt}
    ]

    response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-3B-Instruct",
      messages=messages,
      max_tokens=1024,
      temperature=0.3
    )

    return response.choices[0].message.content

  except Exception as e:
    return f"Analysis could not be generated due to: {str(e)}"


def generate_risk_analysis(pdf_data, yahoo_data):
  """Generate risk assessment for the company."""
  try:
    risk_prompt = f"""
        Based on the financial data provided, identify and analyze the key risk factors for this investment.
        Consider market risks, financial risks, operational risks, and regulatory risks.

        Financial Data: {json.dumps(pdf_data) if pdf_data else 'Limited document data'}
        Market Data: {json.dumps(yahoo_data) if yahoo_data else 'Limited market data'}

        Provide a balanced risk assessment covering:
        1. Financial leverage and debt concerns
        2. Market position and competitive risks
        3. Sector-specific risks
        4. Liquidity and operational risks
        5. Overall risk rating (Low/Medium/High)

        Keep the analysis concise but comprehensive (300-500 words).
        """

    messages = [
      {"role": "system",
       "content": "You are a risk analyst providing comprehensive risk assessment for investment decisions."},
      {"role": "user", "content": risk_prompt}
    ]

    response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-3B-Instruct",
      messages=messages,
      max_tokens=512,
      temperature=0.2
    )

    return response.choices[0].message.content

  except Exception as e:
    return f"Risk analysis could not be generated due to: {str(e)}"


def create_pdf_report(pdf_data, yahoo_data, company_name):
  """Create a beautiful, modern PDF report."""
  try:
    # Create filename
    safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_company_name}_financial_report_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_FOLDER, filename)

    # Create the document
    doc = SimpleDocTemplate(filepath, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Get styles
    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle(
      'CustomTitle',
      parent=styles['Heading1'],
      fontSize=24,
      spaceAfter=30,
      alignment=TA_CENTER,
      textColor=colors.HexColor('#1f4e79')
    )

    heading_style = ParagraphStyle(
      'CustomHeading',
      parent=styles['Heading2'],
      fontSize=16,
      spaceAfter=12,
      spaceBefore=20,
      textColor=colors.HexColor('#2e5c8a')
    )

    subheading_style = ParagraphStyle(
      'CustomSubHeading',
      parent=styles['Heading3'],
      fontSize=14,
      spaceAfter=6,
      spaceBefore=12,
      textColor=colors.HexColor('#4a6fa5')
    )

    body_style = ParagraphStyle(
      'CustomBody',
      parent=styles['Normal'],
      fontSize=11,
      spaceAfter=6,
      alignment=TA_JUSTIFY
    )

    # Build the story (content)
    story = []

    # Title Page
    story.append(Paragraph(f"Financial Analysis Report", title_style))
    story.append(Paragraph(f"{company_name}", title_style))
    story.append(Spacer(1, 20))

    # Company info table
    company_info_data = [
      ['Company Name', yahoo_data.get('longName', 'N/A')],
      ['Symbol', yahoo_data.get('symbol', 'N/A')],
      ['Sector', yahoo_data.get('sector', 'N/A')],
      ['Industry', yahoo_data.get('industry', 'N/A')],
      ['Report Date', datetime.now().strftime('%B %d, %Y')]
    ]

    company_table = Table(company_info_data, colWidths=[2 * inch, 4 * inch])
    company_table.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
      ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, 0), 12),
      ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
      ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
      ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(company_table)
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))

    # Key metrics summary
    key_metrics_data = [
      ['Metric', 'Value'],
      ['Market Cap', format_number(yahoo_data.get('marketCap'))],
      ['Current Price', f"${yahoo_data.get('regularMarketPrice', 'N/A')}"],
      ['Daily Change', format_percentage(yahoo_data.get('regularMarketChangePercent'))],
      ['P/E Ratio', str(yahoo_data.get('trailingPE', 'N/A'))],
      ['Profit Margin', format_percentage(yahoo_data.get('profitMargins'))],
      ['ROE', format_percentage(yahoo_data.get('returnOnEquity'))],
      ['Debt/Equity', str(yahoo_data.get('debtToEquity', 'N/A'))],
      ['Current Ratio', str(yahoo_data.get('currentRatio', 'N/A'))]
    ]

    metrics_table = Table(key_metrics_data, colWidths=[3 * inch, 2 * inch])
    metrics_table.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
      ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, 0), 11),
      ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
      ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
      ('ALTERNATEROWCOLOR', (0, 1), (-1, -1), colors.white, colors.lightgrey),
      ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 20))

    # Financial Analysis from AI
    story.append(Paragraph("Financial Analysis", heading_style))

    # Generate AI analysis
    ai_analysis = generate_ai_analysis(pdf_data, yahoo_data, company_name)
    story.append(Paragraph(ai_analysis, body_style))
    story.append(PageBreak())

    # Charts Section
    story.append(Paragraph("Financial Visualizations", heading_style))

    # Create and add charts
    chart_paths = create_financial_charts(yahoo_data, company_name)
    for chart_path in chart_paths:
      if os.path.exists(chart_path):
        story.append(Image(chart_path, width=6 * inch, height=4 * inch))
        story.append(Spacer(1, 12))

    # Detailed Financial Data
    story.append(PageBreak())
    story.append(Paragraph("Detailed Financial Information", heading_style))

    # Document data if available
    if pdf_data:
      story.append(Paragraph("Document-Based Financial Data", subheading_style))

      # Revenue data
      if pdf_data.get('revenue_data'):
        story.append(Paragraph("Revenue Information:", subheading_style))
        revenue_data = pdf_data.get('revenue_data', {})
        for key, value in revenue_data.items():
          if value and value != 'null':
            story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", body_style))

      # Profitability data
      if pdf_data.get('profitability'):
        story.append(Paragraph("Profitability Metrics:", subheading_style))
        profit_data = pdf_data.get('profitability', {})
        for key, value in profit_data.items():
          if value and value != 'null':
            story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", body_style))

    # Market Data Section
    story.append(Paragraph("Live Market Data", subheading_style))

    market_data = [
      ['Financial Metric', 'Value', 'Industry Context'],
      ['Total Revenue', format_number(yahoo_data.get('totalRevenue')), 'Annual revenue'],
      ['Gross Margins', format_percentage(yahoo_data.get('grossMargins')), 'Operational efficiency'],
      ['Operating Margins', format_percentage(yahoo_data.get('operatingMargins')), 'Core business profitability'],
      ['Total Debt', format_number(yahoo_data.get('totalDebt')), 'Financial leverage'],
      ['Total Cash', format_number(yahoo_data.get('totalCash')), 'Liquidity position'],
      ['Beta', str(yahoo_data.get('beta', 'N/A')), 'Market volatility vs S&P 500'],
      ['Dividend Yield', format_percentage(yahoo_data.get('dividendYield')), 'Shareholder returns']
    ]

    detailed_table = Table(market_data, colWidths=[2 * inch, 1.5 * inch, 2.5 * inch])
    detailed_table.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
      ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, 0), 10),
      ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
      ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
      ('ALTERNATEROWCOLOR', (0, 1), (-1, -1), colors.white, colors.lightgrey),
      ('GRID', (0, 0), (-1, -1), 1, colors.black),
      ('FONTSIZE', (0, 1), (-1, -1), 9)
    ]))

    story.append(detailed_table)
    story.append(Spacer(1, 20))

    # Risk Assessment
    story.append(Paragraph("Risk Assessment & Investment Considerations", heading_style))

    risk_analysis = generate_risk_analysis(pdf_data, yahoo_data)
    story.append(Paragraph(risk_analysis, body_style))

    # Footer with disclaimer
    story.append(Spacer(1, 30))
    disclaimer = """
        <i>Disclaimer: This report is generated using AI analysis and publicly available financial data. 
        It is for informational purposes only and should not be considered as investment advice. 
        Please consult with a qualified financial advisor before making investment decisions.</i>
        """
    story.append(Paragraph(disclaimer, body_style))

    # Build the PDF
    doc.build(story)

    return {
      "success": True,
      "filename": filename,
      "filepath": filepath,
      "charts_created": len(chart_paths)
    }

  except Exception as e:
    return {
      "success": False,
      "error": str(e),
      "filename": None,
      "filepath": None
    }


# ===== EXISTING ROUTES =====

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
  """Upload PDF and extract financial data directly."""
  global current_financial_data, current_pdf_text

  if 'file' not in request.files:
    return jsonify({"error": "No file part"}), 400

  file = request.files['file']
  if file.filename == '':
    return jsonify({"error": "No selected file"}), 400

  # Save uploaded file
  filepath = os.path.join(UPLOAD_FOLDER, file.filename)
  file.save(filepath)

  try:
    # Extract full text from PDF
    text = ""
    with fitz.open(filepath) as pdf:
      for page in pdf:
        text += page.get_text()

    # Store the full text for Q&A context
    current_pdf_text = text

    # Extract financial data using the model
    extraction_result = extract_financial_data(text)

    # Store the extracted data globally
    current_financial_data = extraction_result["financial_data"]

    # Save to file for persistence
    data_filename = file.filename.replace('.pdf', '_financial_data.json')
    data_filepath = os.path.join(FINANCIAL_DATA_FOLDER, data_filename)

    with open(data_filepath, 'w') as f:
      json.dump({
        "filename": file.filename,
        "extraction_result": extraction_result,
        "text_length": len(text)
      }, f, indent=2)

    # Clean up uploaded file
    os.remove(filepath)

    return jsonify({
      "status": "PDF processed successfully",
      "filename": file.filename,
      "extraction_success": extraction_result["extraction_success"],
      "financial_data": current_financial_data,
      "document_stats": {
        "text_length": len(text),
        "has_financial_data": bool(current_financial_data)
      }
    }), 200

  except Exception as e:
    return jsonify({"error":