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
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
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
PDF_REPORTS_FOLDER = "pdf_reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FINANCIAL_DATA_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(PDF_REPORTS_FOLDER, exist_ok=True)

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
        'quickRatio': info.get('quickRatio', 'N/A')
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


def create_financial_chart(data, chart_type="bar"):
  """Create financial charts using matplotlib and return as image."""
  plt.style.use('seaborn-v0_8')
  fig, ax = plt.subplots(figsize=(10, 6))

  if chart_type == "bar" and data:
    # Create a sample financial metrics chart
    metrics = list(data.keys())[:6]  # Take first 6 metrics
    values = []

    for metric in metrics:
      val = data[metric]
      if val and val != 'N/A':
        try:
          # Extract numeric value from formatted strings
          if isinstance(val, str):
            if '%' in val:
              values.append(float(val.replace('%', '')))
            elif '$' in val:
              # Handle billion/million notations
              val_clean = val.replace('$', '').replace(',', '')
              if 'B' in val_clean:
                values.append(float(val_clean.replace('B', '')) * 1000)
              elif 'M' in val_clean:
                values.append(float(val_clean.replace('M', '')))
              else:
                values.append(float(val_clean))
            else:
              values.append(float(val))
          else:
            values.append(float(val))
        except:
          values.append(0)
      else:
        values.append(0)

    if values:
      bars = ax.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0'])
      ax.set_title('Key Financial Metrics', fontsize=16, fontweight='bold')
      ax.set_ylabel('Value', fontsize=12)
      plt.xticks(rotation=45, ha='right')

      # Add value labels on bars
      for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.1f}',
                ha='center', va='bottom')

  plt.tight_layout()

  # Save to BytesIO
  img_buffer = BytesIO()
  plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
  img_buffer.seek(0)
  plt.close()

  return img_buffer


def generate_comprehensive_report(pdf_data, yahoo_data):
  """Generate a comprehensive financial report combining PDF and Yahoo data."""
  try:
    # Prepare data summary for the prompt
    pdf_summary = json.dumps(pdf_data, indent=2) if pdf_data else "No PDF data available"
    yahoo_summary = json.dumps(yahoo_data, indent=2) if yahoo_data else "No market data available"

    report_prompt = f"""
        Create a comprehensive, professional financial analysis report for this company. 
        Write in clear, professional language suitable for a financial report.

        Structure the analysis in these sections:

        1. EXECUTIVE SUMMARY (2-3 paragraphs)
        - Key investment thesis and recommendation
        - Overall financial health assessment

        2. COMPANY OVERVIEW (2 paragraphs)
        - Business description and market position
        - Competitive advantages

        3. FINANCIAL PERFORMANCE (3-4 paragraphs)
        - Revenue trends and profitability
        - Key financial ratios analysis
        - Cash flow assessment

        4. VALUATION ANALYSIS (2 paragraphs)
        - Current valuation metrics
        - Peer comparison insights

        5. RISK FACTORS (2 paragraphs)
        - Major operational and market risks
        - Financial stability concerns

        6. INVESTMENT RECOMMENDATION (1-2 paragraphs)
        - Buy/Hold/Sell recommendation with rationale
        - Target price considerations if applicable

        FINANCIAL DOCUMENT DATA:
        {pdf_summary}

        LIVE MARKET DATA:
        {yahoo_summary}

        Make each section substantive and include specific financial metrics and analysis.
        """

    report_messages = [
      {
        "role": "system",
        "content": "You are a senior equity research analyst. Create professional, detailed investment reports with specific insights and clear recommendations."
      },
      {"role": "user", "content": report_prompt}
    ]

    report_response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-3B-Instruct",
      messages=report_messages,
      max_tokens=2048,
      temperature=0.3
    )

    generated_report = report_response.choices[0].message.content

    return {
      "success": True,
      "report_text": generated_report,
      "pdf_data": pdf_data,
      "yahoo_data": yahoo_data
    }

  except Exception as e:
    return {
      "success": False,
      "error": str(e),
      "report_text": None
    }


def create_professional_pdf_report(report_data, company_name, symbol):
  """Create a professional-looking PDF report."""

  # Generate filename
  safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name)
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = f"{safe_company_name}_{symbol}_Financial_Report_{timestamp}.pdf"
  filepath = os.path.join(PDF_REPORTS_FOLDER, filename)

  # Create PDF document
  doc = SimpleDocTemplate(filepath, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)

  # Get styles
  styles = getSampleStyleSheet()

  # Custom styles
  title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#2E86AB'),
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
  )

  heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=16,
    textColor=colors.HexColor('#2E86AB'),
    spaceBefore=20,
    spaceAfter=12,
    fontName='Helvetica-Bold'
  )

  subheading_style = ParagraphStyle(
    'CustomSubHeading',
    parent=styles['Heading3'],
    fontSize=14,
    textColor=colors.HexColor('#4A4A4A'),
    spaceBefore=15,
    spaceAfter=10,
    fontName='Helvetica-Bold'
  )

  body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=11,
    textColor=colors.HexColor('#333333'),
    spaceAfter=12,
    alignment=TA_JUSTIFY,
    fontName='Helvetica'
  )

  # Build content
  story = []

  # Title Page
  story.append(Spacer(1, 0.5 * inch))
  story.append(Paragraph(f"FINANCIAL ANALYSIS REPORT", title_style))
  story.append(Spacer(1, 0.3 * inch))

  # Company header box
  company_data = [
    ['Company:', report_data['yahoo_data'].get('longName', company_name)],
    ['Symbol:', symbol],
    ['Sector:', report_data['yahoo_data'].get('sector', 'N/A')],
    ['Industry:', report_data['yahoo_data'].get('industry', 'N/A')],
    ['Report Date:', datetime.now().strftime('%B %d, %Y')]
  ]

  company_table = Table(company_data, colWidths=[2 * inch, 4 * inch])
  company_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
    ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2E86AB')),
    ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#333333')),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 11),
    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
  ]))

  story.append(company_table)
  story.append(Spacer(1, 0.5 * inch))

  # Key Metrics Table
  story.append(Paragraph("KEY FINANCIAL METRICS", heading_style))

  yahoo_data = report_data['yahoo_data']
  metrics_data = [
    ['Metric', 'Value'],
    ['Market Cap', format_number(yahoo_data.get('marketCap', 'N/A'))],
    ['Current Price', f"${yahoo_data.get('regularMarketPrice', 'N/A')}"],
    ['Daily Change', format_percentage(yahoo_data.get('regularMarketChangePercent', 'N/A'))],
    ['P/E Ratio (TTM)', str(yahoo_data.get('trailingPE', 'N/A'))],
    ['Forward P/E', str(yahoo_data.get('forwardPE', 'N/A'))],
    ['Profit Margin', format_percentage(yahoo_data.get('profitMargins', 'N/A'))],
    ['Revenue Growth', format_percentage(yahoo_data.get('revenueGrowth', 'N/A'))],
    ['ROE', format_percentage(yahoo_data.get('returnOnEquity', 'N/A'))],
    ['Debt-to-Equity', str(yahoo_data.get('debtToEquity', 'N/A'))],
    ['Beta', str(yahoo_data.get('beta', 'N/A'))],
    ['Dividend Yield', format_percentage(yahoo_data.get('dividendYield', 'N/A'))]
  ]

  metrics_table = Table(metrics_data, colWidths=[3 * inch, 2.5 * inch])
  metrics_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ('TOPPADDING', (0, 0), (-1, -1), 6),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
  ]))

  story.append(metrics_table)
  story.append(PageBreak())

  # Add financial chart if we have data
  try:
    chart_data = {
      'Market Cap (B)': yahoo_data.get('marketCap', 0) / 1e9 if yahoo_data.get('marketCap') and yahoo_data.get(
        'marketCap') != 'N/A' else 0,
      'P/E Ratio': yahoo_data.get('trailingPE', 0) if yahoo_data.get('trailingPE') and yahoo_data.get(
        'trailingPE') != 'N/A' else 0,
      'Profit Margin (%)': yahoo_data.get('profitMargins', 0) * 100 if yahoo_data.get(
        'profitMargins') and yahoo_data.get('profitMargins') != 'N/A' else 0,
      'ROE (%)': yahoo_data.get('returnOnEquity', 0) * 100 if yahoo_data.get('returnOnEquity') and yahoo_data.get(
        'returnOnEquity') != 'N/A' else 0,
      'Beta': yahoo_data.get('beta', 0) if yahoo_data.get('beta') and yahoo_data.get('beta') != 'N/A' else 0
    }

    chart_img = create_financial_chart(chart_data, "bar")
    if chart_img:
      story.append(Paragraph("FINANCIAL METRICS VISUALIZATION", heading_style))
      img = Image(chart_img, width=6 * inch, height=3.6 * inch)
      story.append(img)
      story.append(Spacer(1, 0.3 * inch))
  except Exception as e:
    print(f"Error creating chart: {e}")

  # Report Content
  # Report Content
  story.append(Paragraph("DETAILED ANALYSIS", heading_style))

  # Split the report text into sections more intelligently
  report_text = report_data['report_text']

  # First, try to split by common section patterns
  section_patterns = [
    r'\*\*(.*?)\*\*',  # **SECTION NAME**
    r'(\d+\.\s+[A-Z][A-Z\s]+)',  # 1. SECTION NAME
    r'^([A-Z][A-Z\s]{10,}?)$'  # ALL CAPS lines
  ]

  # Clean up the text and split into logical sections
  cleaned_text = report_text.replace('**', '').strip()
  paragraphs = [p.strip() for p in cleaned_text.split('\n') if p.strip()]

  current_section_title = None
  current_section_content = []

  for paragraph in paragraphs:
    # Check if this looks like a section header
    is_header = (
      paragraph.isupper() and len(paragraph) > 5 or  # All caps, reasonable length
      any(keyword in paragraph.upper() for keyword in
          ['EXECUTIVE SUMMARY', 'COMPANY OVERVIEW', 'FINANCIAL PERFORMANCE',
           'VALUATION ANALYSIS', 'RISK FACTORS', 'INVESTMENT RECOMMENDATION']) or
      re.match(r'^\d+\.\s+[A-Z]', paragraph)  # Numbered sections
    )

    if is_header:
      # Add previous section if it exists
      if current_section_title and current_section_content:
        story.append(Paragraph(current_section_title, subheading_style))
        story.append(Spacer(1, 0.1 * inch))

        # Join content paragraphs properly
        section_text = ' '.join(current_section_content)
        story.append(Paragraph(section_text, body_style))
        story.append(Spacer(1, 0.2 * inch))

      # Start new section
      current_section_title = paragraph.title() if paragraph.isupper() else paragraph
      current_section_content = []
    else:
      # Add to current section content
      if paragraph and not paragraph.startswith('|'):  # Skip table-like content
        current_section_content.append(paragraph)

  # Add the last section
  if current_section_title and current_section_content:
    story.append(Paragraph(current_section_title, subheading_style))
    story.append(Spacer(1, 0.1 * inch))

    section_text = ' '.join(current_section_content)
    story.append(Paragraph(section_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

  # Footer
  story.append(Spacer(1, 0.5 * inch))
  
  
  # Build PDF
  doc.build(story)

  return filename, filepath


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
    return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


@app.route('/financial-qa', methods=['GET'])
def financial_qa():
  """Answer questions using extracted financial data and full document context."""
  query = request.args.get("q")
  if not query:
    return jsonify({"error": "Query parameter 'q' is required"}), 400

  if not current_financial_data:
    return jsonify({"error": "No financial data available. Please upload a document first."}), 400

  try:
    # Create context from extracted financial data
    financial_context = json.dumps(current_financial_data, indent=2)

    # Limit document text for context (use relevant portions)
    text_sample = current_pdf_text[:5000] if len(current_pdf_text) > 5000 else current_pdf_text

    qa_prompt = f"""
        You are a financial expert with access to comprehensive financial data from a company document.

        Question: {query}

        Use the extracted financial data and document context below to provide a detailed, accurate answer.
        Include specific numbers, percentages, and financial metrics when relevant.
        If the information isn't available in the data, clearly state that.

        EXTRACTED FINANCIAL DATA:
        {financial_context}

        DOCUMENT CONTEXT:
        {text_sample}
        """

    qa_messages = [
      {"role": "system",
       "content": "You are a financial expert providing precise answers based on company financial documents."},
      {"role": "user", "content": qa_prompt}
    ]

    qa_response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-3B-Instruct",
      messages=qa_messages,
      max_tokens=512,
      temperature=0.2
    )

    answer = qa_response.choices[0].message.content

    return jsonify({
      "question": query,
      "answer": answer,
      "data_available": bool(current_financial_data),
      "context_used": "extracted_financial_data + document_sample"
    })

  except Exception as e:
    return jsonify({"error": f"Q&A processing failed: {str(e)}"}), 500


@app.route('/company-overview', methods=['GET'])
def company_overview():
  """Get complete financial overview of the processed document."""
  if not current_financial_data:
    return jsonify({"error": "No financial data available. Please upload a document first."}), 400

  return jsonify({
    "financial_overview": current_financial_data,
    "data_available": bool(current_financial_data),
    "summary": {
      "has_company_info": bool(current_financial_data.get("company_info")),
      "has_revenue_data": bool(current_financial_data.get("revenue_data")),
      "has_profitability": bool(current_financial_data.get("profitability")),
      "has_financial_position": bool(current_financial_data.get("financial_position")),
      "has_cash_flow": bool(current_financial_data.get("cash_flow")),
      "has_key_metrics": bool(current_financial_data.get("key_metrics")),
      "has_risks_outlook": bool(current_financial_data.get("risks_and_outlook"))
    }
  })


@app.route('/api/company', methods=['GET'])
def get_company():
  """Get company information from Yahoo Finance."""
  company = request.args.get('company')
  if not company:
    return jsonify({"error": "Missing 'company' parameter"}), 400

  symbols = company_name_to_symbol(company)
  if not symbols:
    return jsonify({"error": "No symbols found"}), 404

  info_list = get_company_info_for_symbols(symbols)
  if info_list:
    return jsonify(info_list[0])
  return jsonify({"error": "No info found"}), 404


# ===== NEW PDF REPORT GENERATION ROUTES =====

@app.route('/generate-pdf-report', methods=['GET'])
def generate_pdf_report():
  """Generate comprehensive PDF financial report using both PDF data and Yahoo Finance data."""

  # Get company parameter
  company_name = request.args.get('company')
  if not company_name:
    return jsonify({"error": "Missing 'company' parameter. Usage: /generate-pdf-report?company=Apple"}), 400

  # Check if we have PDF data
  if not current_financial_data:
    return jsonify({"error": "No PDF financial data available. Please upload a document first using /upload-pdf"}), 400

  try:
    print(f"Generating PDF report for company: {company_name}")

    # Get Yahoo Finance data
    symbols = company_name_to_symbol(company_name)
    if not symbols:
      return jsonify({"error": f"No stock symbols found for company: {company_name}"}), 404

    print(f"Found symbols: {symbols}")
    yahoo_data = get_company_info_for_symbols(symbols)[0] if symbols else {}

    # Generate comprehensive report content
    report_result = generate_comprehensive_report(current_financial_data, yahoo_data)

    if report_result["success"]:
      # Create PDF report
      pdf_filename, pdf_filepath = create_professional_pdf_report(
        report_result,
        company_name,
        yahoo_data.get('symbol', 'N/A')
      )

      # Also save JSON data for reference
      json_filename = pdf_filename.replace('.pdf', '_data.json')
      json_filepath = os.path.join(REPORTS_FOLDER, json_filename)

      report_metadata = {
        "report_metadata": {
          "title": f"Financial Analysis Report - {yahoo_data.get('longName', company_name)}",
          "generated_date": datetime.now().strftime("%Y-%m-%d"),
          "company_symbol": yahoo_data.get('symbol', 'N/A'),
          "company_name": yahoo_data.get('longName', company_name),
          "sector": yahoo_data.get('sector', 'N/A'),
          "industry": yahoo_data.get('industry', 'N/A'),
          "pdf_filename": pdf_filename,
          "pdf_filepath": pdf_filepath
        },
        "financial_data": current_financial_data,
        "market_data": yahoo_data,
        "report_text": report_result["report_text"],
        "generation_info": {
          "ai_model": "meta-llama/Llama-3.2-3B-Instruct",
          "generation_timestamp": datetime.now().isoformat()
        }
      }

      with open(json_filepath, 'w') as f:
        json.dump(report_metadata, f, indent=2)

      return jsonify({
        "status": "PDF report generated successfully",
        "company": company_name,
        "symbol": yahoo_data.get('symbol', 'N/A'),
        "pdf_filename": pdf_filename,
        "pdf_path": pdf_filepath,
        "json_data_file": json_filename,
        "report_metadata": report_metadata["report_metadata"],
        "generation_time": datetime.now().isoformat()
      }), 200
    else:
      return jsonify({
        "error": "Failed to generate report content",
        "details": report_result["error"]
      }), 500

  except Exception as e:
    return jsonify({"error": f"PDF report generation failed: {str(e)}"}), 500


@app.route('/download-pdf/<filename>', methods=['GET'])
def download_pdf(filename):
  """Download a specific PDF report."""
  try:
    pdf_filepath = os.path.join(PDF_REPORTS_FOLDER, filename)

    if not os.path.exists(pdf_filepath):
      return jsonify({"error": "PDF report not found"}), 404

    return send_file(
      pdf_filepath,
      as_attachment=True,
      download_name=filename,
      mimetype='application/pdf'
    )

  except Exception as e:
    return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 500


@app.route('/pdf-reports', methods=['GET'])
def list_pdf_reports():
  """List all generated PDF reports."""
  try:
    reports = []

    # Get PDF files
    if os.path.exists(PDF_REPORTS_FOLDER):
      for filename in os.listdir(PDF_REPORTS_FOLDER):
        if filename.endswith('.pdf'):
          try:
            filepath = os.path.join(PDF_REPORTS_FOLDER, filename)
            file_stats = os.stat(filepath)

            # Try to find corresponding JSON data file
            json_filename = filename.replace('.pdf', '_data.json')
            json_filepath = os.path.join(REPORTS_FOLDER, json_filename)

            metadata = {}
            if os.path.exists(json_filepath):
              with open(json_filepath, 'r') as f:
                data = json.load(f)
                metadata = data.get("report_metadata", {})

            # Extract info from filename if no metadata
            if not metadata:
              parts = filename.replace('.pdf', '').split('_')
              if len(parts) >= 3:
                metadata = {
                  "company_name": parts[0].replace('_', ' '),
                  "company_symbol": parts[1] if len(parts) > 1 else 'N/A',
                  "generated_date": "Unknown"
                }

            reports.append({
              "filename": filename,
              "company_name": metadata.get("company_name", "Unknown"),
              "symbol": metadata.get("company_symbol", "N/A"),
              "sector": metadata.get("sector", "N/A"),
              "generated_date": metadata.get("generated_date", "Unknown"),
              "file_size": f"{file_stats.st_size / 1024:.1f} KB",
              "download_url": f"/download-pdf/{filename}",
              "has_json_data": os.path.exists(json_filepath)
            })
          except Exception as e:
            print(f"Error processing PDF report {filename}: {e}")
            continue

    # Sort by generation date (newest first)
    reports.sort(key=lambda x: x.get("generated_date", ""), reverse=True)

    return jsonify({
      "pdf_reports": reports,
      "total_reports": len(reports),
      "pdf_reports_folder": PDF_REPORTS_FOLDER
    })

  except Exception as e:
    return jsonify({"error": f"Failed to list PDF reports: {str(e)}"}), 500


@app.route('/generate-report', methods=['GET'])
def generate_report():
  """Generate JSON financial report (legacy endpoint)."""

  # Get company parameter
  company_name = request.args.get('company')
  if not company_name:
    return jsonify({"error": "Missing 'company' parameter. Usage: /generate-report?company=Apple"}), 400

  # Check if we have PDF data
  if not current_financial_data:
    return jsonify({"error": "No PDF financial data available. Please upload a document first using /upload-pdf"}), 400

  try:
    print(f"Generating JSON report for company: {company_name}")

    # Get Yahoo Finance data
    symbols = company_name_to_symbol(company_name)
    if not symbols:
      return jsonify({"error": f"No stock symbols found for company: {company_name}"}), 404

    print(f"Found symbols: {symbols}")
    yahoo_data = get_company_info_for_symbols(symbols)[0] if symbols else {}

    # Generate comprehensive report
    report_result = generate_comprehensive_report(current_financial_data, yahoo_data)

    if report_result["success"]:
      # Create structured report data
      current_date = datetime.now().strftime("%Y-%m-%d")

      report_data = {
        "report_metadata": {
          "title": f"Financial Analysis Report - {yahoo_data.get('longName', 'Company Analysis')}",
          "generated_date": current_date,
          "company_symbol": yahoo_data.get('symbol', 'N/A'),
          "company_name": yahoo_data.get('longName', 'N/A'),
          "sector": yahoo_data.get('sector', 'N/A'),
          "industry": yahoo_data.get('industry', 'N/A')
        },
        "executive_summary": {
          "full_analysis": report_result["report_text"],
          "key_metrics_snapshot": {
            "market_cap": format_number(yahoo_data.get('marketCap')),
            "current_price": f"${yahoo_data.get('regularMarketPrice', 'N/A')}",
            "daily_change": format_percentage(yahoo_data.get('regularMarketChangePercent')),
            "trailing_pe": yahoo_data.get('trailingPE', 'N/A'),
            "forward_pe": yahoo_data.get('forwardPE', 'N/A'),
            "profit_margin": format_percentage(yahoo_data.get('profitMargins')),
            "revenue_growth": format_percentage(yahoo_data.get('revenueGrowth')),
            "earnings_growth": format_percentage(yahoo_data.get('earningsGrowth')),
            "dividend_yield": format_percentage(yahoo_data.get('dividendYield')),
            "beta": yahoo_data.get('beta', 'N/A'),
            "debt_to_equity": yahoo_data.get('debtToEquity', 'N/A'),
            "roe": format_percentage(yahoo_data.get('returnOnEquity')),
            "current_ratio": yahoo_data.get('currentRatio', 'N/A')
          }
        },
        "document_financial_data": current_financial_data,
        "market_data": yahoo_data,
        "report_generation_info": {
          "data_sources": ["PDF Document Analysis", "Yahoo Finance API"],
          "ai_model": "meta-llama/Llama-3.2-3B-Instruct",
          "generation_timestamp": datetime.now().isoformat()
        }
      }

      # Save report to file
      safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name)
      report_filename = f"{safe_company_name}_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
      report_filepath = os.path.join(REPORTS_FOLDER, report_filename)

      with open(report_filepath, 'w') as f:
        json.dump(report_data, f, indent=2)

      return jsonify({
        "status": "JSON report generated successfully",
        "company": company_name,
        "symbol": yahoo_data.get('symbol', 'N/A'),
        "report": report_data,
        "saved_to": report_filename,
        "generation_time": datetime.now().isoformat(),
        "note": "For PDF reports, use /generate-pdf-report endpoint"
      }), 200
    else:
      return jsonify({
        "error": "Failed to generate report",
        "details": report_result["error"]
      }), 500

  except Exception as e:
    return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


@app.route('/reports', methods=['GET'])
def list_reports():
  """List all generated JSON reports."""
  try:
    reports = []
    if os.path.exists(REPORTS_FOLDER):
      for filename in os.listdir(REPORTS_FOLDER):
        if filename.endswith('.json') and 'comprehensive_report' in filename:
          try:
            filepath = os.path.join(REPORTS_FOLDER, filename)
            with open(filepath, 'r') as f:
              report_data = json.load(f)
              metadata = report_data.get("report_metadata", {})
              reports.append({
                "filename": filename,
                "company_name": metadata.get("company_name", "Unknown"),
                "symbol": metadata.get("company_symbol", "N/A"),
                "sector": metadata.get("sector", "N/A"),
                "generated_date": metadata.get("generated_date", "Unknown"),
                "file_path": filepath
              })
          except Exception as e:
            print(f"Error reading report {filename}: {e}")
            continue

    # Sort by generation date (newest first)
    reports.sort(key=lambda x: x.get("generated_date", ""), reverse=True)

    return jsonify({
      "json_reports": reports,
      "total_reports": len(reports),
      "reports_folder": REPORTS_FOLDER,
      "note": "These are JSON reports. For PDF reports, use /pdf-reports endpoint"
    })

  except Exception as e:
    return jsonify({"error": f"Failed to list reports: {str(e)}"}), 500


@app.route('/report/<filename>', methods=['GET'])
def get_report(filename):
  """Get a specific JSON report by filename."""
  try:
    report_filepath = os.path.join(REPORTS_FOLDER, filename)

    if not os.path.exists(report_filepath):
      return jsonify({"error": "Report not found"}), 404

    with open(report_filepath, 'r') as f:
      report_data = json.load(f)

    return jsonify({
      "status": "Report retrieved successfully",
      "filename": filename,
      "report": report_data
    })

  except Exception as e:
    return jsonify({"error": f"Failed to retrieve report: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
  """Health check endpoint."""
  return jsonify({
    "status": "healthy",
    "api_available": bool(HF_API_KEY),
    "financial_data_loaded": bool(current_financial_data),
    "folders": {
      "uploads": os.path.exists(UPLOAD_FOLDER),
      "financial_data": os.path.exists(FINANCIAL_DATA_FOLDER),
      "reports": os.path.exists(REPORTS_FOLDER),
      "pdf_reports": os.path.exists(PDF_REPORTS_FOLDER)
    },
    "endpoints": [
      "POST /upload-pdf",
      "GET /financial-qa?q=question",
      "GET /company-overview",
      "GET /api/company?company=name",
      "GET /generate-pdf-report?company=name (NEW - Creates PDF)",
      "GET /generate-report?company=name (Legacy - Creates JSON)",
      "GET /pdf-reports (List PDF reports)",
      "GET /download-pdf/<filename> (Download PDF)",
      "GET /reports (List JSON reports)",
      "GET /report/<filename> (Get JSON report)",
      "GET /health"
    ],
    "new_features": [
      "Professional PDF report generation with charts and tables",
      "Modern styling with company branding",
      "Financial metrics visualization",
      "Downloadable PDF reports",
      "Enhanced report structure and formatting"
    ]
  })


if __name__ == '__main__':
  print("Starting Financial PDF Analysis API with Enhanced PDF Report Generation...")
  print(f"Upload folder: {UPLOAD_FOLDER}")
  print(f"Financial data folder: {FINANCIAL_DATA_FOLDER}")
  print(f"JSON reports folder: {REPORTS_FOLDER}")
  print(f"PDF reports folder: {PDF_REPORTS_FOLDER}")
  print("\nKey Features:")
  print("✅ PDF document analysis and financial data extraction")
  print("✅ Yahoo Finance API integration for live market data")
  print("✅ Professional PDF report generation with charts and styling")
  print("✅ JSON reports for API integration")
  print("✅ Financial Q&A system")
  print("✅ Downloadable PDF reports with modern design")
  print("\nMain Endpoints:")
  print("📄 POST /upload-pdf - Upload and analyze financial documents")
  print("🔍 GET /financial-qa?q=question - Ask questions about uploaded documents")
  print("📊 GET /generate-pdf-report?company=name - Generate professional PDF reports")
  print("📋 GET /generate-report?company=name - Generate JSON reports")
  print("📂 GET /pdf-reports - List all PDF reports")
  print("⬇️  GET /download-pdf/<filename> - Download PDF reports")
  print("🏥 GET /health - Health check and available endpoints")
  print("\n" + "=" * 60)
  print("🚀 Server starting on http://0.0.0.0:5000")
  print("=" * 60)
  app.run(debug=True, host='0.0.0.0', port=5000)