from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS
import fitz  # PyMuPDF
import os
import json
import re
from dotenv import load_dotenv

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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FINANCIAL_DATA_FOLDER, exist_ok=True)

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
        "pages_processed": len(list(fitz.open(filepath))) if os.path.exists(filepath) else "N/A"
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


@app.route('/health', methods=['GET'])
def health_check():
  """Health check endpoint."""
  return jsonify({
    "status": "healthy",
    "api_available": bool(HF_API_KEY),
    "financial_data_loaded": bool(current_financial_data)
  })


if __name__ == '__main__':
  print("Starting Financial PDF Analysis API...")
  print(f"Upload folder: {UPLOAD_FOLDER}")
  print(f"Financial data folder: {FINANCIAL_DATA_FOLDER}")
  app.run(debug=True, host='0.0.0.0', port=5000)