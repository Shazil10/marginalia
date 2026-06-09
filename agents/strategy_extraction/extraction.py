import re
import os

# Minimum non-empty text to consider extraction "successful" (avoid empty or junk output)
MIN_TEXT_LEN = 50


def _extract_pypdf2(filepath, max_chars):
    """Extract text using PyPDF2 (built-in, works for most text-layer PDFs)."""
    import PyPDF2
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "")
            if len(text) >= max_chars:
                break
    return text[:max_chars].strip()


def _extract_pdfplumber(filepath, max_chars):
    """Extract text using pdfplumber (often better for academic/layout-heavy PDFs)."""
    import pdfplumber
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
            if len(text) >= max_chars:
                break
    return text[:max_chars].strip()


def _extract_pymupdf(filepath, max_chars):
    """Extract text using PyMuPDF/fitz (very robust for many PDFs)."""
    import fitz  # PyMuPDF
    text = ""
    doc = fitz.open(filepath)
    try:
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
    finally:
        doc.close()
    return text[:max_chars].strip()


def _extract_ocr_pymupdf(filepath, max_chars):
    """OCR using PyMuPDF to render pages to images + Tesseract. No poppler required."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io
    except ImportError as e:
        print("  OCR fallback skipped: pip install pytesseract Pillow (pymupdf already used above)")
        return ""
    text = ""
    try:
        doc = fitz.open(filepath)
        try:
            # 2x scale ~= 144 DPI, good for OCR
            matrix = fitz.Matrix(2, 2)
            max_pages = min(20, len(doc))
            for i in range(max_pages):
                page = doc[i]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                png_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(png_bytes))
                text += pytesseract.image_to_string(img) + "\n"
                if len(text) >= max_chars:
                    break
        finally:
            doc.close()
    except Exception as e:
        err = str(e).lower()
        if "tesseract" in err or "not found" in err:
            print("  OCR failed: Tesseract not found. Install: brew install tesseract")
        else:
            print(f"  OCR failed: {e}")
        return ""
    return text[:max_chars].strip()


def _extract_ocr_pdf2image(filepath, max_chars):
    """OCR via pdf2image + pytesseract. Requires poppler (brew install poppler)."""
    try:
        import pdf2image
        import pytesseract
    except ImportError:
        return ""
    text = ""
    try:
        images = pdf2image.convert_from_path(filepath, dpi=150, first_page=1, last_page=20)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
            if len(text) >= max_chars:
                break
    except Exception:
        return ""
    return text[:max_chars].strip()


def extract_text_from_pdf(filepath, max_chars=8000):
    """Extracts the first N characters from a PDF using multiple methods with fallbacks."""
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return ""

    extractors = [
        ("PyPDF2", _extract_pypdf2),
        ("pdfplumber", _extract_pdfplumber),
        ("PyMuPDF", _extract_pymupdf),
        ("OCR (PyMuPDF + tesseract)", _extract_ocr_pymupdf),
        ("OCR (pdf2image)", _extract_ocr_pdf2image),  # fallback if poppler is installed
    ]

    for name, extract_fn in extractors:
        try:
            text = extract_fn(filepath, max_chars)
            if text and len(text.strip()) >= MIN_TEXT_LEN:
                print(f"  Extracted {len(text)} chars with {name}.")
                return text
            if text:
                print(f"  {name} returned too little text ({len(text)} chars), trying next...")
        except Exception as e:
            print(f"  {name} failed: {e}")
            continue

    print("All extraction methods failed or returned no usable text.")
    return ""

def run_extraction_agent(client, model, paper_text):
    """Analyzes academic paper text and extracts financial logic into JSON."""
    print("Running Extraction Agent...")
    system_prompt = """
    Act as a quantitative analyst extracting trading strategies from academic papers.
    Extract these fields and return only valid JSON:
    - strategy_name: descriptive name
    - strategy_type: one of momentum, mean_reversion, factor, trend_following, other
    - entry_signal: specific, implementable description of when to buy
    - exit_signal: specific, implementable description of when to sell
    - rebalance_frequency: one of daily, weekly, monthly, quarterly
    - asset_universe: list of tickers or description
    - parameters: dict of all tunable numeric parameters
    - parameter_ranges: dict mapping these parameters to a list of values for grid search (e.g., [10, 20, 30])
    - hidden_params: list of parameters the paper obscurely defines
    - suggested_defaults: conservative default values for hidden params
    - source_paper: author + year string
    - python_indicators_needed: list of technical indicators needed
    Return only valid JSON, no markdown, no explanation.
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Paper text:\n{paper_text}"}
        ],
        temperature=0.1
    )
    
    content = response.choices[0].message.content.strip()
    
    # Clean reasoning blocks if using structured thinking models explicitly
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    # Attempt to extract from markdown JSON block
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        content = match.group(1).strip()
        
    return content