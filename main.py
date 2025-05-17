from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Union
import os
import json
import sqlite3
import PyPDF2
from PIL import Image
import pytesseract
import spacy
import re
from datetime import datetime
import google.generativeai as genai
import logging
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import io

# Configure logging with DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MediMitra")

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load scispaCy model
nlp = spacy.load("en_core_sci_sm")

# Load medical terms and health tips
try:
    with open("medical_terms.json", "r", encoding="utf-8") as f:
        MEDICAL_TERMS = json.load(f)
except Exception as e:
    logger.error(f"Failed to load medical_terms.json: {str(e)}")
    MEDICAL_TERMS = {}

try:
    with open("health_tips.json", "r", encoding="utf-8") as f:
        HEALTH_TIPS = json.load(f)
except Exception as e:
    logger.error(f"Failed to load health_tips.json: {str(e)}")
    HEALTH_TIPS = {}

# Initialize database
def init_db():
    conn = sqlite3.connect("medimitra.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY, date TEXT, findings TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS symptoms (id INTEGER PRIMARY KEY, date TEXT, symptoms TEXT)")
    conn.commit()
    conn.close()

init_db()

# Image preprocessing with multiple strategies
def preprocess_image(image: Image.Image, strategy: str = "standard") -> Image.Image:
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:  # Convert RGB to grayscale if needed
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Resize image to increase resolution (2x scaling)
        scale_factor = 2
        img_array = cv2.resize(img_array, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        logger.debug("Image resized for better OCR")

        if strategy == "standard":
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
            logger.debug("Applied CLAHE contrast enhancement")

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
            )
            logger.debug("Applied adaptive thresholding")

        elif strategy == "binary":
            # Apply binary thresholding with Otsu
            _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logger.debug("Applied binary thresholding with Otsu")

        else:  # minimal preprocessing
            thresh = img_array
            logger.debug("Applied minimal preprocessing (grayscale only)")

        # Light dilation to separate characters if needed
        if strategy in ["standard", "binary"]:
            kernel = np.ones((1, 1), np.uint8)  # Reduced kernel size
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            logger.debug("Applied light dilation to separate characters")

        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(thresh)
        logger.debug("Applied denoising")

        # Convert back to PIL Image
        processed_image = Image.fromarray(denoised)
        return processed_image
    except Exception as e:
        logger.error(f"Error preprocessing image with strategy '{strategy}': {str(e)}")
        return image

# Post-process extracted text to fix common OCR issues
def post_process_text(text: str) -> str:
    # Insert spaces before units
    text = re.sub(r"(\d+\.?\d*)(g/dl|cumm|%|fl|pg|lakhs/cumm|million/cumm)", r"\1 \2", text, flags=re.IGNORECASE)
    # Insert spaces between words and numbers
    text = re.sub(r"([A-Z]+)(\d+)", r"\1 \2", text)
    # Fix common OCR mistakes
    text = text.replace("millionfcumm", "million/cumm")
    text = text.replace("conmchc", "con mchc")
    text = text.replace("lymphocyte", "lymphocytes")
    # Add spaces after colons or other separators if missing
    text = re.sub(r"([A-Za-z]+)([0-9]+)", r"\1 \2", text)
    return text

# Utility: Extract values and units
def extract_value_and_unit(text, entity_start, entity_end):
    snippet = text[entity_end : entity_end + 50].lower()
    match = re.search(r"(\d+\.?\d*)\s*([a-z%]+)", snippet)
    if match:
        value = float(match.group(1).replace(",", ""))
        unit = match.group(2).strip()
        return value, unit
    return None, None

# Regex and spaCy fallback analysis
def regex_fallback(text):
    findings = []
    patterns = {
        "hemoglobin": r"hemoglobin\s*(\d+\.?\d*)\s*g/dl",
        "total_leukocyte_count": r"total\s+leukocyte\s+count\s*(\d+\.?\d*)\s*/cumm",
        "neutrophils": r"neutrophils\s*(\d+\.?\d*)\s*%",
        "lymphocytes": r"lymphocytes\s*(\d+\.?\d*)\s*%",
        "eosinophils": r"eosinophils\s*(\d+\.?\d*)\s*%",
        "monocytes": r"monocytes\s*(\d+\.?\d*)\s*%",
        "basophils": r"basophils\s*(\d+\.?\d*)\s*%",
        "platelet_count": r"platelet\s+count\s*(\d+\.?\d*)\s*lakhs/cumm",
        "total_rbc_count": r"total\s+rbc\s+count\s*(\d+\.?\d*)\s*million/cumm",
        "hematocrit_value": r"hematocrit\s+value\s*(\d+\.?\d*)\s*%",
        "mean_corpuscular_volume": r"mean\s+corpuscular\s+volume\s*(\d+\.?\d*)\s*fl",
        "mean_cell_haemoglobin": r"mean\s+cell\s+haemoglobin\s*(\d+\.?\d*)\s*pg",
        "mean_cell_haemoglobin_concentration": r"mean\s+cell\s+haemoglobin\s+con\s*(\d+\.?\d*)\s*%",
    }

    for term, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            try:
                if term not in MEDICAL_TERMS:
                    logger.warning(f"Term '{term}' not found in medical_terms.json. Skipping.")
                    continue
                if "unit" not in MEDICAL_TERMS[term] or "normal_range" not in MEDICAL_TERMS[term]:
                    logger.warning(f"Term '{term}' missing required fields in medical_terms.json. Skipping.")
                    continue

                value = float(match.group(1).replace(",", ""))
                unit = MEDICAL_TERMS[term]["unit"]
                normal_range = MEDICAL_TERMS[term]["normal_range"]
                status = "Normal" if normal_range[0] <= value <= normal_range[1] else "High" if value > normal_range[1] else "Low"

                explanation_key = f"{status.lower()}_explanation"
                if explanation_key not in MEDICAL_TERMS[term] or "en" not in MEDICAL_TERMS[term][explanation_key]:
                    logger.warning(f"Term '{term}' missing explanation for status '{status}'. Using default message.")
                    explanation = f"Your {term.replace('_', ' ')} level is {status.lower()}."
                else:
                    explanation = MEDICAL_TERMS[term][explanation_key]["en"]

                findings.append({
                    "term": term.replace("_", " ").capitalize(),
                    "value": value,
                    "unit": unit,
                    "status": status,
                    "explanation": explanation,
                    "normal_range": f"{normal_range[0]} - {normal_range[1]}",
                    "next_steps": HEALTH_TIPS.get(term, {}).get("next_steps", {}).get("en", ""),
                    "remedies": HEALTH_TIPS.get(term, {}).get("remedies", {}).get("en", "")
                })
            except Exception as e:
                logger.error(f"Error processing term '{term}': {str(e)}")
                continue

    doc = nlp(text)
    for ent in doc.ents:
        term = ent.text.lower().replace(" ", "_")
        try:
            if term in MEDICAL_TERMS:
                value, unit = extract_value_and_unit(text, ent.start_char, ent.end_char)
                if value and unit:
                    normal_range = MEDICAL_TERMS[term]["normal_range"]
                    status = "Normal" if normal_range[0] <= value <= normal_range[1] else "High" if value > normal_range[1] else "Low"

                    explanation_key = f"{status.lower()}_explanation"
                    if explanation_key not in MEDICAL_TERMS[term] or "en" not in MEDICAL_TERMS[term][explanation_key]:
                        explanation = f"Your {term.replace('_', ' ')} level is {status.lower()}."
                    else:
                        explanation = MEDICAL_TERMS[term][explanation_key]["en"]

                    findings.append({
                        "term": term.replace("_", " ").capitalize(),
                        "value": value,
                        "unit": unit,
                        "status": status,
                        "explanation": explanation,
                        "normal_range": f"{normal_range[0]} - {normal_range[1]}",
                        "next_steps": HEALTH_TIPS.get(term, {}).get("next_steps", {}).get("en", ""),
                        "remedies": HEALTH_TIPS.get(term, {}).get("remedies", {}).get("en", "")
                    })
        except Exception as e:
            logger.error(f"Error processing term '{term}' in spaCy NER: {str(e)}")
            continue

    return findings

# Call Gemini API
async def ask_gemini(prompt):
    if not HF_API_KEY:
        logger.error("Missing Google API key")
        return "Error: Missing Google API key."
    try:
        genai.configure(api_key=HF_API_KEY)
        client = genai.GenerativeModel("gemini-1.5-flash")
        response = await client.generate_content_async(prompt)
        logger.info("Gemini API call successful")
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Error from Gemini: {str(e)}"

# Summarize report findings
async def summarize_findings(findings):
    if not findings:
        return "No significant findings were detected in the report."
    summary_input = "\n".join([f"- {f['term']}: {f['value']} {f['unit']} ({f['status']}, Normal range: {f['normal_range']})" for f in findings])
    prompt = (
        f"I am MediMitra, your medical assistant. Summarize the following medical report findings in simple, patient-friendly language. "
        f"Explain what each result means and provide general advice (e.g., see a doctor if abnormal). Avoid technical jargon:\n{summary_input}"
    )
    return await ask_gemini(prompt)

# Analyze report content
async def analyze_report(text):
    findings = regex_fallback(text)
    return findings

# === API Routes ===

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    logger.info(f"Processing file: {file.filename}, type: {file.content_type}")
    try:
        text = ""
        content = await file.read()

        if file.content_type == "application/pdf":
            try:
                pdf = PyPDF2.PdfReader(io.BytesIO(content))
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                logger.info("Extracted text using PyPDF2")
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {str(e)}. Falling back to OCR.")

            if not text.strip():
                logger.info("Converting PDF to images for OCR")
                try:
                    images = convert_from_bytes(content, dpi=300)
                    for i, image in enumerate(images):
                        best_text = ""
                        best_findings_count = 0
                        for strategy in ["standard", "binary", "minimal"]:
                            for psm in [3, 6]:  # PSM 3 (default), PSM 6 (single block)
                                processed_image = preprocess_image(image, strategy=strategy)
                                page_text = pytesseract.image_to_string(
                                    processed_image,
                                    lang="eng",
                                    config=f"--psm {psm} -c preserve_interword_spaces=1"
                                )
                                logger.debug(f"Raw OCR output for PDF page {i+1}, strategy '{strategy}', PSM {psm}:\n{page_text}")
                                page_text = post_process_text(page_text)
                                findings = regex_fallback(page_text)
                                findings_count = len(findings)
                                logger.debug(f"PDF page {i+1}, strategy '{strategy}', PSM {psm}: extracted {findings_count} findings")
                                if findings_count > best_findings_count:
                                    best_text = page_text
                                    best_findings_count = findings_count
                        text += best_text + "\n"
                        logger.info(f"Extracted text from PDF page {i+1} using OCR (best strategy)")
                except Exception as e:
                    logger.error(f"OCR extraction from PDF failed: {str(e)}")
                    raise HTTPException(status_code=400, detail="Unable to extract text from the PDF.")

        elif file.content_type in ["image/png", "image/jpeg"]:
            image = Image.open(io.BytesIO(content))
            best_text = ""
            best_findings_count = 0
            for strategy in ["standard", "binary", "minimal"]:
                for psm in [3, 6]:  # PSM 3 (default), PSM 6 (single block)
                    processed_image = preprocess_image(image, strategy=strategy)
                    text_attempt = pytesseract.image_to_string(
                        processed_image,
                        lang="eng",
                        config=f"--psm {psm} -c preserve_interword_spaces=1"
                    )
                    logger.debug(f"Raw OCR output for image, strategy '{strategy}', PSM {psm}:\n{text_attempt}")
                    text_attempt = post_process_text(text_attempt)
                    findings = regex_fallback(text_attempt)
                    findings_count = len(findings)
                    logger.debug(f"Image extraction, strategy '{strategy}', PSM {psm}: extracted {findings_count} findings")
                    if findings_count > best_findings_count:
                        best_text = text_attempt
                        best_findings_count = findings_count
            text = best_text
            logger.info(f"Extracted text from image using OCR (best strategy, best PSM)")

        else:
            logger.error(f"Unsupported file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        if not text.strip():
            logger.warning("No text extracted from file. Proceeding with empty findings.")
            text = "No text could be extracted from the file."

        logger.debug(f"Final extracted text:\n{text}")

        findings = await analyze_report(text)
        if not findings:
            logger.warning("No findings extracted from the text. The OCR may have failed to detect key terms.")

        summary = await summarize_findings(findings)

        conn = sqlite3.connect("medimitra.db")
        c = conn.cursor()
        c.execute("INSERT INTO reports (date, findings) VALUES (?, ?)", (datetime.now().isoformat(), json.dumps(findings)))
        conn.commit()
        conn.close()

        logger.info("Report analysis completed")
        return {"findings": findings, "summary": summary}
    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MediMitra server error: {str(e)}")

class ChatQuery(BaseModel):
    query: str
    language: str = "en"
    report_date: Union[str, None] = None

@app.post("/chat")
async def chat(query: ChatQuery):
    logger.info(f"Processing chat query: {query.query}")
    try:
        conn = sqlite3.connect("medimitra.db")
        c = conn.cursor()
        if query.report_date:
            c.execute("SELECT findings FROM reports WHERE date LIKE ? ORDER BY date DESC LIMIT 1", (f"{query.report_date}%",))
        else:
            c.execute("SELECT findings FROM reports ORDER BY date DESC LIMIT 1")
        report = c.fetchone()
        conn.close()

        findings_context = ""
        if report:
            findings = json.loads(report[0])
            findings_context = "\n".join([f"- {f['term']}: {f['value']} {f['unit']} ({f['status']}, Normal range: {f['normal_range']})" for f in findings])
            findings_context = f"The patient's most recent medical report contains:\n{findings_context}\n\n"
        else:
            findings_context = "No medical report is available.\n\n"

        prompt = (
            f"I am MediMitra, your medical assistant. Answer the following health question in simple, patient-friendly language. "
            f"If the question relates to the patient's medical report, use the provided findings to inform your answer. "
            f"Avoid technical jargon:\n\n{findings_context}Question: {query.query}"
        )

        response = await ask_gemini(prompt)
        if response.startswith("Error"):
            logger.error(f"Chat error: {response}")
            raise HTTPException(status_code=500, detail=response)
        logger.info("Chat response generated")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MediMitra server error: {str(e)}")

class SymptomLog(BaseModel):
    symptoms: str
    language: str = "en"

@app.post("/symptoms")
async def log_symptoms(symptom: SymptomLog):
    try:
        conn = sqlite3.connect("medimitra.db")
        c = conn.cursor()
        c.execute("INSERT INTO symptoms (date, symptoms) VALUES (?, ?)", (datetime.now().isoformat(), symptom.symptoms))
        conn.commit()
        conn.close()
        logger.info("Symptoms logged successfully")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in /symptoms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MediMitra server error: {str(e)}")

@app.get("/reports")
async def get_reports():
    try:
        conn = sqlite3.connect("medimitra.db")
        c = conn.cursor()
        c.execute("SELECT date, findings FROM reports ORDER BY date DESC")
        reports = [{"date": r[0], "findings": json.loads(r[1])} for r in c.fetchall()]
        conn.close()
        logger.info("Reports retrieved successfully")
        return {"reports": reports}
    except Exception as e:
        logger.error(f"Error in /reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MediMitra server error: {str(e)}")