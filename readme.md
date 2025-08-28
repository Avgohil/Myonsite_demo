## Document OCR & AI Classification System
## Overview
This app uses OCR (Tesseract), AI-based classification, and entity extraction to process uploaded documents (PNG, JPG, JPEG, PDF) with a simple Streamlit UI.

## Environment Setup
## 1. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate 

## 2. Install Python requirements:
pip install -r requirements.txt

## 3.Install Tesseract OCR (required for Windows)

Download from UB Mannheim.

Install to default folder

## 4. Install spaCy language model:
python -m spacy download en_core_web_sm

## Running the Server
## 1.Start the Streamlit app:
streamlit run app.py
## 2. Access the web UI

The app will open automatically or visit http://localhost:8502.

## Usage
Upload document: PDF or image (PNG, JPG, JPEG).

(Optional) Enter your OpenAI API key for advanced classification.

## Results: You’ll see OCR text, document classification, and named entity extraction.

If OCR confidence or extraction is zero, try a clearer document image.

## Troubleshooting
If you see "Tesseract OCR not found"
→ Set correct Tesseract path in your code or reinstall Tesseract.

If no text is extracted
→ Use a high-quality image or scan with printed text.

