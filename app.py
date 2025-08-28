import streamlit as st
from ocr_utils import OCRProcessor, display_ocr_results
from classify_utils import DocumentClassifier, display_classification_results
from entity_utils import EntityExtractor

st.set_page_config(page_title="AI Document OCR & Classification", layout="wide")
st.title("Document OCR & AI Classification System")

uploaded_file = st.file_uploader("Upload document", type=["png", "jpg", "jpeg", "pdf"])

# Optional: OpenAI Key field for secure user input
api_key = st.text_input("Enter your OpenAI API Key (optional):", type="password")

if uploaded_file:
    # Step 1: OCR Extraction
    ocr = OCRProcessor(tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    ocr_result = ocr.process_document(uploaded_file)
    display_ocr_results(ocr_result, uploaded_file)

    # Step 2: Document Classification (Rule-based, AI-powered, or Hybrid)
    if ocr_result.get("text"):
        classifier = DocumentClassifier(api_key=api_key if api_key else None)
        classification_result = classifier.classify_hybrid(ocr_result["text"])
        display_classification_results(classification_result)

        extractor = EntityExtractor()
        entities = extractor.extract_entities(ocr_result["text"])
        st.subheader("🔍 Extracted Entities")
        for ent in entities:
            st.write(f"**{ent['type']}**: {ent['text']}")