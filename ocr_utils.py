
import pytesseract
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import io
import os

class OCRProcessor:
    def __init__(self, tesseract_path=None):
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Test if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except:
            self.tesseract_available = False
            st.warning("Tesseract OCR not found. Using basic text extraction.")
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert PIL Image to OpenCV format
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
        else:
            img_cv = image
        
       
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
        
        
        denoised = cv2.medianBlur(gray, 3)
        
       
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_tesseract(self, image, config='--oem 3 --psm 6'):
        """Extract text using Tesseract OCR"""
        if not self.tesseract_available:
            return "Tesseract OCR not available"
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image)
            
            # Extract text with custom config
            text = pytesseract.image_to_string(processed_img, config=config)
            
            return text.strip()
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""
    
    def get_text_confidence(self, image):
        """Get confidence scores for OCR results"""
        if not self.tesseract_available:
            return {}
        
        try:
            processed_img = self.preprocess_image(image)
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'average_confidence': avg_confidence / 100.0,  # Convert to 0-1 scale
                'word_count': len([word for word in data['text'] if word.strip()]),
                'low_confidence_words': sum(1 for conf in confidences if conf < 60)
            }
        except Exception as e:
            return {'average_confidence': 0.0, 'error': str(e)}
    
    def extract_with_multiple_configs(self, image):
        """Try multiple OCR configurations and return best result"""
        configs = [
            '--oem 3 --psm 6',  
            '--oem 3 --psm 3',  
            '--oem 3 --psm 1', 
            '--oem 3 --psm 4',  
        ]
        
        results = []
        
        for config in configs:
            text = self.extract_text_tesseract(image, config)
            confidence_data = self.get_text_confidence(image)
            
            results.append({
                'text': text,
                'confidence': confidence_data.get('average_confidence', 0),
                'config': config
            })
        
        # Return result with highest confidence
        best_result = max(results, key=lambda x: x['confidence'])
        return best_result
    
    def process_document(self, uploaded_file):
        """Main function to process uploaded document"""
        try:
            if uploaded_file.type.startswith('image'):
                # Handle image files
                image = Image.open(uploaded_file)
                
                # Get best OCR result
                result = self.extract_with_multiple_configs(image)
                
                return {
                    'text': result['text'],
                    'confidence': result['confidence'],
                    'method': 'OCR',
                    'config_used': result['config']
                }
            
            elif uploaded_file.type == "application/pdf":
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(uploaded_file.read())
                page_results = []
                for img in images:
                    res = self.extract_with_multiple_configs(img)
                    page_results.append(res)
                full_text = "\n".join([p["text"] for p in page_results])
                avg_conf = sum([p["confidence"] for p in page_results]) / len(page_results)
                
                return {
                    "text": full_text,
                    "confidence": avg_conf,
                    "method": "PDF+OCR",
                    "pages": len(images)
                }

            
            else:
                return {
                    'text': "Unsupported file format",
                    'confidence': 0.0,
                    'method': 'Error'
                }
        
        except Exception as e:
            return {
                'text': "",
                'confidence': 0.0,
                'method': 'Error',
                'error': str(e)
            }

# Utility functions for Streamlit integration
def display_ocr_results(ocr_result, uploaded_file):
    """Display OCR results in Streamlit"""
    st.subheader(" OCR Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Document", use_column_width=True)
    
    with col2:
        st.metric("Extraction Method", ocr_result.get('method', 'Unknown'))
        st.metric("Confidence Score", f"{ocr_result.get('confidence', 0):.1%}")
        
        if 'config_used' in ocr_result:
            st.text(f"OCR Config: {ocr_result['config_used']}")
    
    # Display extracted text
    st.subheader("Extracted Text:")
    if ocr_result.get('text'):
        st.text_area("", ocr_result['text'], height=200, disabled=True)
    else:
        st.warning("No text extracted from document")
    
    return ocr_result

def enhance_image_quality(image):
    """Additional image enhancement functions"""
    img_array = np.array(image)
    
  
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return Image.fromarray(enhanced)