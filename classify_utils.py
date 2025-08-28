

import openai
import json
import re
import streamlit as st
from typing import Dict, List, Tuple

class DocumentClassifier:
    def __init__(self, api_key=None):
        """Initialize document classifier with OpenAI API key"""
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
        
        # Define document categories and their characteristics
        self.categories = {
            "Invoice": {
                "keywords": ["invoice", "bill", "amount due", "total", "tax", "vendor", "invoice number"],
                "patterns": [r"invoice\s*#?\s*\d+", r"total\s*amount", r"due\s*date"],
                "description": "Business invoices, bills, purchase orders"
            },
            "Receipt": {
                "keywords": ["receipt", "store", "purchase", "paid", "change", "subtotal"],
                "patterns": [r"receipt\s*#?\s*\d+", r"total\s*\$?\d+", r"thank\s*you"],
                "description": "Store receipts, payment confirmations"
            },
            "Form": {
                "keywords": ["application", "form", "name:", "date:", "signature", "please fill"],
                "patterns": [r"name\s*:", r"date\s*:", r"signature", r"\[\s*\]", r"please\s+fill"],
                "description": "Application forms, surveys, questionnaires"
            },
            "Contract": {
                "keywords": ["agreement", "contract", "terms", "conditions", "party", "hereby"],
                "patterns": [r"this\s+agreement", r"terms\s+and\s+conditions", r"party\s+of\s+the"],
                "description": "Legal agreements, terms of service"
            },
            "Medical": {
                "keywords": ["patient", "doctor", "prescription", "diagnosis", "medical", "hospital"],
                "patterns": [r"patient\s+name", r"date\s+of\s+birth", r"prescription", r"diagnosis"],
                "description": "Medical records, prescriptions, lab reports"
            },
            "Letter": {
                "keywords": ["dear", "sincerely", "regards", "letter", "correspondence"],
                "patterns": [r"dear\s+\w+", r"sincerely", r"best\s+regards"],
                "description": "Business letters, correspondence"
            },
            "Certificate": {
                "keywords": ["certificate", "diploma", "awarded", "completion", "achievement"],
                "patterns": [r"certificate\s+of", r"hereby\s+certify", r"awarded\s+to"],
                "description": "Diplomas, awards, licenses"
            },
            "Report": {
                "keywords": ["report", "analysis", "summary", "findings", "conclusion"],
                "patterns": [r"executive\s+summary", r"findings", r"conclusion"],
                "description": "Research reports, analysis documents"
            }
        }
    
    def classify_with_rules(self, text: str) -> Dict:
        """Classify document using rule-based approach"""
        text_lower = text.lower()
        scores = {}
        
        for category, info in self.categories.items():
            score = 0
            matched_keywords = []
            matched_patterns = []
            
            # Check keywords
            for keyword in info["keywords"]:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Check patterns
            for pattern in info["patterns"]:
                if re.search(pattern, text_lower):
                    score += 2  # Patterns have higher weight
                    matched_patterns.append(pattern)
            
            scores[category] = {
                "score": score,
                "matched_keywords": matched_keywords,
                "matched_patterns": matched_patterns
            }
        
        # Find best match
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1]["score"])
            
            if best_category[1]["score"] > 0:
                total_possible = len(self.categories[best_category[0]]["keywords"]) + len(self.categories[best_category[0]]["patterns"]) * 2
                confidence = min(best_category[1]["score"] / total_possible, 1.0)
                
                return {
                    "category": best_category[0],
                    "confidence": confidence,
                    "method": "rule_based",
                    "details": best_category[1]
                }
        
        return {
            "category": "Other",
            "confidence": 0.0,
            "method": "rule_based",
            "details": {"score": 0, "matched_keywords": [], "matched_patterns": []}
        }
    
    def classify_with_ai(self, text: str) -> Dict:
        """Classify document using OpenAI API"""
        if not self.api_key:
            return {"error": "OpenAI API key not provided"}
        
        # Truncate text if too long (to stay within token limits)
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        prompt = f"""
        You are a document classification expert. Analyze the provided document text and classify it into ONE of these categories:

        Categories:
        - Invoice: Business invoices, bills, purchase orders
        - Receipt: Store receipts, payment confirmations  
        - Form: Application forms, surveys, questionnaires
        - Contract: Legal agreements, terms of service
        - Medical: Medical records, prescriptions, lab reports
        - Letter: Business letters, correspondence
        - Certificate: Diplomas, awards, licenses
        - Report: Research reports, analysis documents
        - Other: If none of the above categories fit

        Document Text:
        {text}

        Instructions:
        1. Read the document text carefully
        2. Identify key indicators (headers, fields, format)
        3. Choose the most appropriate category
        4. Provide confidence level (0.0-1.0)
        5. Give brief reasoning

        Respond with JSON format:
        {{
            "category": "CategoryName",
            "confidence": 0.95,
            "reasoning": "Brief explanation of classification decision"
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Using 3.5-turbo for speed and cost
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = json.loads(result_text)
            result["method"] = "ai_powered"
            
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract information manually
            return self._parse_ai_response_fallback(result_text)
            
        except Exception as e:
            return {
                "category": "Other",
                "confidence": 0.0,
                "method": "ai_error",
                "error": str(e)
            }
    
    def _parse_ai_response_fallback(self, response_text: str) -> Dict:
        """Fallback parser for AI responses that aren't valid JSON"""
        category_match = re.search(r"category[\":\s]*([A-Za-z]+)", response_text, re.IGNORECASE)
        confidence_match = re.search(r"confidence[\":\s]*(0?\.\d+|1\.0|0)", response_text, re.IGNORECASE)
        
        category = category_match.group(1) if category_match else "Other"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return {
            "category": category,
            "confidence": confidence,
            "method": "ai_fallback",
            "reasoning": "Parsed from non-JSON response"
        }
    
    def classify_hybrid(self, text: str) -> Dict:
        """Use both rule-based and AI classification, return best result"""
        # Get rule-based classification
        rule_result = self.classify_with_rules(text)
        
        # Get AI classification if API key available
        if self.api_key:
            ai_result = self.classify_with_ai(text)
            
            # If both methods agree and have good confidence, boost confidence
            if (rule_result["category"] == ai_result.get("category") and 
                rule_result["confidence"] > 0.3 and ai_result.get("confidence", 0) > 0.7):
                
                return {
                    "category": rule_result["category"],
                    "confidence": min((rule_result["confidence"] + ai_result["confidence"]) / 2 + 0.1, 1.0),
                    "method": "hybrid",
                    "rule_result": rule_result,
                    "ai_result": ai_result
                }
            
            # Otherwise, return the result with higher confidence
            elif ai_result.get("confidence", 0) > rule_result["confidence"]:
                ai_result["method"] = "hybrid_ai_preferred"
                return ai_result
            else:
                rule_result["method"] = "hybrid_rule_preferred"
                return rule_result
        
        # If no AI available, return rule-based result
        return rule_result
    
    def get_category_info(self, category: str) -> Dict:
        """Get information about a specific category"""
        return self.categories.get(category, {
            "keywords": [],
            "patterns": [],
            "description": "Unknown category"
        })
    
    def get_all_categories(self) -> List[str]:
        """Get list of all supported categories"""
        return list(self.categories.keys()) + ["Other"]

# Streamlit integration functions
def display_classification_results(classification_result: Dict):
    """Display classification results in Streamlit"""
    st.subheader("📋 Document Classification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Type", classification_result.get("category", "Unknown"))
    
    with col2:
        confidence = classification_result.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        method = classification_result.get("method", "unknown")
        st.metric("Method", method.replace("_", " ").title())
    
    # Show reasoning if available
    if "reasoning" in classification_result:
        st.write("**Reasoning:**", classification_result["reasoning"])
    
    # Show matched patterns for rule-based classification
    if "details" in classification_result:
        details = classification_result["details"]
        if details.get("matched_keywords"):
            st.write("**Matched Keywords:**", ", ".join(details["matched_keywords"]))
        if details.get("matched_patterns"):
            st.write("**Matched Patterns:**", len(details["matched_patterns"]), "patterns")
    
    # Confidence indicator
    if confidence > 0.8:
        st.success(" High confidence classification")
    elif confidence > 0.5:
        st.warning(" Medium confidence classification")
    else:
        st.error(" Low confidence - manual review recommended")
    
    return classification_result

def create_classification_demo():
    """Create demo data for classification"""
    demo_documents = {
        "Sample Invoice": {
            "text": "INVOICE\nInvoice #: INV-2024-001\nDate: August 28, 2024\nVendor: Tech Solutions Inc.\nBill To: ABC Company\nDescription: Software License\nAmount: $1,250.00\nTax: $125.00\nTotal: $1,375.00\nDue Date: September 15, 2024",
            "expected": "Invoice"
        },
        "Sample Receipt": {
            "text": "SuperMart\nStore #1234\nDate: 08/28/2024\nTime: 14:30\n\nGroceries\nBread          $3.99\nMilk           $4.50\nEggs           $2.99\n\nSubtotal:     $11.48\nTax:           $0.92\nTotal:        $12.40\n\nThank you for shopping!",
            "expected": "Receipt"
        },
        "Sample Form": {
            "text": "APPLICATION FORM\n\nName: ________________\nDate of Birth: ________\nAddress: _____________\nCity: _______________\nPhone: ______________\nEmail: ______________\n\nPlease fill out all fields and sign below.\n\nSignature: ____________\nDate: ________________",
            "expected": "Form"
        }
    }
    
    return demo_documents