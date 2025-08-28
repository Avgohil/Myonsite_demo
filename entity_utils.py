import re
import spacy

class EntityExtractor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def extract_entities(self, text, doc_type=None):
        entities = []

        # Use regex for amounts, emails, phones, and dates
        amount_matches = re.findall(r"\$[\d,]+(\.\d{2})?|\d+\.\d{2}", text)
        for amt in amount_matches:
            entities.append({"type": "AMOUNT", "text": amt})

        date_matches = re.findall(r"\b\d{2,4}/\d{1,2}/\d{1,4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        for dt in date_matches:
            entities.append({"type": "DATE", "text": dt})

        email_matches = re.findall(r"\b\S+@\S+\.\S+\b", text)
        for em in email_matches:
            entities.append({"type": "EMAIL", "text": em})

        phone_matches = re.findall(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
        for ph in phone_matches:
            entities.append({"type": "PHONE", "text": ph})

        # Use spaCy for general entities
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({"type": ent.label_, "text": ent.text})

        return entities
