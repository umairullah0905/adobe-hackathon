# process_pdfs.py (Updated function)

import pdfplumber
import json
import pandas as pd
import os
import re
import joblib
import sys

# --- Load the model as before ---
MODEL_PATH = 'heading_classifier.joblib'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    sys.exit(f"Error: Model file not found at {MODEL_PATH}")

def predict_structure(pdf_path):
    """
    Processes a new PDF and predicts its structure using the trained model.
    """
    outline = []
    potential_title_parts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        if not any(page.extract_text(x_tolerance=2) for page in pdf.pages):
             print(f"Warning: PDF '{os.path.basename(pdf_path)}' has no extractable text.")
             return {"title": "", "outline": []}

        previous_line_style = {'size': 0, 'fontname': ''}
        
        for page_num, page in enumerate(pdf.pages):
            # --- NEW: Get page width for calculating centered-ness ---
            page_width = page.width
            
            words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, extra_attrs=["fontname", "size"])
            if not words:
                continue

            lines = {}
            for word in words:
                top = round(word['top'])
                if top not in lines:
                    lines[top] = []
                lines[top].append(word)

            for top in sorted(lines.keys()):
                line_words = sorted(lines[top], key=lambda w: w['x0'])
                line_text = " ".join(w['text'] for w in line_words).strip()
                
                if not line_text:
                    continue

                # --- FEATURE EXTRACTION: Must match training script exactly ---
                first_word = line_words[0]
                last_word = line_words[-1]
                
                font_size = first_word['size']
                font_name = first_word['fontname']
                is_bold = 'bold' in font_name.lower()
                size_diff_from_prev = font_size - previous_line_style['size']
                word_count = len(line_words)
                starts_with_numbering = bool(re.match(r'^\d+(\.\d+)*', line_text))
                
                # --- NEW FEATURE 1: Vertical Position (y_position) ---
                y_position = top / page.height
                
                # --- NEW FEATURE 2: Centered-ness (is_centered) ---
                left_margin = first_word['x0']
                right_margin = page_width - last_word['x1']
                is_centered = abs(left_margin - right_margin) < 0.2 * page_width
                
                # Create feature vector with the new features included
                feature_vector = pd.DataFrame([[
                    font_size, is_bold, word_count, 
                    size_diff_from_prev, starts_with_numbering,
                    y_position, is_centered  # <-- NEW
                ]], columns=model.feature_names_in_)
                # --- END OF FEATURE EXTRACTION ---

                predicted_label = model.predict(feature_vector)[0]

                if predicted_label == 'Title' and page_num == 0:
                    potential_title_parts.append(line_text)
                elif predicted_label in ['H1', 'H2', 'H3']:
                    outline.append({
                        "level": predicted_label, 
                        "text": line_text, 
                        "page": page_num + 1
                    })
                
                previous_line_style = {'size': font_size, 'fontname': font_name}

    doc_title = " ".join(potential_title_parts)
    
    # Fallback Logic (unchanged but now more likely to succeed)
    if not doc_title:
        first_page_h1s = [item for item in outline if item['level'] == 'H1' and item['page'] == 1]
        if first_page_h1s:
            title_item = first_page_h1s[0]
            doc_title = title_item['text']
            outline.remove(title_item)
    
    return {"title": doc_title, "outline": outline}


# --- Main execution block remains the same ---
if __name__ == "__main__":
    INPUT_DIR = 'D:/adobe-hackathon/Challenge_1a/sample_dataset/pdfs'
    OUTPUT_DIR = 'pdfwebApp/currout'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.pdf'):
            print(f"Processing {filename}...")
            pdf_path = os.path.join(INPUT_DIR, filename)
            result = predict_structure(pdf_path)
            
            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            
            print(f"Saved output to {output_path}")