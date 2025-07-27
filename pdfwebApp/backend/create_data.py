# create_dataset.py

import pdfplumber
import json
import pandas as pd
import os
import re

def get_ground_truth(json_path):
    """
    Reads the ground truth JSON.
    --- CHANGE ---
    Returns a lookup for headings AND a set of words for the title for robust matching.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    heading_lookup = {item['text'].strip(): item['level'] for item in data['outline']}
    
    title_text = data.get('title', '').strip()
    title_words = {word for word in title_text.split() if len(word) > 2} if title_text else set()
    
    return heading_lookup, title_words, title_text

# --- NEW: More robust function to identify potential heading patterns ---
def is_potential_heading(text):
    """
    Checks if a line starts with common heading patterns (e.g., numbers, letters, Roman numerals).
    Also checks for keywords like Abstract or Introduction.
    """
    text = text.strip()
    # Regex for 1., 1.1., A., I., etc.
    if re.match(r'^((\d+\.)+|[A-Z]\.|[IVXLCDM]+\.)\s', text):
        return True
    # Check for common non-numbered headings (case-insensitive)
    if re.match(r'^(Abstract|Introduction|Conclusion|References|Index Terms)', text, re.IGNORECASE):
        return True
    return False

# --- NEW: Function to check the text case ---
def get_text_case(text):
    """Determines if the text is mostly uppercase, title case, or none."""
    if text.isupper():
        return 'upper'
    if text.istitle():
        return 'title'
    return 'other'

def extract_features_and_labels(pdf_path, heading_lookup, title_words, full_title_text):
    """Extracts features and labels for every line in the PDF."""
    data_rows = []
    
    with pdfplumber.open(pdf_path) as pdf:
        previous_line_style = {'size': 0, 'fontname': ''}
        
        for page_num, page in enumerate(pdf.pages):
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

                first_word = line_words[0]
                last_word = line_words[-1]
                font_size = first_word['size']
                font_name = first_word['fontname']
                is_bold = 'bold' in font_name.lower()
                size_diff_from_prev = font_size - previous_line_style['size']
                word_count = len(line_words)
                
                # --- FEATURE ENHANCEMENT ---
                starts_with_pattern = is_potential_heading(line_text) # Use new robust function
                text_case = get_text_case(line_text) # Add new text case feature
                
                y_position = top / page.height
                left_margin = first_word['x0']
                right_margin = page_width - last_word['x1']
                # More tolerant centering logic
                is_centered = abs(left_margin - right_margin) < 0.25 * page_width

                label = 'paragraph' 
                if line_text in heading_lookup:
                    label = heading_lookup[line_text]
                elif title_words and page_num == 0: 
                    if line_text == full_title_text:
                         label = 'Title'
                    else:
                        line_word_set = {word for word in line_text.split() if len(word) > 2}
                        if len(line_word_set) > 0 and len(line_word_set.intersection(title_words)) / len(line_word_set) > 0.6:
                            label = 'Title'

                data_rows.append({
                    'text': line_text, 'font_size': font_size, 'is_bold': is_bold,
                    'word_count': word_count, 'size_diff_from_prev': size_diff_from_prev,
                    'starts_with_pattern': starts_with_pattern, # New feature
                    'text_case': text_case,                     # New feature
                    'y_position': y_position, 'is_centered': is_centered,
                    'page_num': page_num, 'label': label
                })
                
                previous_line_style = {'size': font_size, 'fontname': font_name}
    
    return data_rows

# --- Main Execution ---
if __name__ == "__main__":
    PDF_DIR = 'D:/adobe-hackathon/Challenge_1a/sample_dataset/pdfs'
    JSON_DIR = 'D:/adobe-hackathon/Challenge_1a/sample_dataset/outputs'
    
    all_data = []
    
    for pdf_filename in os.listdir(PDF_DIR):
        if pdf_filename.endswith('.pdf'):
            base_name = os.path.splitext(pdf_filename)[0]
            pdf_path = os.path.join(PDF_DIR, pdf_filename)
            json_path = os.path.join(JSON_DIR, f"{base_name}.json")
            
            if not os.path.exists(json_path):
                print(f"Warning: No matching JSON for {pdf_filename}")
                continue
                
            print(f"Processing {pdf_filename}...")
            headings, title_words, full_title = get_ground_truth(json_path)
            all_data.extend(extract_features_and_labels(pdf_path, headings, title_words, full_title))

    df = pd.DataFrame(all_data)
    # --- Convert categorical text_case feature to numerical for the model ---
    df = pd.get_dummies(df, columns=['text_case'], drop_first=True)
    
    df.to_csv('training_datas.csv', index=False, encoding='utf-8')
    print("\nSuccessfully created training_data.csv")