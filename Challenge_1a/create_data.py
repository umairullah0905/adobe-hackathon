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
    
    # Lookup for H1, H2, etc. remains the same
    heading_lookup = {item['text'].strip(): item['level'] for item in data['outline']}
    
    # --- NEW: Create a set of all words in the title for multi-line matching ---
    title_text = data.get('title', '').strip()
    # Create a word set, ignoring very short words that might cause false positives
    title_words = {word for word in title_text.split() if len(word) > 2} if title_text else set()
    
    return heading_lookup, title_words, title_text

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
                starts_with_numbering = bool(re.match(r'^\d+(\.\d+)*', line_text))
                y_position = top / page.height
                left_margin = first_word['x0']
                right_margin = page_width - last_word['x1']
                is_centered = abs(left_margin - right_margin) < 0.2 * page_width

                # --- NEW, SMARTER LABELING LOGIC ---
                label = 'paragraph' # Default label

                # 1. Check for an exact match for a heading first (most reliable).
                if line_text in heading_lookup:
                    label = heading_lookup[line_text]
                # 2. If not a heading, check if it's likely part of the title.
                elif title_words and page_num == 0: # Only look for titles on the first page
                    # Check for an exact match of the full title
                    if line_text == full_title_text:
                         label = 'Title'
                    # Check for multi-line titles using word overlap
                    else:
                        line_word_set = {word for word in line_text.split() if len(word) > 2}
                        # If more than 60% of the words in this line are in the title, label it as Title.
                        # This threshold is robust to small mismatches.
                        if len(line_word_set) > 0 and len(line_word_set.intersection(title_words)) / len(line_word_set) > 0.6:
                            label = 'Title'

                data_rows.append({
                    'text': line_text, 'font_size': font_size, 'is_bold': is_bold,
                    'word_count': word_count, 'size_diff_from_prev': size_diff_from_prev,
                    'starts_with_numbering': starts_with_numbering, 'y_position': y_position,
                    'is_centered': is_centered, 'page_num': page_num, 'label': label
                })
                
                previous_line_style = {'size': font_size, 'fontname': font_name}
    
    return data_rows

# --- Main Execution (Updated to handle new return values) ---
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
            # --- CHANGE: Get the new outputs from the ground truth function ---
            headings, title_words, full_title = get_ground_truth(json_path)
            all_data.extend(extract_features_and_labels(pdf_path, headings, title_words, full_title))

    df = pd.DataFrame(all_data)
    df.to_csv('D:/adobe-hackathon/Challenge_1a/training_datas.csv', index=False, encoding='utf-8')
    print("\nSuccessfully created training_data.csv")