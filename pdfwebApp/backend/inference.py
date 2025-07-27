# process_pdfs.py (inference - for processing a whole directory)

import pdfplumber
import json
import pandas as pd
import os
import re
import joblib
import sys

# --- Load the trained model ---
MODEL_PATH = 'heading_classifier.joblib'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    sys.exit(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")

# --- HELPER FUNCTIONS (These must be identical to the ones in create_data.py) ---
def is_potential_heading(text):
    """
    Checks if a line starts with common heading patterns (e.g., numbers, letters, Roman numerals)
    or keywords.
    """
    text = text.strip()
    # Regex for 1., 1.1., A., I., etc.
    if re.match(r'^((\d+\.)+|[A-Z]\.|[IVXLCDM]+\.)\s', text):
        return True
    # Check for common non-numbered headings (case-insensitive)
    if re.match(r'^(Abstract|Introduction|Conclusion|References|Index Terms)', text, re.IGNORECASE):
        return True
    return False

def get_text_case(text):
    """Determines if the text is mostly uppercase, title case, or none."""
    if text.isupper():
        return 'upper'
    if text.istitle():
        return 'title'
    return 'other'

def predict_structure(pdf_path):
    """
    Processes a new PDF and predicts its structure using the trained model.
    """
    outline = []
    potential_title_parts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not any(page.extract_text(x_tolerance=2) for page in pdf.pages):
                 print(f"Warning: PDF '{os.path.basename(pdf_path)}' has no extractable text.")
                 return {"title": "", "outline": []}

            previous_line_style = {'size': 0, 'fontname': ''}
            
            for page_num, page in enumerate(pdf.pages):
                page_width = page.width
                page_height = page.height
                
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

                    # --- FEATURE EXTRACTION (Must exactly match the training script) ---
                    first_word = line_words[0]
                    last_word = line_words[-1]
                    
                    font_size = first_word['size']
                    font_name = first_word['fontname']
                    is_bold = 'bold' in font_name.lower()
                    size_diff_from_prev = font_size - previous_line_style['size']
                    word_count = len(line_words)
                    
                    starts_with_pattern = is_potential_heading(line_text)
                    text_case = get_text_case(line_text)

                    y_position = top / page_height
                    left_margin = first_word['x0']
                    right_margin = page_width - last_word['x1']
                    is_centered = abs(left_margin - right_margin) < 0.25 * page_width
                    
                    # Create a dictionary to build the feature vector safely
                    feature_dict = {
                        'font_size': font_size, 'is_bold': is_bold, 'word_count': word_count,
                        'size_diff_from_prev': size_diff_from_prev, 'starts_with_pattern': starts_with_pattern,
                        'y_position': y_position, 'is_centered': is_centered,
                        'text_case_title': 1 if text_case == 'title' else 0,
                        'text_case_upper': 1 if text_case == 'upper' else 0
                    }
                    
                    # Ensure the order of columns matches the model's training
                    feature_vector = pd.DataFrame([feature_dict], columns=model.feature_names_in_)

                    predicted_label = model.predict(feature_vector)[0]

                    # --- LOGIC TO HANDLE PREDICTIONS ---
                    if predicted_label == 'Title' and page_num == 0 and y_position < 0.5:
                        potential_title_parts.append(line_text)
                    elif predicted_label in ['H1', 'H2', 'H3']:
                        outline.append({
                            "level": predicted_label, "text": line_text, "page": page_num
                        })
                    
                    previous_line_style = {'size': font_size, 'fontname': font_name}

    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return {"title": "", "outline": []}

    doc_title = " ".join(potential_title_parts)
    return {"title": doc_title, "outline": outline}


# --- Main execution block to process all PDFs in a directory ---
if __name__ == "__main__":
    # Define the input and output directories
    INPUT_DIR = 'D:/adobe-hackathon/Challenge_1a/sample_dataset/pdfs'
    OUTPUT_DIR = 'pdfwebApp/currout'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Check if the input directory exists
    if not os.path.isdir(INPUT_DIR):
        sys.exit(f"Error: Input directory not found at '{INPUT_DIR}'")

    # Loop through all files in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.pdf'):
            print(f"Processing {filename}...")
            pdf_path = os.path.join(INPUT_DIR, filename)
            
            # Get the structured data from the prediction function
            result = predict_structure(pdf_path)
            
            # Define the output path for the corresponding JSON file
            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Save the result to a JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            
            print(f"Saved output to {output_path}\n")
    
    print("--- All PDF files processed. ---")