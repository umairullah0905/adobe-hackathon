# process_pdfs.py (Corrected logic, processing all files in a directory)

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
    sys.exit(f"Error: Model file not found at {MODEL_PATH}")

def predict_structure(pdf_path):
    """
    Processes a new PDF and predicts its structure using the trained model.
    Applies logic to differentiate between titles and bottom-of-page headings.
    """
    outline = []
    potential_title_parts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check if the PDF contains any extractable text to avoid errors
            if not any(page.extract_text(x_tolerance=2) for page in pdf.pages):
                 print(f"Warning: PDF '{os.path.basename(pdf_path)}' has no extractable text.")
                 return {"title": "", "outline": []}

            previous_line_style = {'size': 0, 'fontname': ''}
            
            for page_num, page in enumerate(pdf.pages):
                page_width = page.width
                page_height = page.height
                
                # Extract words to reconstruct lines accurately
                words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, extra_attrs=["fontname", "size"])
                if not words:
                    continue

                # Group words into lines based on their vertical position
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

                    # --- FEATURE EXTRACTION (Must match the training script) ---
                    first_word = line_words[0]
                    last_word = line_words[-1]
                    
                    font_size = first_word['size']
                    font_name = first_word['fontname']
                    is_bold = 'bold' in font_name.lower()
                    size_diff_from_prev = font_size - previous_line_style['size']
                    word_count = len(line_words)
                    starts_with_numbering = bool(re.match(r'^\d+(\.\d+)*', line_text))
                    y_position = top / page_height
                    left_margin = first_word['x0']
                    right_margin = page_width - last_word['x1']
                    is_centered = abs(left_margin - right_margin) < 0.2 * page_width
                    
                    # Create a feature vector for the model
                    feature_vector = pd.DataFrame([[
                        font_size, is_bold, word_count, 
                        size_diff_from_prev, starts_with_numbering,
                        y_position, is_centered
                    ]], columns=model.feature_names_in_)

                    # Predict the label for the line
                    predicted_label = model.predict(feature_vector)[0]

                    # --- LOGIC TO HANDLE PREDICTIONS ---
                    # If the model predicts 'Title', apply the positional check.
                    if predicted_label == 'Title' and page_num == 0:
                        # A true title should be in the top half of the first page.
                        if y_position < 0.5:
                            potential_title_parts.append(line_text)
                        else:
                            # If it's in the bottom half, it's a heading, not a title.
                            # Reclassify it as H1 to match the desired output format.
                            outline.append({
                                "level": "H1", 
                                "text": line_text, 
                                "page": page_num # Use 0-based index for consistency
                            })
                    elif predicted_label in ['H1', 'H2', 'H3']:
                        outline.append({
                            "level": predicted_label, 
                            "text": line_text, 
                            "page": page_num
                        })
                    
                    previous_line_style = {'size': font_size, 'fontname': font_name}

    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return {"title": "", "outline": []}


    doc_title = " ".join(potential_title_parts)
    
    # This logic ensures that if no title is found in the top half, the title field remains empty,
    # and the headings are kept in the outline, matching your required output for the invitation PDF.
    
    return {"title": doc_title, "outline": outline}


# --- Main execution block to process all PDFs in a directory ---
if __name__ == "__main__":
    INPUT_DIR = 'D:/adobe-hackathon/Challenge_1a/sample_dataset/pdfs'
    OUTPUT_DIR = 'pdfwebApp/currout'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
            
            print(f"Saved output to {output_path}")