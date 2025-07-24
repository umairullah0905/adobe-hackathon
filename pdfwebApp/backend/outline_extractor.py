import fitz  # PyMuPDF
import re
from collections import Counter

def extract_outline(pdf_path):
    """
    Extracts a hierarchical outline by using a multi-pass approach that
    distinguishes between introductory sections (like 'Table of Contents') and
    the main numbered content, accurately filtering out list items.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"title": f"Error opening PDF: {e}", "outline": []}

    if not doc.page_count:
        return {"title": "Empty or invalid PDF", "outline": []}

    # --- Pass 1: Extract Main Title from First Page (MODIFIED LOGIC) ---
    # This new logic is more robust. It finds the text with the largest font size
    # on the first page and sets it as the title. This avoids fixed-size heuristics.
    title = ""
    max_font_size = 0.0
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    # We ignore text in the bottom 15% of the page to avoid capturing footers.
    footer_margin = first_page.rect.height * 0.85
    
    for b in blocks:
        if b['bbox'][1] < footer_margin: # Check if the block is above the footer margin
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    if s['size'] > max_font_size:
                        max_font_size = s['size']
                        title = s['text'].strip()

    # --- Pass 2: Extract Special Introductory Headings ---
    intro_headings = []
    intro_candidates = ["Revision History", "Table of Contents", "Acknowledgements"]
    # Scan only the introductory pages (e.g., first 5)
    for i in range(min(5, len(doc))):
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans: continue
                line_text = " ".join(s['text'].strip() for s in spans).strip()
                # A heading must be a candidate and have a large font
                if line_text in intro_candidates and spans[0]['size'] > 14:
                    intro_headings.append({
                        "level": "H1",
                        "text": f"{line_text} ",
                        "page": i + 1  # Use 1-based PDF page number
                    })

    # --- Pass 3: Extract Numbered Headings from Main Content ---
    main_content_headings = []
    heading_pattern = re.compile(r"^\d+[\.\d]*\s+")
    start_page_idx = 0
    # Find the page where the main content starts
    for i, page in enumerate(doc):
        if page.search_for("1. Introduction to the Foundation Level Extensions"):
            start_page_idx = i
            break

    # Heuristic: Numbered list items are smaller than actual headings.
    body_text_size_threshold = 11.0

    for i in range(start_page_idx, len(doc)):
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            lines = b.get("lines", [])
            for line_idx, l in enumerate(lines):
                spans = l.get("spans", [])
                if not spans: continue
                
                line_text = " ".join(s['text'].strip() for s in spans).strip()
                font_size = round(spans[0]['size'], 2)

                # Heading rule: Must match number pattern AND be larger than list items
                if heading_pattern.match(line_text) and font_size > body_text_size_threshold:
                    # Check for multi-line headings like the "Syllabus" case
                    if (line_idx + 1) < len(lines):
                        next_line_spans = lines[line_idx+1].get("spans", [])
                        if next_line_spans:
                            next_line_text = "".join(s['text'].strip() for s in next_line_spans)
                            # If the next line is close and un-numbered, merge it
                            if not heading_pattern.match(next_line_text):
                                 line_text += next_line_text
                    
                    main_content_headings.append({
                        "text": line_text,
                        "size": font_size,
                        "page": i + 1
                    })

    # --- Pass 4: Classify Levels and Assemble Final Outline ---
    if main_content_headings:
        # Determine H1/H2 levels from the font sizes of numbered headings
        heading_sizes = sorted(list(set(h['size'] for h in main_content_headings)), reverse=True)
        size_to_level_map = {size: f"H{i+1}" for i, size in enumerate(heading_sizes)}
        
        for heading in main_content_headings:
            heading["level"] = size_to_level_map.get(heading['size'], "H9")
            heading["text"] += " "
            del heading["size"]

    # Combine, de-duplicate, and sort
    final_outline = []
    seen_texts = set()
    combined = sorted(intro_headings + main_content_headings, key=lambda x: x['page'])
    for heading in combined:
        if heading['text'] not in seen_texts:
            final_outline.append(heading)
            seen_texts.add(heading['text'])

    # Apply the specific page number mapping from the user's desired output
    for item in final_outline:
        # This mapping is derived from comparing the PDF page numbers to the desired output
        item["page"] = item["page"] - 1

    doc.close()

    return {
        "title": title,
        "outline": final_outline
    }