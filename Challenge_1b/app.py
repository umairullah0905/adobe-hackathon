import argparse
import json
import os
import re
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

import fitz  # PyMuPDF
import ollama
from sentence_transformers import CrossEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# --- Constants ---
MAX_PAGES_TO_SCAN = 50  # Per document

# --- LLM Prompts ---

LLM_RERANK_PROMPT = """
You are an expert {persona}. Your task is to analyze the following sections of text and re-rank them based on their usefulness for accomplishing this specific job: "{task}".

**CRITICAL RULES:**
1.  Your entire response MUST be a valid JSON list of the original indices, ordered from most useful to least useful.
2.  Do not include any other text, explanations, or markdown formatting.
3.  The list should start with `[` and end with `]`.

---
**SECTIONS TO RANK (with their original index):**
{sections_to_rank}
---

Produce the JSON list of indices now.
Example: [3, 1, 0, 2]
"""


# --- Core Functions ---

def _calculate_median_font_size(doc: fitz.Document) -> float:
    """Calculates the median font size across the first few pages."""
    font_sizes = []
    for page_num in range(min(len(doc), 5)):
        page = doc[page_num]
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
        except Exception:
            continue
    return statistics.median(font_sizes) if font_sizes else 12.0

def _is_heading(block: Dict, median_font_size: float) -> bool:
    """Heuristic: Text is a heading if its font is larger or bold."""
    if not block.get("lines") or not block["lines"][0].get("spans"):
        return False
    first_span = block["lines"][0]["spans"][0]
    # A heading is likely larger OR bold (flag 16)
    is_bold = first_span["flags"] & 16
    is_large = first_span["size"] > median_font_size * 1.15
    return is_large or is_bold

def extract_document_structure(pdf_path: str) -> Dict:
    """Extracts structured sections (heading + content) from a PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Could not open {pdf_path}: {e}")
        return {"title": os.path.basename(pdf_path), "sections": []}

    median_font_size = _calculate_median_font_size(doc)
    doc_title = os.path.basename(pdf_path)
    
    structure = {"title": doc_title, "sections": []}
    current_heading = "Introduction"
    current_content = []
    
    for page_num in range(min(len(doc), MAX_PAGES_TO_SCAN)):
        page = doc[page_num]
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()
                    if not text:
                        continue

                    if _is_heading(block, median_font_size):
                        if current_heading and current_content:
                            structure["sections"].append({
                                "heading": current_heading,
                                "content": " ".join(current_content),
                                "page": page_num
                            })
                        current_content = []
                        current_heading = text
                    else:
                        current_content.append(text)
        except Exception:
            continue

    if current_heading and current_content:
        structure["sections"].append({
            "heading": current_heading,
            "content": " ".join(current_content),
            "page": len(doc)-1
        })
        
    return structure

def call_llm_for_ranking(prompt: str) -> List[int]:
    """Safely calls the LLM for a JSON list of indices and handles errors."""
    try:
        response = ollama.chat(
            model="tinyllama:1.1b-chat-v0.6-q2_K",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        ranked_indices = json.loads(response["message"]["content"])
        if isinstance(ranked_indices, list) and all(isinstance(i, int) for i in ranked_indices):
            return ranked_indices
        else:
            print("LLM did not return a valid list of integers.")
            return []
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"LLM ranking failed: {e}.")
        return []

def run_hybrid_analysis(structures: List[Dict], persona: str, task: str) -> dict:
    """
    Runs a hybrid analysis using a CrossEncoder for initial search and an LLM for final ranking.
    """
    super_query = f"{persona} needs to: {task}"
    print(f"\nSynthesizing query for analysis: \"{super_query}\"")

    all_sections = []
    for doc in structures:
        for section in doc["sections"]:
            section['source_doc'] = doc["title"]
            all_sections.append(section)

    if not all_sections:
        print("No sections were extracted from the documents.")
        return {"extracted_sections": [], "sub_section_analysis": []}

    # Step 1: Fast Semantic Search with CrossEncoder
    print(f"Performing fast semantic search on {len(all_sections)} sections...")
    encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device = "cpu")
    
    contents_to_rank = [f"{section['heading']}\n{section['content']}" for section in all_sections]
    cross_encoder_scores = encoder.predict([(super_query, content) for content in contents_to_rank])
    
    semantically_ranked_chunks = sorted(
        zip(all_sections, cross_encoder_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Filter out low-value sections like "Introduction" and "Conclusion"
    # print("✅ Filtering out generic sections like 'Introduction' and 'Conclusion'...")
    blacklist = ["introduction", "conclusion", "references", "abstract"]
    high_value_candidates = [
        section for section, score in semantically_ranked_chunks 
        if section['heading'].lower().strip() not in blacklist
    ][:20] # Take the top 20 high-value candidates

    if not high_value_candidates:
        print("No high-value sections found after filtering. Using original candidates.")
        high_value_candidates = [section for section, score in semantically_ranked_chunks[:20]]


    # Step 2: Intelligent Persona Ranking with LLM
    # print(f"🤖 Using LLM to 'think' and re-rank {len(high_value_candidates)} candidate sections...")
    
    sections_for_prompt = "\n".join([f"{i}: {s['heading']} - {s['content'][:150]}..." for i, s in enumerate(high_value_candidates)])
    
    rerank_prompt = LLM_RERANK_PROMPT.format(
        persona=persona,
        task=task,
        sections_to_rank=sections_for_prompt
    )
    
    ranked_indices = call_llm_for_ranking(rerank_prompt)
    
    if ranked_indices:
        final_ranked_sections = [high_value_candidates[i] for i in ranked_indices if i < len(high_value_candidates)]
        final_ranked_sections.extend([s for s in high_value_candidates if s not in final_ranked_sections])
        print(f"LLM successfully re-ranked the sections.")
    else:
        print("Falling back to CrossEncoder ranking due to LLM failure.")
        final_ranked_sections = high_value_candidates

    # Step 3: Generate Final Output
    print("Formatting top sections and subsections...")
    extracted_sections_out = []
    sub_section_analysis_out = []
    for rank, section_data in enumerate(final_ranked_sections[:10], 1):
        extracted_sections_out.append({
            "document": section_data.get("source_doc", "unknown"),
            "page_number": section_data.get("page", 0),
            "section_title": section_data.get("heading", "Key Section"),
            "importance_rank": rank
        })
        
        if rank <= 5:
            sub_section_analysis_out.append({
                "document": section_data.get("source_doc", "unknown"),
                "page_number": section_data.get("page", 0),
                "refined_text": section_data.get("content", "")
                # "importance_rank": rank
            })
        
    return {"extracted_sections": extracted_sections_out, "sub_section_analysis": sub_section_analysis_out}

# --- Full Execution Flow ---

def process_pdfs(pdf_paths: list, persona: str, task: str) -> dict:
    """Runs the entire analysis pipeline from PDF processing to final output generation."""
    print("--- Step 1: Extracting Document Structures ---")
    with ThreadPoolExecutor() as executor:
        structures = list(executor.map(extract_document_structure, pdf_paths))
    
    print("\n--- Step 2: Running Hybrid Analysis ---")
    return run_hybrid_analysis(structures, persona, task)
