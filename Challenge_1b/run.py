import os
import json
import argparse
from datetime import datetime

# Import the functions from your main analysis script.
import app as analyst
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Execution Logic ---

def run_analysis_for_collection(input_json_path: str):
    """
    Loads a single analysis task from a JSON file, runs the full pipeline,
    and saves the output in the same directory.
    """
    collection_dir = os.path.dirname(input_json_path)
    pdf_base_dir = os.path.join(collection_dir, "PDFs")
    output_json_path = os.path.join(collection_dir, "challenge1b_output.json")

    # --- Step 1: Load and Parse Input JSON ---
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input JSON file was not found at '{input_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: The input file at '{input_json_path}' is not a valid JSON file.")
        return

    # Extract information from the JSON data
    persona = data.get("persona", {}).get("role", "Default Persona")
    task = data.get("job_to_be_done", {}).get("task", "Default Task")
    documents = data.get("documents", [])
    
    # --- Step 2: Construct PDF Paths and Validate ---
    pdf_files = [os.path.join(pdf_base_dir, doc["filename"]) for doc in documents]

    for path in pdf_files:
        if not os.path.exists(path):
            print(f"Error: The PDF file '{os.path.basename(path)}' was not found in the '{pdf_base_dir}' directory.")
            return

    print(f"\n--- Starting Analysis for: {os.path.basename(collection_dir)} ---")
    print(f"Persona: {persona}")
    print(f"Task: {task}")
    print(f"Output File: {output_json_path}")
    print(f"Found {len(pdf_files)} PDF(s) to analyze.")
    print("------------------------------------")
    
    start_time = datetime.now()
    try:
        # --- Step 3: Run the Full Analysis Pipeline ---
        result = analyst.process_pdfs(pdf_files, persona, task)
        
        if not result or (not result.get("extracted_sections") and not result.get("sub_section_analysis")):
            print("Analysis did not produce any relevant sections. Exiting.")
            return

        # --- Step 4: Assemble and Save Final Output ---
        processing_time = str(datetime.now() - start_time)
        final_output = {
            "metadata": {
                "input_documents": [os.path.basename(p) for p in pdf_files],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat(),
                # "processing_time": processing_time
            },
            **result
        }
        
        print(f"\n💾 Saving final analysis to {output_json_path}...")
        with open(output_json_path, "w") as f:
            json.dump(final_output, f, indent=2)
        
        print(f"Analysis completed in {processing_time}")
        print(f"Results saved to {output_json_path}")
        print(f"Output contains: {len(result.get('extracted_sections', []))} sections, {len(result.get('sub_section_analysis', []))} subsections")

    except Exception as e:
        print(f"An unexpected error occurred during the analysis: {e}")
    
    finally:
        print("Run complete.")


if __name__ == "__main__":
    # The script will search for collection directories in the same directory it is run from.
    current_directory = "."
    print(f"Searching for collections in the current directory: {os.path.abspath(current_directory)}")

    collection_found = False
    # Iterate over items in the current directory
    for item_name in os.listdir(current_directory):
        item_path = os.path.join(current_directory, item_name)
        # Check if the item is a directory (e.g., "Collection 1")
        if os.path.isdir(item_path):
            input_json_path = os.path.join(item_path, "challenge1b_input.json")
            # Check if the input JSON exists in this directory
            if os.path.exists(input_json_path):
                collection_found = True
                run_analysis_for_collection(input_json_path)

    if not collection_found:
        print("\n No collection directories containing 'challenge1b_input.json' were found.")