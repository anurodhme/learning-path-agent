# src/build_index.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os

def build_and_save_index():
    """
    This script performs the following steps:
    1. Loads the Coursera course data.
    2. Generates sentence embeddings for each course title using a pre-trained model.
    3. Builds a FAISS index for efficient similarity search.
    4. Saves the index and corresponding metadata to the 'data/processed' directory.
    """
    
    # --- 1. Load the Pre-trained Sentence Transformer Model ---
    print("Loading the 'all-MiniLM-L6-v2' model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully.")

    # --- 2. Load and Prepare the Data ---
    print("\nLoading and preparing the course data...")
    try:
        # Paths are relative to the project root directory
        df = pd.read_csv("/Users/anurodhbudhathoki/learning-path-agent/data/raw/coursea_data.csv")
        # Handle potential missing titles
        df['course_title'] = df['course_title'].fillna('No Title')
        print(f"Loaded {len(df)} courses.")
    except FileNotFoundError:
        print("Error: data/raw/coursea_data.csv not found. Make sure you are in the project root and have run the download step.")
        return
    except KeyError:
        print("Error: The CSV does not seem to have the 'course_title' column. Please check the file.")
        return

    # Use the course_title as the text source for embedding
    sentences = df['course_title'].tolist()
    print(f"Created {len(sentences)} sentences for embedding from course titles.")

    # --- 3. Generate Embeddings for All Courses ---
    print("\nGenerating embeddings for all courses. This may take a few minutes...")
    start_time = time.time()
    embeddings = model.encode(sentences, show_progress_bar=True)
    end_time = time.time()
    print(f"Embeddings generated successfully in {end_time - start_time:.2f} seconds.")
    print(f"Shape of the embeddings matrix: {embeddings.shape}")

    # --- 4. Build and Save the FAISS Index ---
    print("\nBuilding the FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(embeddings).astype("float32"))
    print(f"FAISS index built. Total entries in index: {index.ntotal}")

    # --- 5. Save the Index and Metadata ---
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "course.index")
    meta_path = os.path.join(output_dir, "meta.parquet")
    
    print(f"\nSaving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Saving metadata to {meta_path}")
    # Save metadata needed to identify courses later. URL is not available.
    # We will save course_title and course_organization.
    meta_df = df[["course_title", "course_organization"]].copy()
    meta_df.to_parquet(meta_path)
    
    print("\n--- Indexing complete! ---")


if __name__ == "__main__":
    build_and_save_index()