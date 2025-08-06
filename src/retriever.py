# src/retriever.py

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class CourseRetriever:
    """
    A class to handle loading the FAISS index and course metadata,
    and performing semantic searches for courses.
    """
    def __init__(self, index_path="data/processed/course.index", meta_path="data/processed/meta.parquet"):
        print("Initializing CourseRetriever...")
        
        # --- Load FAISS index ---
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run build_index.py first.")
        self.index = faiss.read_index(index_path)
        print(f"FAISS index loaded with {self.index.ntotal} vectors.")

        # --- Load metadata ---
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}. Please run build_index.py first.")
        self.meta_df = pd.read_parquet(meta_path)
        print(f"Metadata loaded for {len(self.meta_df)} courses.")

        # --- Load the embedding model ---
        # This must be the same model used to create the embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer model 'all-MiniLM-L6-v2' loaded.")

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Performs a semantic search for a given query.

        Args:
            query (str): The user's search query (e.g., "data science for beginners").
            top_k (int): The number of top results to return.

        Returns:
            list[str]: A list of the titles of the top_k most relevant courses.
        """
        if top_k <= 0:
            return []
            
        print(f"\nSearching for top {top_k} courses with query: '{query}'")
        
        # 1. Encode the query into an embedding
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")

        # 2. Search the FAISS index
        # D: distances, I: indices (IDs) of the vectors
        distances, indices = self.index.search(query_embedding, top_k)

        # 3. Retrieve the course titles using the indices
        # The indices[0] contains the list of IDs for our single query
        retrieved_ids = indices[0]
        
        # Filter out potential -1 indices if k is larger than the index size
        retrieved_ids = [idx for idx in retrieved_ids if idx != -1]

        results = self.meta_df.iloc[retrieved_ids]
        
        print(f"Found {len(results)} relevant courses.")
        return results['course_title'].tolist()

# Example of how to use the retriever (optional, for testing)
if __name__ == '__main__':
    try:
        retriever = CourseRetriever()
        search_query = "Introduction to Python programming"
        retrieved_courses = retriever.search(search_query, top_k=5)
        
        print("\n--- Search Results ---")
        if retrieved_courses:
            for i, course in enumerate(retrieved_courses):
                print(f"{i+1}. {course}")
        else:
            print("No courses found.")
            
    except FileNotFoundError as e:
        print(e)