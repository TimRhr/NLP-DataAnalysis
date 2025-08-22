from main.preprocessor import Preprocessor
from main.vectorizers import BoWVectorizer, TFIDFVectorizer
from main.utils import get_latest_dataset, load_or_preprocess, vectorize_documents, compute_comparison_df, compute_euclidean_distances
from dotenv import load_dotenv
import os

load_dotenv()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 50))
MAX_FEATURES = int(os.getenv("MAX_FEATURES", 100))
COLUMN_NAME = os.getenv("COLUMN_NAME", "narrative")
MAX_ROWS = int(os.getenv("MAX_ROWS", 0))

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")

if __name__ == "__main__":
    try:
        latest_file = get_latest_dataset(RAW_DIR)
        print(f"Using dataset: {latest_file}")

        pre = Preprocessor()
        processed_texts = load_or_preprocess(pre, latest_file, CLEANED_DIR, MAX_ROWS, COLUMN_NAME)

        X_bow, df_bow, X_tfidf, df_tfidf = vectorize_documents(processed_texts, BoWVectorizer, TFIDFVectorizer, MAX_FEATURES)
        
        comparison_df = compute_comparison_df(df_bow, df_tfidf, doc_idx=2)
        print(comparison_df.head(10))
        
        df_dist_bow, df_dist_tfidf = compute_euclidean_distances(X_bow, X_tfidf, n_docs=5)
        print("BoW Euclidean distances (first 5 docs):")
        print(df_dist_bow)
        print("\nTF-IDF Euclidean distances (first 5 docs):")
        print(df_dist_tfidf)

    except Exception as e:
        print(f"An error occurred: {e}")
