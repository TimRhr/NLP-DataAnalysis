from main.preprocessor import Preprocessor
from main.vectorizers import BoWVectorizer, TFIDFVectorizer
from main.topic_models import semantic_analysis
from main.utils import get_latest_dataset, load_or_preprocess, vectorize_documents, compute_comparison_df, compute_euclidean_distances
from dotenv import load_dotenv
import os

load_dotenv()

# configuration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 50))
MAX_FEATURES = int(os.getenv("MAX_FEATURES", 100))
COLUMN_NAME = os.getenv("COLUMN_NAME", "narrative")
MAX_ROWS = int(os.getenv("MAX_ROWS", 0))

# Topic Modeling / Display
N_LSA_TOPICS = int(os.getenv("N_LSA_TOPICS", 5))
N_LDA_TOPICS = int(os.getenv("N_LDA_TOPICS", 5))
N_WORDS_PER_TOPIC = int(os.getenv("N_WORDS_PER_TOPIC", 8))
COMPARE_DOC_IDX = int(os.getenv("COMPARE_DOC_IDX", 2))

# Directory paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")

if __name__ == "__main__":
    try:
        latest_file = get_latest_dataset(RAW_DIR)
        print(f"Using dataset: {latest_file}")

        # Load or preprocess the dataset
        pre = Preprocessor()
        processed_texts = load_or_preprocess(pre, latest_file, CLEANED_DIR, MAX_ROWS, COLUMN_NAME)

        X_bow, df_bow, X_tfidf, df_tfidf = vectorize_documents(processed_texts, BoWVectorizer, TFIDFVectorizer, MAX_FEATURES)
        
        comparison_df = compute_comparison_df(df_bow, df_tfidf, doc_idx=COMPARE_DOC_IDX)
        print(comparison_df.head(10))
        
        # compute and display Euclidean distances
        df_dist_bow, df_dist_tfidf = compute_euclidean_distances(X_bow, X_tfidf, n_docs=5)

        print("BoW Euclidean distances (first 5 docs):")
        print(df_dist_bow)
        print("\nTF-IDF Euclidean distances (first 5 docs):")
        print(df_dist_tfidf)
        
        # analyze topics with LSA and LDA
        semantic_analysis(processed_texts)

    except Exception as e:
        print(f"An error occurred: {e}")
