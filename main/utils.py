import os, pickle, re
from glob import glob
from datetime import datetime
import pandas as pd
from sklearn.metrics import pairwise_distances

def get_latest_dataset(raw_dir, pattern="Consumer_Complaints*.csv"):
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    files = glob(os.path.join(raw_dir, pattern))
    
    if not files:
        from main import dataset_loader
        print("Downloading dataset...")
        dataset_loader.download_dataset()
        files = glob(os.path.join(raw_dir, pattern))
    if not files:
        raise FileNotFoundError("No dataset files found after download.")
    return max(files, key=lambda f: datetime.fromisoformat(date_re.search(f).group(1)))

def load_or_preprocess(preprocessor, dataset_file, cleaned_dir, max_rows=0, column_name="narrative"):
    # Load or preprocess the dataset
    base_name = os.path.basename(dataset_file)
    name_without_ext = os.path.splitext(base_name)[0]
    cleaned_file = os.path.join(cleaned_dir, f"Cleaned_{name_without_ext}.pkl")
    
    if os.path.exists(cleaned_file):
        with open(cleaned_file, "rb") as f:
            return pickle.load(f)
    
    texts = preprocessor.load_data(dataset_file, column_name)
    if max_rows > 0:
        texts = texts[:max_rows]
    
    processed = preprocessor.preprocess_corpus(texts)
    
    with open(cleaned_file, "wb") as f:
        pickle.dump(processed, f)
    return processed

def vectorize_documents(processed_texts, bow_class, tfidf_class, max_features):
    # Convert list of tokenized documents to strings for vectorization
    docs_as_strings = [" ".join(doc) for doc in processed_texts]
    
    bow = bow_class(max_features=max_features)
    X_bow = bow.fit_transform(docs_as_strings)
    df_bow = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names())
    
    tfidf = tfidf_class(max_features=max_features)
    X_tfidf = tfidf.fit_transform(docs_as_strings)
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names())
    
    return X_bow, df_bow, X_tfidf, df_tfidf

def compute_comparison_df(df_bow, df_tfidf, doc_idx=0):
    # Create a DataFrame comparing BoW and TF-IDF vectors for a specific document index
    return pd.DataFrame({
        "Word": df_bow.columns,
        "BoW": df_bow.iloc[doc_idx],
        "TF-IDF": df_tfidf.iloc[doc_idx]
    }).sort_values(by="BoW", ascending=False)
    
def compute_euclidean_distances(X_bow, X_tfidf, n_docs=5):
    # Compute pairwise Euclidean distances for the first n_docs
    dist_bow = pairwise_distances(X_bow[:n_docs], metric="euclidean")
    dist_tfidf = pairwise_distances(X_tfidf[:n_docs], metric="euclidean")
    
    df_dist_bow = pd.DataFrame(dist_bow, columns=[f"Doc{i}" for i in range(n_docs)], index=[f"Doc{i}" for i in range(n_docs)])
    df_dist_tfidf = pd.DataFrame(dist_tfidf, columns=[f"Doc{i}" for i in range(n_docs)], index=[f"Doc{i}" for i in range(n_docs)])
    
    return df_dist_bow, df_dist_tfidf
