import os, pickle, re
from glob import glob
from datetime import datetime
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from tqdm import tqdm

def get_latest_dataset(raw_dir, pattern="Consumer_Complaints*.csv"):
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
    
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
    os.makedirs(cleaned_dir, exist_ok=True)
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

def find_optimal_lda_model(tokenized_docs, min_topics=2, max_topics=15, step=1, passes=10, random_state=42):
    # Optimize LDA model by varying number of topics and evaluating coherence score
    print("\n=== Starte Optimierung der Topic-Anzahl anhand des Coherence Scores ===")

    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    results = []
    best_model = None
    best_score = float('-inf')
    best_num_topics = None

    # test different numbers of topics
    for num_topics in tqdm(range(min_topics, max_topics + 1, step), desc="LDA Modelle"):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=random_state,
            passes=passes,
            alpha='auto',
            eta='auto'
        )

        # calculate coherence score
        cm = CoherenceModel(
            model=model,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        results.append((num_topics, coherence))

        print(f"→ {num_topics:2d} Topics | Coherence Score (c_v): {coherence:.4f}")

        if coherence > best_score:
            best_score = coherence
            best_model = model
            best_num_topics = num_topics

    # plot results
    x, y = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.title("LDA Topic Optimization (Coherence Score)")
    plt.xlabel("Anzahl der Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nBestes Modell: {best_num_topics} Topics mit Coherence Score {best_score:.4f}")

    return best_model, best_num_topics, results