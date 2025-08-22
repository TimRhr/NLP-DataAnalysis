import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import os

load_dotenv()

class LSAAnalyzer:
    
    def __init__(self, n_topics=None, n_words=None):
        self.n_topics = n_topics or int(os.getenv("N_TOPICS", 5))
        self.n_words = n_words or int(os.getenv("TOP_N_WORDS", 10))
        self.model = None
        self.feature_names = None

    def fit(self, X, feature_names):
        # X: sparse matrix of shape (n_samples, n_features)
        self.model = TruncatedSVD(n_components=self.n_topics, random_state=42)
        self.model.fit(X)
        self.feature_names = feature_names

    def get_topics(self):
        # Returns a dictionary of topics with their top words
        topics = {}
        for idx, comp in enumerate(self.model.components_):
            total = comp.sum()
            terms_idx = np.argsort(comp)[::-1][:self.n_words]
            topic_terms = [(self.feature_names[i], comp[i] / total * 100) for i in terms_idx]
            topics[f"Topic {idx+1}"] = topic_terms
        return topics

    def print_topics(self):
        for topic, words in self.get_topics().items():
            title_word = words[0][0] if words else ""
            word_str = ", ".join([f"{w} ({p:.1f}%)" for w, p in words])
            print(f"{topic} ({title_word}): {word_str}")


class LDAAnalyzer:
    
    def __init__(self, n_topics=None, max_features=None, top_n_words=None):
        self.n_topics = n_topics or int(os.getenv("N_TOPICS", 5))
        self.max_features = max_features or int(os.getenv("MAX_FEATURES", 100))
        self.top_n_words = top_n_words or int(os.getenv("TOP_N_WORDS", 10))
        self.vectorizer = CountVectorizer(max_features=self.max_features)
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42
        )
        self.feature_names = None

    def fit(self, tokenized_docs):
        # tokenized_docs: list of lists of tokens
        docs_as_strings = [" ".join(doc) for doc in tokenized_docs]
        X = self.vectorizer.fit_transform(docs_as_strings)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.model.fit(X)

    def get_topics(self):
        # Returns a list of topics with their top words
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            total = topic.sum()
            top_indices = topic.argsort()[-self.top_n_words:][::-1]
            top_words = [(self.feature_names[i], topic[i] / total * 100) for i in top_indices]
            topics.append({
                "Topic": topic_idx + 1,
                "Title": top_words[0][0] if top_words else "",
                "Words": top_words
            })
        return topics

    def print_topics(self):
        for t in self.get_topics():
            word_str = ", ".join([f"{w} ({p:.1f}%)" for w, p in t["Words"]])
            print(f"Topic {t['Topic']} ({t['Title']}): {word_str}")

def semantic_analysis(processed_texts):
    docs_as_strings = [" ".join(doc) for doc in processed_texts]

    # LSA
    vectorizer = CountVectorizer(max_features=int(os.getenv("MAX_FEATURES", 100)))
    X = vectorizer.fit_transform(docs_as_strings)
    feature_names = vectorizer.get_feature_names_out()
    print("\n--- LSA Topics ---")
    lsa = LSAAnalyzer()
    lsa.fit(X, feature_names)
    lsa.print_topics()

    # LDA
    print("\n--- LDA Topics ---")
    lda = LDAAnalyzer()
    lda.fit(processed_texts)
    lda.print_topics()