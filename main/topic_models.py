import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import os
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from main.utils import visualize_lda_model

from main.utils import find_optimal_lda_model

load_dotenv()

class LSAAnalyzer:
    
    def __init__(self, n_topics=None, max_features=None, top_n_words=None):
        self.n_topics = n_topics or int(os.getenv("N_TOPICS", 5))
        self.max_features = max_features or int(os.getenv("MAX_FEATURES", 100))
        self.n_words = top_n_words or int(os.getenv("TOP_N_WORDS", 10))
        self.model = None
        self.feature_names = None
        self.vectorizer = CountVectorizer(max_features=self.max_features)

    def fit(self, X, feature_names):
        # X: sparse matrix of shape (n_samples, n_features)
        X = self.vectorizer.fit_transform(feature_names)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.model = TruncatedSVD(n_components=self.n_topics, random_state=42)
        self.model.fit(X)

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
    
    def __init__(self, n_topics=None, top_n_words=None):
        self.n_topics = n_topics or int(os.getenv("N_TOPICS", 5))
        self.top_n_words = top_n_words or int(os.getenv("TOP_N_WORDS", 10))
        self.model = None
        self.dictionary = None
        self.corpus = None

    def fit(self, tokenized_docs):
        self.dictionary = Dictionary(tokenized_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]

        # automated model selection
        best_model, best_num_topics, _ = find_optimal_lda_model(
            tokenized_docs,
            min_topics=2,
            max_topics=15,
            step=1,
            passes=10,
            random_state=42
        )

        # use the best model found
        self.model = best_model
        self.n_topics = best_num_topics

        print(f"\nBestes Modell übernommen: {best_num_topics} Topics mit maximaler Kohärenz.")
        
        # visualize the LDA model
        try:
            visualize_lda_model(self.model, self.corpus, self.dictionary)
        except Exception as e:
            print(f"LDA-Visualisierung konnte nicht erstellt werden: {e}")

    def get_topics(self):
        topics = []
        for idx in range(self.n_topics):
            top_words = self.model.show_topic(idx, topn=self.top_n_words)
            topics.append({
                "Topic": idx + 1,
                "Title": top_words[0][0] if top_words else "",
                "Words": top_words
            })
        return topics

    def print_topics(self, topic_idx=None):
        topics = self.get_topics()
        for t in topics:
            if topic_idx is not None and t['Topic'] != topic_idx:
                continue
            word_str = ", ".join([f"{w} ({v:.4f})" for w, v in t['Words']])
            print(f"Topic {t['Topic']} ({t['Title']}): {word_str}")

    def compute_coherence(self, tokenized_docs, coherence="c_v"):
        if self.model is None:
            raise ValueError("Modell nicht trainiert. Rufe zuerst .fit() auf.")

        topics = [[w for w, _ in t['Words']] for t in self.get_topics()]

        if coherence == "u_mass":
            corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
            cm = CoherenceModel(
                topics=topics,
                corpus=corpus,
                dictionary=self.dictionary,
                coherence=coherence
            )
        else:
            cm = CoherenceModel(
                topics=topics,
                texts=tokenized_docs,
                dictionary=self.dictionary,
                coherence=coherence
            )

        score = cm.get_coherence()
        print(f"Coherence Score ({coherence}): {score:.4f}")
        return score

def semantic_analysis(processed_texts):
    docs_as_strings = [" ".join(doc) for doc in processed_texts]

    # LSA
    vectorizer = CountVectorizer(max_features=int(os.getenv("MAX_FEATURES", 100)))
    X = vectorizer.fit_transform(docs_as_strings)
    feature_names = vectorizer.get_feature_names_out()
    print("\n--- LSA: semantic structure ---")
    lsa = LSAAnalyzer(
        n_topics=int(os.getenv("N_LSA_TOPICS", 5)),
        max_features=int(os.getenv("MAX_FEATURES", 100)),
        top_n_words=int(os.getenv("N_WORDS_PER_TOPIC", 8))
    )
    lsa.fit(X, feature_names)
    lsa.print_topics()

    # LDA
    print("\n--- LDA: probability-based topics ---")
    lda = LDAAnalyzer(
        n_topics=int(os.getenv("N_LDA_TOPICS", 5)),
        top_n_words=int(os.getenv("N_WORDS_PER_TOPIC", 8))
    )
    lda.fit(processed_texts)
    lda.print_topics()

    # show latest coherence scores
    final_cv = lda.compute_coherence(processed_texts, coherence="c_v")
    final_umass = lda.compute_coherence(processed_texts, coherence="u_mass")
    print(f"\nFinale Coherence-Werte:\n  c_v: {final_cv:.4f}\n  u_mass: {final_umass:.4f}")
