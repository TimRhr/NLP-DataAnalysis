from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class BoWVectorizer:
    def __init__(self, max_features=100):
        # initialize the CountVectorizer with a maximum number of features
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.features = None  # list to hold feature names

    def fit_transform(self, texts):
        # Fits the vectorizer to the texts and transforms them into BoW vectors
        X = self.vectorizer.fit_transform(texts)
        self.features = self.vectorizer.get_feature_names_out()
        return X

    def transform(self, texts):
        # Transforms new texts into BoW vectors using the fitted vectorizer
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        # Returns the feature names (vocabulary) learned by the vectorizer
        return self.features


class TFIDFVectorizer:
    def __init__(self, max_features=100):
        # initialize the TfidfVectorizer with a maximum number of features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.features = None

    def fit_transform(self, texts):
        # Fits the vectorizer to the texts and transforms them into TF-IDF vectors
        X = self.vectorizer.fit_transform(texts)
        self.features = self.vectorizer.get_feature_names_out()
        return X

    def transform(self, texts):
        # Transforms new texts into TF-IDF vectors using the fitted vectorizer
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        # Returns the feature names (vocabulary) learned by the vectorizer
        return self.features