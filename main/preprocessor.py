import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy
import time

class Preprocessor:
    def __init__(self, spacy_model="en_core_web_sm", language="english"):
        # get stopwords
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Error downloading stopwords: {e}")

        self.stop_words = set(stopwords.words(language))
        
        # load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            print(f"Error loading spaCy model '{spacy_model}': {e}")
            raise
    
    def load_data(self, filepath, text_column):
        
        print(f"Loading data from {filepath}...")
        # loading dataset from a CSV file and extracting the specified text column
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Number of rows: {len(df)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filepath} not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File {filepath} is empty.")
        except pd.errors.ParserError:
            raise ValueError(f"File {filepath} could not be parsed.")
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")
        
        try:
            texts = df[text_column].dropna().tolist()
            if not texts:
                raise ValueError(f"No texts found in column '{text_column}'.")
            print(f"Extracted {len(texts)} texts from column '{text_column}'.")
            return texts
        except Exception as e:
            raise ValueError(f"Error extracting column '{text_column}': {e}")
    
    def clean_text(self, text):
        # cleans the input text by removing HTML tags, special characters, and extra spaces
        try:
            text = re.sub(r'<.*?>', '', text)  # remove HTML tags
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
            text = re.sub(r'\s+', ' ', text)  # remove extra spaces
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error cleaning text: {e}")
    
    def tokenize_lower(self, text):
        # Tokenizes the input text and converts it to lowercase
        try:
            text_lowercase = text.lower().split()
            return text_lowercase
        except Exception as e:
            raise ValueError(f"Error tokenizing text: {e}")
    
    def remove_stopwords(self, tokens):
        # removes stopwords from the token list
        try:
            removed_stopwords = [word for word in tokens if word not in self.stop_words]
            return removed_stopwords
        except Exception as e:
            raise ValueError(f"Error removing stopwords: {e}")
    
    def lemmatize(self, tokens):
        # lemmatizes the tokens using spaCy
        try:
            doc = self.nlp(" ".join(tokens))
            lemmatized_tokens = [token.lemma_ for token in doc]
            return lemmatized_tokens
        except Exception as e:
            raise ValueError(f"Error lemmatizing tokens: {e}")
    
    def preprocess_text(self, text):
        # Applies the entire preprocessing pipeline to a single text input
        text = self.clean_text(text)
        tokens = self.tokenize_lower(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    
    def preprocess_corpus(self, texts):
        # Applies the preprocessing pipeline to a list of texts
        results = []
        last_log = time.time()
        total = len(texts)

        for i, text in enumerate(texts, start=1):
            results.append(self.preprocess_text(text))

            # Log progress every 5 seconds
            if time.time() - last_log >= 10:
                print(f"Processed {i}/{total} texts...")
                last_log = time.time()

        return results
