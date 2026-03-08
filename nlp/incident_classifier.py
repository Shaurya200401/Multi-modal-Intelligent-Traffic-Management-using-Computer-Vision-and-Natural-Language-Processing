# nlp/incident_classifier.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class IncidentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.model = LogisticRegression(max_iter=1000)

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def save(self, path="nlp/incident_model.pkl"):
        joblib.dump((self.vectorizer, self.model), path)

    def load(self, path="nlp/incident_model.pkl"):
        self.vectorizer, self.model = joblib.load(path)
