import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from nlu.contracts import IntentResult

class IntentBaselineClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            min_df=1
        )
        base_clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=2.0)
        self.clf = CalibratedClassifierCV(base_clf, cv=3)
        self.is_trained = False
        self.classes_ = []

    def train(self, texts: list[str], labels: list[str]):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        self.is_trained = True

    def predict(self, text: str) -> IntentResult:
        if not self.is_trained:
            return IntentResult(name="check_balance", confidence=0.5, decision="fallback")

        X = self.vectorizer.transform([text])
        probas = self.clf.predict_proba(X)[0]
        best_idx = np.argmax(probas)
        top_intent = self.classes_[best_idx]
        conf = float(probas[best_idx])

        sorted_indices = np.argsort(probas)[::-1]
        top_k = [(self.classes_[i], float(probas[i])) for i in sorted_indices[:3]]

        if conf >= 0.70:
            decision = "accept"
        elif conf >= 0.40:
            decision = "clarify"
        else:
            decision = "fallback"

        return IntentResult(
            name=top_intent,
            confidence=conf,
            top_k=top_k,
            decision=decision
        )

    def save(self, path: str = "reports/baseline_intent.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.vectorizer, self.clf, self.classes_, self.is_trained), f)

    def load(self, path: str = "reports/baseline_intent.pkl") -> bool:
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.vectorizer, self.clf, self.classes_, self.is_trained = pickle.load(f)
            return True
        return False