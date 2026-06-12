from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np


class MatchPredictor:

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
            eval_metric="mlogloss"
        )

    def train(self, X, y, df=None):
        """
        Chronological split: 70% train, 15% validation (early stopping), 15% test.
        """
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

        print(f"Training set size: {len(X_train)} (70%)")
        print(f"Validation set size: {len(X_val)} (15%)")
        print(f"Test set size: {len(X_test)} (15%)")

        # Balance classes (home/draw/away rarely uniform)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\nBest iteration: {self.model.best_iteration}")
        print(f"Accuracy: {accuracy:.4f}")

        self.X_test = X_test
        self.y_test = y_test
        self.probabilities = probabilities
        self.predictions = predictions

        return accuracy, probabilities

    def predict_match(self, home_features, away_features):
        combined_features = home_features + away_features
        prob = self.model.predict_proba([combined_features])
        return prob[0]