import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

class ScratchNaiveBayes:
    def __init__(self):
        self.class_probs = None
        self.feature_probs = None
        self.classes = None
        self.means = None
        self.stds = None

    def fit(self, X, y):
        """ Fit the model using the training data """
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X must match length of y")
        self.classes = np.unique(y)
        class_counts = np.array([np.sum(y == c) for c in self.classes])
        self.class_probs = class_counts / len(y)

        """ Fit the Gaussian Naive Bayes model """
        means = []
        stds = []
        for c in self.classes:
            X_class = X[y == c]
            means.append(X_class.mean(axis=0))
            stds.append(X_class.std(axis=0))
        self.means = np.array(means)
        self.stds = np.array(stds)
        
        return self

    def predict(self, X):
        """ Predict the class labels for the input data """
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """ Predict the class label for a single data point """
        log_probs = np.log(self.class_probs)
        for i, c in enumerate(self.classes):
            mean = self.means[i]
            std = self.stds[i]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * std ** 2) + ((x - mean) ** 2) / (std ** 2))
            log_probs[i] += log_likelihood
        
        return self.classes[np.argmax(log_probs)]

    def predict_proba(self, X):
        """ Predict the class probabilities for the input data """
        probas = [self._predict_proba_single(x) for x in X]
        return np.array(probas)

    def _predict_proba_single(self, x):
        """ Predict the class probabilities for a single data point """
        log_probs = np.log(self.class_probs)
        likelihoods = np.zeros(len(self.classes))

        for i, c in enumerate(self.classes):
            mean = self.means[i]
            std = self.stds[i]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * std ** 2) + ((x - mean) ** 2) / (std ** 2))
            log_probs[i] += log_likelihood
            likelihoods[i] = np.exp(log_probs[i])  # Convert log probs to normal probabilities
        
        total_likelihood = np.sum(likelihoods)
        return likelihoods / total_likelihood  # Normalize to get probabilities

    def score(self, X, y, verbose=True):
        """ Evaluate the model on the test data """
        accuracy = np.mean(self.predict(X) == y)

        return accuracy

    # def _confusion_matrix(self, y_true, y_pred):
    #     """ Generate a confusion matrix """
    #     confusion = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
    #     return confusion
    
    # def _classification_report(self, y_true, y_pred):
    #     """ Generate a classification report """
    #     report = {}
    #     classes = np.unique(y_true)
    #     for c in classes:
    #         tp = np.sum((y_pred == c) & (y_true == c))
    #         fp = np.sum((y_pred == c) & (y_true != c))
    #         fn = np.sum((y_pred != c) & (y_true == c))
    #         tn = np.sum((y_pred != c) & (y_true != c))
            
    #         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #         support = np.sum(y_true == c)
            
    #         report[c] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support}
        
    #     return pd.DataFrame(report).T

    # def compute_classification_report(self, y_true, y_pred):
    #     # Get unique classes from the ground truth
    #     classes = np.unique(y_true)
    #     target_names = [str(cls) for cls in classes]  # Convert class labels to strings if necessary
    #     metrics = []
        
    #     for cls in classes:
    #         # True Positives, False Positives, False Negatives
    #         TP = np.sum((y_pred == cls) & (y_true == cls))
    #         FP = np.sum((y_pred == cls) & (y_true != cls))
    #         FN = np.sum((y_pred != cls) & (y_true == cls))
    #         support = np.sum(y_true == cls)
            
    #         # Precision, Recall, F1-Score
    #         precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    #         recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    #         f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
    #         # Store metrics
    #         metrics.append([precision, recall, f1_score, support])
        
    #     # Convert to DataFrame for better display
    #     metrics_df = pd.DataFrame(
    #         metrics,
    #         columns=["Precision", "Recall", "F1-Score", "Support"],
    #         index=target_names
    #     )
        
    #     # Calculate averages
    #     accuracy = np.sum(y_true == y_pred) / len(y_true)
    #     macro_avg = metrics_df[["Precision", "Recall", "F1-Score"]].mean().tolist()
    #     weighted_avg = (
    #         (metrics_df[["Precision", "Recall", "F1-Score"]].T * metrics_df["Support"]).sum(axis=1) /
    #         metrics_df["Support"].sum()
    #     )

    #     metrics_df.loc[" "] = [" ", " ", " ", " "]
        
    #     # Add averages to DataFrame
    #     metrics_df.loc["Accuracy"] = [" ", " ", accuracy, len(y_true)]
    #     metrics_df.loc["Macro Avg"] = macro_avg + [len(y_true)]
    #     metrics_df.loc["Weighted Avg"] = weighted_avg.tolist() + [len(y_true)]
        
    #     return metrics_df

# Example Usage:

# Simulated Example (using random data for simplicity)
start = time.time()
np.random.seed(42)
X = np.random.randn(150000, 50)  # 100 samples, 4 features
y = np.random.choice([0, 1], size=150000)  # Binary target (0 or 1)

# Train-test split
train_size = 0.7
split = int(len(X) * train_size)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Custom Naive Bayes
print("Custom Naive Bayes:")
custom_model = ScratchNaiveBayes()
custom_model.fit(X_train, y_train)
custom_y_pred = custom_model.predict(X_test)
custom_accuracy = custom_model.score(X_test, y_test)

print(f'Accuracy: {custom_accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, custom_y_pred))

# Scikit-learn Naive Bayes
print("\nScikit-learn Naive Bayes:")
sklearn_model = GaussianNB()
sklearn_model.fit(X_train, y_train)
sklearn_y_pred = sklearn_model.predict(X_test)
sklearn_accuracy = sklearn_model.score(X_test, y_test)

print(f"Accuracy: {sklearn_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, sklearn_y_pred))
end = time.time()
print(f"Time Taken: {end - start:.6f} seconds")
