import numpy as np
from collections import Counter

def fit_naive_bayes(X, y):
    """
    Fits the Naive Bayes model to the training data.

    Parameters:
    X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): The target vector of shape (n_samples,).

    Returns:
    tuple: A tuple containing class_priors, feature_likelihoods, and classes.
    """
    classes, class_counts = np.unique(y, return_counts=True)
    class_priors = class_counts / len(y)

    # Calculate feature likelihoods P(feature | class)
    feature_likelihoods = {}
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        cls_features = X[cls_indices]
        feature_probs = []
        for feature_col in cls_features.T:
            values, counts = np.unique(feature_col, return_counts=True)
            probs = {val: count / len(feature_col) for val, count in zip(values, counts)}
            feature_probs.append(probs)
        feature_likelihoods[cls] = feature_probs

    return class_priors, feature_likelihoods, classes

def predict_proba_naive_bayes(X, class_priors, feature_likelihoods, classes):
    """
    Predicts the class probabilities for the given input data.

    Parameters:
    X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
    class_priors (numpy.ndarray): Prior probabilities of each class.
    feature_likelihoods (dict): Likelihoods of features given a class.
    classes (numpy.ndarray): Unique classes.

    Returns:
    numpy.ndarray: Predicted probabilities of shape (n_samples, n_classes).
    """
    probs = []
    for sample in X:
        class_probs = []
        for cls_idx, cls in enumerate(classes):
            prob = class_priors[cls_idx]
            for feature_idx, feature_value in enumerate(sample):
                feature_probs = feature_likelihoods[cls][feature_idx]
                prob *= feature_probs.get(feature_value, 1e-6)  # Avoid zero probability
            class_probs.append(prob)
        probs.append(class_probs)
    return np.array(probs)

def predict_naive_bayes(X, class_priors, feature_likelihoods, classes):
    """
    Predicts the class labels for the given input data.

    Parameters:
    X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
    class_priors (numpy.ndarray): Prior probabilities of each class.
    feature_likelihoods (dict): Likelihoods of features given a class.
    classes (numpy.ndarray): Unique classes.

    Returns:
    numpy.ndarray: Predicted class labels of shape (n_samples,).
    """
    probs = predict_proba_naive_bayes(X, class_priors, feature_likelihoods, classes)
    return classes[np.argmax(probs, axis=1)]

# Unit Testing Example
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([1, 1, 0, 0])

# Fit the model
class_priors, feature_likelihoods, classes = fit_naive_bayes(X, y)

# Predictions
predictions = predict_naive_bayes(X, class_priors, feature_likelihoods, classes)
print("Predictions:", predictions)

# Predicted probabilities
predicted_probs = predict_proba_naive_bayes(X, class_priors, feature_likelihoods, classes)
print("Predicted Probabilities:\n", predicted_probs)
