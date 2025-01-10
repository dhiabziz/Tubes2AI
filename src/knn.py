from math import pow, sqrt, fabs
from collections import Counter
import pandas as pd


class Data:
    def __init__(self, target, features, distance=0):
        self.target = target
        self.features = features
        self.distance = distance

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def get_target(self):
        return self.target

    def get_features(self):
        return self.features


class Metrik:
    @staticmethod
    def euclidean_distance(x, y):
        return sqrt(sum(pow(xi - yi, 2) for xi, yi in zip(x, y)))

    @staticmethod
    def manhattan_distance(x, y):
        return sum(fabs(xi - yi) for xi, yi in zip(x, y))

    @staticmethod
    def minkowski_distance(x, y, p):
        return pow(sum(pow(fabs(xi - yi), p) for xi, yi in zip(x, y)), 1 / p)


class KNN:
    def __init__(self, n_neighbors=5, metric="euclidean", p=3):
        self.list_data = []
        self.k = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X_train, y_train):
        """
        Fit the model with training data.
        :param X_train: DataFrame containing features.
        :param y_train: DataFrame or Series containing targets.
        """
        if not isinstance(y_train, pd.DataFrame):
            y_train = pd.DataFrame(y_train, columns=['attack_cat'])

        self.list_data = [
            Data(target=y, features=x.tolist()) for x, y in zip(X_train.values, y_train.values)
        ]

    def count_distance(self, a, b):
        if self.metric == "euclidean":
            return Metrik.euclidean_distance(a, b)
        elif self.metric == "manhattan":
            return Metrik.manhattan_distance(a, b)
        elif self.metric == "minkowski":
            return Metrik.minkowski_distance(a, b, self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def calculate_distance(self, test_features):
        for data in self.list_data:
            dist = self.count_distance(data.get_features(), test_features)
            data.set_distance(dist)

    def predict(self, X_test):
        """
        Predict target values for given test data.
        :param X_test: DataFrame containing features.
        :return: List of predicted target values.
        """
        predictions = []
        for test_row in X_test.values:
            self.calculate_distance(test_row)
            self.list_data.sort(key=lambda data: data.get_distance())
            frequency = Counter(data.get_target()[0] for data in self.list_data[:self.k])
            predictions.append(frequency.most_common(1)[0][0])
        return predictions

    def predict_proba(self, X_test):
        """
        Predict probabilities for given test data.
        :param X_test: DataFrame containing features.
        :return: List of dictionaries with probabilities for each class.
        """
        probabilities_list = []
        for test_row in X_test.values:
            self.calculate_distance(test_row)
            self.list_data.sort(key=lambda data: data.get_distance())

            frequency = Counter(data.get_target() for data in self.list_data[:self.k])
            total = sum(frequency.values())
            probabilities = {target: count / total for target, count in frequency.items()}
            probabilities_list.append(probabilities)
        return probabilities_list
    
# import pandas as pd

# # Contoh data
# data = {
#     "feature1": [1, 2, 3, 4, 5],
#     "feature2": [2, 3, 4, 5, 6],
#     "target": ["A", "A", "B", "B", "A"]
# }

# X_train = pd.DataFrame({"feature1": data["feature1"], "feature2": data["feature2"]})
# y_train = pd.Series(data["target"])

# X_test = pd.DataFrame({"feature1": [1.5, 3.5], "feature2": [2.5, 4.5]})

# # Inisialisasi dan pelatihan model
# knn = KNN(n_neighbors=3)
# knn.fit(X_train, y_train)

# # Prediksi
# predictions = knn.predict(X_test)
# print("Predictions:", predictions)

# # Probabilitas prediksi
# probabilities = knn.predict_proba(X_test)
# print("Probabilities:", probabilities)
