from math import pow, sqrt, fabs
from collections import Counter


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

    def compare(self, other):
        if self.distance > other.distance:
            return 1
        elif self.distance < other.distance:
            return -1
        else:
            return 0

    def display(self):
        print(f"Target: {self.target}, Distance: {self.distance}")


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
    def __init__(self, k):
        self.list_data = []
        self.k = k

    def display_data(self):
        for data in self.list_data:
            data.display()

    def fit(self, data):
        self.list_data = data

    def count_distance(self, a, b, metric, p=3):
        if metric == "euclidean":
            return Metrik.euclidean_distance(a, b)
        elif metric == "manhattan":
            return Metrik.manhattan_distance(a, b)
        elif metric == "minkowski":
            return Metrik.minkowski_distance(a, b, p)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def calculate_distance(self, metric, test_features, p=3):
        for data in self.list_data:
            dist = self.count_distance(data.get_features(), test_features, metric, p)
            data.set_distance(dist)

    def predict(self, test_features, metric, p=3):
        self.calculate_distance(metric, test_features, p)
        self.list_data.sort(key=lambda data: data.get_distance())

        frequency = Counter(data.get_target() for data in self.list_data[:self.k])
        return frequency.most_common(1)[0][0]

    def predict_proba(self, test_features, metric, p=3):
        self.calculate_distance(metric, test_features, p)
        self.list_data.sort(key=lambda data: data.get_distance())

        frequency = Counter(data.get_target() for data in self.list_data[:self.k])
        total = sum(frequency.values())
        probabilities = {target: count / total for target, count in frequency.items()}
        return probabilities
