#include <bits/stdc++.h>

using namespace std;

template <typename T>
class Data {
    private:
        T target;
        vector<double> features;
        double distance;

    public:
        Data(T target, vector<double> features, double distance = 0) 
            : target(target), features(features), distance(distance) {}

        void set_distance(double dist) {
            distance = dist;
        }

        double get_distance() const {
            return distance;
        }

        T get_target() const {
            return target;
        }

        vector<double> get_features() const {
            return features;
        }

        int compare(const Data<T>& data) const {
            if (distance > data.distance) {
                return 1;
            } else if (distance < data.distance) {
                return -1;
            } else {
                return 0;
            }
        }

        void display() const {
            cout << "Target: " << target << ", Distance: " << distance << endl;
        }
};

class Metrik {
    public:
        static double euclideanDistance(const vector<double>& x, const vector<double>& y) {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); ++i) {
                sum += pow(x[i] - y[i], 2);
            }
            return sqrt(sum);
        }

        static double manhattanDistance(const vector<double>& x, const vector<double>& y) {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); ++i) {
                sum += fabs(x[i] - y[i]);
            }
            return sum;
        }

        static double minkowskiDistance(const vector<double>& x, const vector<double>& y, double p) {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); ++i) {
                sum += pow(fabs(x[i] - y[i]), p);
            }
            return pow(sum, 1.0 / p);
        }
};

template <typename T>
class KNN {
    private:
        vector<Data<T>> list_data;
        int k;

    public:
        KNN(int k) : k(k) {}

        void display_data(){
            for (Data<T>& data : list_data) {
                data.display();
            }
        }

        void fit(const vector<Data<T>>& data) {
            list_data = data;
        }

        double count_distance(const vector<double>& a, const vector<double>& b, const string& metric, double p) {
            if (metric == "euclidean") {
                return Metrik::euclideanDistance(a, b);
            } else if (metric == "manhattan") {
                return Metrik::manhattanDistance(a, b);
            } else if (metric == "minkowski") {
                return Metrik::minkowskiDistance(a, b, p);
            } else {
                throw invalid_argument("Unknown metric: " + metric);
            }
        }

        void calculate_distance(const string& metric, const vector<double>& test_features, double p) {
            for (Data<T>& data : list_data) {
                double dist = count_distance(data.get_features(), test_features, metric, p);
                data.set_distance(dist);
            }
        }

        T predict(const vector<double>& test_features, const string& metric, double p = 3) {
            calculate_distance(metric, test_features, p);

            sort(list_data.begin(), list_data.end(), [](const Data<T>& a, const Data<T>& b) {
                return a.compare(b) == -1;
            });

            map<T, int> frequency;
            for (int i = 0; i < k && i < list_data.size(); i++) {
                frequency[list_data[i].get_target()]++;
            }

            T predicted_class;
            int max_count = 0;
            for (const auto& pair : frequency) {
                if (pair.second > max_count) {
                    max_count = pair.second;
                    predicted_class = pair.first;
                }
            }

            return predicted_class;
        }

        map<T, double> predict_proba(const vector<double>& test_features, const string& metric, double p = 3) {
            calculate_distance(metric, test_features, p);

            sort(list_data.begin(), list_data.end(), [](const Data<T>& a, const Data<T>& b) {
                return a.compare(b) == -1;
            });

            map<T, int> frequency;
            for (int i = 0; i < k && i < list_data.size(); i++) {
                frequency[list_data[i].get_target()]++;
            }

            map<T, double> probabilities;
            for (const auto& pair : frequency) {
                probabilities[pair.first] = (double)pair.second / k;
            }

            return probabilities;
        }
};

// int main() {
//     vector<Data<int>> training_data = {
//         Data<int>(0, {1.0, 2.0}),
//         Data<int>(1, {2.0, 3.0}),
//         Data<int>(0, {3.0, 4.0}),
//         Data<int>(1, {2.0, 3.0})
//     };

//     KNN<int> knn(3);
//     knn.fit(training_data);

//     vector<double> test_features = {2.0, 3.0};
//     int predicted_class = knn.predict(test_features, "euclidean");

//     cout << "Predicted class: " << predicted_class << endl;

//     auto probabilities = knn.predict_proba(test_features, "euclidean");
//     cout << "Probabilities:" << endl;
//     for (const auto& pair : probabilities) {
//         cout << "Class " << pair.first << ": " << pair.second << endl;
//     }

//     knn.display_data();

//     return 0;
// }

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <k> <preprocessing_result> <file_test>" << endl;
        return 1;
    }

    int k(stoi(argv[1]));
    ifstream inputFile1(argv[2]);
    ifstream inputFile2(argv[3]);

    if (!inputFile1) {
        cerr << "Could not open the file: " << argv[2] << endl;
        return 1;
    }

    if (!inputFile2) {
        cerr << "Could not open the file: " << argv[3] << endl;
        return 1;
    }

    vector<Data<int>> train_data;
    vector<vector<double>> test_data;
    string line;

    // string header;
    // getline(inputFile1, header); // Skip header
    // Membaca train data
    while (getline(inputFile1, line)) {

        stringstream ss(line);
        vector<double> features;
        int target;
        double value;

        // Memproses fitur
        while (ss >> value) {
            features.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        // Target adalah elemen terakhir
        if (!features.empty()) {
            target = static_cast<int>(features.back());
            features.pop_back();
            train_data.emplace_back(target, features);
        }
    }

    // Membaca test data
    // getline(inputFile2, header); // Skip header
    while (getline(inputFile2, line)) {
        
        stringstream ss(line);
        vector<double> features;
        double value;

        // Memproses fitur
        while (ss >> value) {
            features.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        test_data.push_back(features);
    }

    inputFile1.close();
    inputFile2.close();

    KNN<int> knn(k);

    knn.fit(train_data);

    cout << "Predictions:" << endl;
    for (const auto& test_features : test_data) {
        for (auto &&i : test_features)
        {
            cout << i << " ";
        }
        cout << endl;
        
        try {
            int predicted_class = knn.predict(test_features, "euclidean");
            cout << "Predicted class: " << predicted_class << endl;
        } catch (const invalid_argument& e) {
            cerr << e.what() << endl;
        }
    }


    return 0;
}
