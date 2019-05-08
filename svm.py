import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


class GraspPred:

    def __init__(self, feature_file, label_file):
        # Load .mat file
        features = loadmat(feature_file)
        labels = loadmat(label_file)

        X = features['features']
        y = np.ravel(labels['labels'])

        # Split data set into training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)

        self.y_pred = None  # Prediction label
        self.model = None  # SVM model
        self.score = None  # Returns the mean accuracy on the given test data and labels
        self.filename = 'model.sav'  # The filename of model

    def train_model(self):
        self.model = SVC(gamma='auto')
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def save_model(self):
        pickle.dump(self.model, open(self.filename, 'wb'))

    def predict(self):
        loaded_model = pickle.load(open(self.filename, 'rb'))
        self.y_pred = loaded_model.predict(self.X_test)
        return self.y_pred

    def calculate_score(self):
        loaded_model = pickle.load(open(self.filename, 'rb'))
        self.score = loaded_model.score(self.X_test, self.y_test)
        return self.score


if __name__ == "__main__":
    gp = GraspPred('bigger_features.mat', 'bigger_labels.mat')
    gp.train_model()
    gp.save_model()
    gp.predict()
    print(gp.calculate_score())


