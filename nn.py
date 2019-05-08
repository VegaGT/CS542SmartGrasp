from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.layers import *
import numpy as np
from sklearn.model_selection import train_test_split
import h5py


def readFile(file):
    dataset = np.loadtxt('./shadow_robot_dataset.csv',
                         skiprows=1, usecols=range(1, 30),
                         delimiter=',')

    with open(file, 'r') as f:
        header = f.readline()
    header = header.strip('\n').split(',')
    header = [i.strip(' ') for i in header]
    saved_cols = []
    for index, col in enumerate(header[1:]):
        if ('vel' in col) or ('eff' in col):
            saved_cols.append(index)

    new_X = []

    for x in dataset:
        new_X.append([x[i] for i in saved_cols])

    X = np.array(new_X)
    Y = dataset[:, 0]

    return X, Y


class GraspPred:

    def __init__(self, file):
        self.X, self.Y = readFile(file)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.20)
        self.model = None
        self.score = None

        GOOD_GRASP_THRESHOLD = 100
        itemindex = np.where(self.Y_test > 1.05 * GOOD_GRASP_THRESHOLD)
        best_grasps = self.X_test[itemindex[0]]
        itemindex = np.where(self.Y_test <= 0.95 * GOOD_GRASP_THRESHOLD)
        bad_grasps = self.X_test[itemindex[0]]

        # transform Y into labels (0 or 1)
        self.Y_train = np.array([int(i > GOOD_GRASP_THRESHOLD) for i in self.Y_train])
        self.Y_train = np.reshape(self.Y_train, (self.Y_train.shape[0],))
        self.Y_test = np.array([int(i > GOOD_GRASP_THRESHOLD) for i in self.Y_test])
        self.Y_test = np.reshape(self.Y_test, (self.Y_test.shape[0],))

    def build_model(self):
        # Create model
        self.model = Sequential()
        self.model.add(Dense(20 * len(self.X[0]), use_bias=True, input_dim=len(self.X[0]), activation='relu'))
        # model.add(Dropout(0.5))

        # model.add(Dense(20*len(X[0]), activation='relu'))
        # model.add(Dropout(0.5))

        self.model.add(Dense(20 * len(self.X[0]), activation='relu'))
        # model.add(Dropout(0.5))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train, validation_split=0.10, epochs=20, batch_size=500000)

    def save_model(self):
        model_json = self.model.to_json()
        with open('model_ep20_nodrop.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('model_ep20_nodrop.h5')

    def load_model(self, json_fn, h5_fn):
        json_file = open(json_fn, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(h5_fn)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return self.model

    def predict(self, x):
        y_pred = self.model.predict_classes(np.array([x]))
        return y_pred

    def calculate_score(self):
        scores = self.model.evaluate(self.X_test, self.Y_test)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores


if __name__ == "__main__":
    gp = GraspPred('./shadow_robot_dataset.csv')

    # for training new model
    # gp.build_model()
    # gp.train_model()
    # gp.save_model()

    # directly load existed model
    model = gp.load_model('model_ep20_nodrop.json', 'model_ep20_nodrop.h5')

    # get score of model
    score = gp.calculate_score()

    # choose x for predict
    x = gp.X_test[2, :]
    y = gp.predict(x)
    print(y)
