import csv
import numpy as np
import os


class Data(object):
    def __init__(self, data_path):
        self._features = np.array([])
        self._labels = np.array([])
        self._test_features = np.array([])
        self._test_labels = np.array([])

        self._load_dataset(data_path)

        self._num_examples = len(self._labels)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def test_features(self):
        return self._test_features

    @property
    def test_labels(self):
        return self._test_labels

    def _load_dataset(self, data_path):
        feature_path = os.path.join(data_path, "train", "features.csv")
        with open(feature_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = [int(i) for i in row]
                if len(self._features) == 0:
                    self._features = np.array([row])
                    continue
                self._features = np.concatenate((self._features, np.array([row])), axis=0)

        label_path = os.path.join(data_path, "train", "labels.csv")
        with open(label_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = [float(i) for i in row]
                if len(self._labels) == 0:
                    self._labels = np.array([row])
                    continue
                self._labels = np.concatenate((self._labels, np.array([row])), axis=0)

        feature_path = os.path.join(data_path, "test", "features.csv")
        with open(feature_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = [int(i) for i in row]
                if len(self._test_features) == 0:
                    self._test_features = np.array([row])
                    continue
                self._test_features = np.concatenate((self._test_features, np.array([row])), axis=0)

        label_path = os.path.join(data_path, "test", "labels.csv")
        with open(label_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = [float(i) for i in row]
                if len(self._test_labels) == 0:
                    self._test_labels = np.array([row])
                    continue
                self._test_labels = np.concatenate((self._test_labels, np.array([row])), axis=0)

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (features_rest_part, features_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]


# data_path = "./dataset"
# dataset = Data(data_path)
# print(dataset.features[0])
