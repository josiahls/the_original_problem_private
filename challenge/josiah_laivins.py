## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagiarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
from util import filters
from util.hough_accumulator import HoughAccumulator
from util import image


# It's kk to import whatever you want from the local util module if you would like:
# from util.X import ...

def classify(im: np.array([0])):
    """
    Based on the data given there are a few observations.


    We need something simple to classify the images will manually input parameters since we will be unable to pickle.


    :param im: (numpy 3 dimensional array RGB) unsigned 8-bit color image
    :return: One of three strings: 'brick', 'ball', or 'cylinder'

    """
    # Let's guess randomly! Maybe we'll get lucky.
    labels = ['brick', 'ball', 'cylinder']

    gray_scale_image = image.convert_to_grayscale(im)

    random_integer = np.random.randint(low=0, high=3)

    return labels[random_integer]


class K_Means(object):
    # noinspection PyDefaultArgument
    def __init__(self, k=3, labels=list(), tolerance=0.001, max_iterations=1000):
        """
        This is planned to take in a histogram,
        and give the classification of it.

        :param k: Number of groups (3)
        :param tolerance:
        :param max_iterations:
        """
        self.k = k
        if len(labels) != self.k:
            raise Exception('You cannot have number of labels differing from '
                            'the number of desired groups!')
        self.labels = labels
        self.channels = 1
        self.tol = tolerance
        self.max_iter = max_iterations
        # Generate centroids will random point locations
        self.centroids = {}  # {label: (np.random.random_integers(0, 100)) for label in self.labels}
        self.classifications = {}

    def fit(self, data: list):
        """
        fit will take in a data variable as such:

        data = [{'classification': np.array}]

        :param data:
        :return:
        """
        for channel in range(data[0][1].shape[2]):

            # Get the starting centroids for this respective channel
            self.centroids[channel] = {label: [data[[f[0] for f in data].index(label)][0]] +
                                              [data[[f[0] for f in data].index(label)][1][:, :, channel]]
                                       for label in self.labels}

            # Get those centroids as histograms
            self.centroids[channel] = {label: np.histogram(self.centroids[channel][label][1], bins=255)[0]
                                       for label in self.centroids[channel]}

            for i in range(self.max_iter):
                print(f"Inter: {i} for channel: {channel}")
                self.classifications = {}

                # Init all recorded classifications
                for j in range(self.k):
                    self.classifications[self.labels[j]] = []

                for feature_set in data:
                    # feature set is a single histogram
                    # distances = [np.linalg.norm(feature_set[1] - self.centroids[centroid][1]) for centroid in self.centroids]
                    # classification = self.labels[distances.index(min(distances))]
                    feature_hist = np.histogram(feature_set[1][:, :, channel], bins=255)[0]
                    self.classifications[feature_set[0]].append(feature_hist)

                prev_centroids = {centroid: np.copy(self.centroids[channel][centroid])
                                  for centroid in self.centroids[channel]}

                for classification in self.classifications:
                    # This is where the centroids get changed. Originally, they would separate based on
                    # center of groups, however I need to differentiate this more.
                    all_classifications = []
                    for c in self.classifications:
                        all_classifications += self.classifications[c]

                    # Right now: category average - all average = noiseless full diff
                    self.centroids[channel][classification] = np.average(self.classifications[classification], axis=0)

                    # Set all negative values to 0
                    self.centroids[channel][classification][self.centroids[channel][classification] < 0] = 0

                optimized = True
                for c in self.centroids[channel]:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[channel][c]
                    if np.sum((current_centroid[1] - original_centroid[1]) / (original_centroid[1] + 0.001) *
                              100.0) > self.tol:
                        print(np.sum((current_centroid[1] - original_centroid[1]) / (original_centroid[1] + 0.001)
                                     * 100.0))
                        optimized = False
                if optimized:
                    break

    def predict(self, data: np.array):
        """
        data is a 3d shape:
        - for every RGB channel,
        - The search grid is 3 x categories

        Example - matrix of distances:
        cat:  Ball Brick Cylinder
            R  10    2      6
            G  8     7      8
            B  7     9      9

        :param data:
        :return:
        """
        distances = [i for i in range(data.shape[2])]
        for channel in range(data.shape[2]):
            distances[channel] = [np.linalg.norm(np.histogram(data[:, :, channel], bins=255)[0] -
                                                 self.centroids[channel][centroid][1])
                                  for centroid in self.centroids[channel]]

        distances = np.average(distances, axis=0)
        distances = [int(_) for _ in distances]
        classification = self.labels[distances.index(int(min(distances)))]
        return classification
