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

CONSTANTS = {
    'IS_CURVE': 1,
    'CURVE_THRESH': 15,
    'CURVE_ACCUMULATOR_THRESH': 10,
    'LINE_ACCUMULATOR_THRESH': 90,
    'LINE_THRESH': 6,
}


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
        # Convert Data to the expected format
        # data = [{d[0]:self.convert(d[1])} for d in data]
        data = list(map(lambda x: [x[0], self.convert(x[1])], data))

        # Get the starting centroids for this respective channel
        self.centroids = {label: [data[[f[0] for f in data].index(label)][0]] +
                                 [data[[f[0] for f in data].index(label)][1]]
                          for label in self.labels}

        # Get those centroids as histograms
        self.centroids = {label: self.centroids[label][1]
                          for label in self.centroids}

        for i in range(self.max_iter):
            print(f'Iteration: {i}')
            self.classifications = {}

            # Init all recorded classifications
            for j in range(self.k):
                self.classifications[self.labels[j]] = []
            print('Setting Features')
            for feature_set in data:
                # feature set is a single histogram
                # distances = [np.linalg.norm(feature_set[1] - self.centroids[centroid][1]) for centroid in self.centroids]
                # classification = self.labels[distances.index(min(distances))]
                features = feature_set[1]
                self.classifications[feature_set[0]].append(features)

            # Get the previous centroid
            prev_centroids = {centroid: np.copy(self.centroids[centroid]) for centroid in self.centroids}
            print('Setting Centroids')
            # Set the new centroids
            for classification in self.classifications:
                # This is where the centroids get changed. Originally, they would separate based on
                # center of groups, however I need to differentiate this more.
                all_classifications = []
                for c in self.classifications:
                    all_classifications += self.classifications[c]

                # Right now: category average - all average = noiseless full diff
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # Remove noise from those centroids (keeping only common values for each class)
            # temp = {centroid: np.copy(self.centroids[centroid]) for centroid in self.centroids}
            # for c in self.centroids:
            #     for other in [others for others in self.centroids if others != c]:
            #         self.centroids[c] -= temp[other]
            #     self.centroids[c][self.centroids[c] < 0] = 0
            print('Optimizing Centroids')
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / (original_centroid + 0.001) *
                          100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / (original_centroid + 0.001) * 100.0))
                    optimized = False
            if optimized:
                break
        print("Hello")

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
        converted_data = self.convert(data)
        distances = [np.linalg.norm(converted_data - self.centroids[centroid])
                     for centroid in self.centroids]
        classification = self.labels[distances.index(min(distances))]
        return classification

    # def convert(self, im) -> np.array:
    #     """ General image conversion method"""
    #     return np.histogram(im, bins=255)[0]

    def convert(self, im) -> np.array:
        """ Curve, Line Tally image conversion method"""
        print('Converting...')
        ### Convert Image to gray
        gray_im = image.convert_to_grayscale(im / 255)
        ### Implement Sobel kernels as numpy arrays
        Kx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])

        Ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
        Gx = filters.filter_2d(gray_im, Kx)
        Gy = filters.filter_2d(gray_im, Ky)
        # Compute Gradient Magnitude and Direction:
        G_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
        ### Set up the accumulator
        # How many bins for each variable in parameter space?
        phi_bins = 128
        theta_bins = 128
        edge_im = G_magnitude > 1.0
        rho_min = -edge_im.shape[0]
        rho_max = edge_im.shape[1]
        # Compute the rho and theta values for the grids in our accumulator:
        ha = HoughAccumulator(theta_bins, phi_bins, phi_min=rho_min, phi_max=rho_max)
        y_coords, x_coords = np.where(edge_im)
        accumulator = ha.accumulate(x_coords, y_coords)
        ### Set up curve and line detection
        y_coords, x_coords = np.where((accumulator > CONSTANTS['CURVE_ACCUMULATOR_THRESH']) &
                                      (accumulator < CONSTANTS['LINE_ACCUMULATOR_THRESH']))
        curves = self.count_curves(y_coords, x_coords)
        y_coords, x_coords = np.where(accumulator > CONSTANTS['LINE_ACCUMULATOR_THRESH'])
        lines = self.count_lines(y_coords, x_coords)
        print('Done!')
        return [curves, lines]

    def count_curves(self, line_detected_accumulator):
        phi_bins = 128
        theta_bins = 128
        rho_min = -line_detected_accumulator.shape[0]
        rho_max = line_detected_accumulator.shape[1]
        # Compute the rho and theta values for the grids in our accumulator:
        ha = HoughAccumulator(theta_bins, phi_bins, phi_min=rho_min, phi_max=rho_max)
        y_coords, x_coords = np.where(line_detected_accumulator)
        accumulator = ha.accumulate(x_coords, y_coords)

        return np.sum(accumulator > 10)

    # def count_curves(self, y_coords: list, x_coords: list):
    #     groups = []
    #     # Group the lines together
    #     for y, x in zip(y_coords, x_coords):
    #         is_in = False
    #         # If the x and y are in a group (or close)
    #         for i, group in enumerate(groups):
    #             if x in [_[1] for _ in group] or any([abs(_[1] - x) == 1 for _ in group]) and \
    #                y in [_[0] for _ in group] or any([abs(_[0] - y) == 1 for _ in group]):
    #                 groups[i].append([y, x])
    #                 is_in = True
    #                 break
    #         if not is_in:
    #             groups.append([[y, x]])
    #
    #     return len([group for group in groups if len(group) > CONSTANTS['CURVE_THRESH']
    #                 and self.is_curve(group)])

    def count_lines(self, y_coords: list, x_coords: list):
        groups = []
        for y, x in zip(y_coords, x_coords):
            is_in = False
            # If the x and y are in a group (or close)
            for i, group in enumerate(groups):
                if x in [_[1] for _ in group] or any([abs(_[1] - x) == 1 for _ in group]) and \
                   y in [_[0] for _ in group] or any([abs(_[0] - y) == 1 for _ in group]):
                    groups[i].append([y, x])
                    is_in = True
                    break
            if not is_in:
                groups.append([[y, x]])

        return len([group for group in groups if len(group) < CONSTANTS['LINE_THRESH']])
