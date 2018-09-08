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
"""
Note: Medium 2 has:
Mode for medium_2 is:
	ball is 161.0
	brick is 162.0
	cylinder is 117.33333333333333
	Mode is : 146.77777777777777
The values are very low, which is telltale that the lighting is bad.
Propose having a mex thresh of 166 mode. If below, then the image is very dark.
Also the median is very low:
Median for medium_2 is:
	ball is 149.66666666666666
	brick is 128.75
	cylinder is 123.16666666666667
	Median is : 133.86111111111111

Both mediums have low modes and medians


Entropy for hard is:
	ball is 13.487393685234817
	brick is 13.484935989008285
	cylinder is 13.504931440508395
	Entropy is : 13.492420371583833
Entropy for hard is predictably high.


Average for easy is:
	ball is 111.98553574045147
	brick is 103.08977062919521
	cylinder is 107.50633890346586
	Average is : 107.52721509103752
Easy has a very high average.

I propose at least a way for filtering image thresholds based on this.


I propose using fa variable called: average curve and line directions distribution.
Prediction:
brick: uneven dist
cylinder: partial dist
ball: even dist


"""

import numpy as np
from util import filters
from util.hough_accumulator import HoughAccumulator
from util import image

PARAMS = {}
CENTROIDS = []


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

    def fit(self, data: list, type=''):
        """
        fit will take in a data variable as such:

        data = [{'classification': np.array}]

        :param data:
        :return:
        """
        # Convert Data to the expected format
        # data = [{d[0]:self.convert(d[1])} for d in data]
        data = list(map(lambda x: [x[0], self.convert(x[1], x[2])], data))

        # Fill data with dummy data if there is missing data
        for label in self.labels:
            if label not in [f[0] for f in data]:
                data.append([label, [0,0]])

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
        print(str(self.centroids))
        global CENTROIDS
        CENTROIDS = self.centroids

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

    def entropy_of_im(self, im: np.array):
        """
        Code for this is found here:
        https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
        :param im:
        :return:
        """
        im = np.copy(im)
        N = 30
        dimensions = im.shape
        # for row in range(dimensions[0]):
        #     for col in range(dimensions[1]):
        #         Lx = np.max([0, col - N])
        #         Ux = np.min([dimensions[1], col + N])
        #         Ly = np.max([0, row - N])
        #         Uy = np.min([dimensions[0], row + N])
        #         region = im[Ly:Uy,Lx:Ux].flatten()
        #         im[row, col] = self.entropy(region)

        def get_region(im, col, row, N):
            Lx = np.max([0, col - N])
            Ux = np.min([dimensions[1], col + N])
            Ly = np.max([0, row - N])
            Uy = np.min([dimensions[0], row + N])
            return im[Ly:Uy, Lx:Ux].flatten()

        im = [[self.entropy(get_region(im, col, row, N)) for col in range(0, dimensions[1], N)]
              for row in range(0, dimensions[0], N)]

        return np.array(im)

    def entropy(self, signal):
        '''
        Code for this is found here:
        https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html

        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig = signal.size
        symset = list(set(signal))
        numsym = len(symset)
        propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
        ent = np.sum([p * np.log2(1.0 / p) for p in propab])
        return ent

    def entropy_fast(self, im:np.array):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        im = im.flatten()
        lensig = im.size
        symset = list(set(im))
        propab = [np.size(im[im == i]) / (1.0 * lensig) for i in symset]
        ent = np.sum([p * np.log2(1.0 / p) for p in propab])
        return ent


    def convert(self, im, type='') -> np.array:
        """ Curve, Line Tally image conversion method

        import matplotlib.pyplot as plt
        plt.imshow(im)
        plt.show()
        plt.imshow(G_magnitude)
        plt.show()
        plt.imshow(edge_im)
        plt.show()
        plt.imshow(accumulator)
        plt.show()
        """
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
        # Arctan2 works a little better here, allowing us to avoid dividing by zero:
        G_direction = np.arctan2(Gy, Gx)

        ### Pre Process Image
        # Get the entropy
        entropy_im = self.entropy_of_im(G_magnitude)
        entropy_im = np.flip(entropy_im, axis=0)
        entropy = np.average(entropy_im.flatten())
        print(entropy)
        #


        ### Set up the accumulator
        # How many bins for each variable in parameter space?
        phi_bins = 128
        theta_bins = 128
        edge_im = G_magnitude > \
                  np.max(G_magnitude) * PARAMS['G_MAG_MAX_ADJUST'] +           \
                  np.average(np.array(G_magnitude).flatten()) * PARAMS['G_MAG_AVERAGE_ADJUST'] + \
                  np.median(np.array(G_magnitude).flatten()) * PARAMS['G_MAG_MEDIAN_ADJUST']
        edge_direction_im = np.zeros(G_magnitude.shape) * np.NaN  # Create empty array o
        edge_direction_im[edge_im] = G_direction[edge_im]
        rho_min = -edge_im.shape[0]
        rho_max = edge_im.shape[1]

        # Compute the rho and theta values for the grids in our accumulator:
        ha = HoughAccumulator(theta_bins, phi_bins, phi_min=rho_min, phi_max=rho_max)
        y_coords, x_coords = np.where(edge_im)
        accumulator = ha.accumulatev2(x_coords, y_coords)

        if type != '':
            import matplotlib.pyplot as plt
            # plt.imshow(gray_im)
            # plt.title(type)
            # plt.show()
            # plt.imshow(G_magnitude)
            # plt.title(type)
            # plt.show()
            # plt.imshow(edge_im)
            # plt.title(type)
            # plt.show()
            # plt.imshow(accumulator)
            # plt.title(type)
            # plt.show()
            self.imshow_with_values(entropy_im)
            plt.title(type)
            plt.show()

        ### Set the features we need

        # Set up curve and line detection
        curves, curve_groups = self.count_curves(accumulator)
        # y_coords, x_coords = np.where(accumulator > CONSTANTS['LINE_ACCUMULATOR_THRESH'])
        lines, line_groups = self.count_lines(accumulator)
        print('Done!')
        return [curves, lines]

    def count_curves(self, line_detected_accumulator):
        """
        Quick evaluation:
        import matplotlib.pyplot as plt
        plt.imshow(line_detected_accumulator)
        plt.show()
        plt.imshow(accumulator)
        plt.show()
        plt.imshow(filtered_groups)
        plt.show()


        :param line_detected_accumulator:
        :return:
        """
        phi_bins = 128
        theta_bins = 128
        rho_min = -line_detected_accumulator.shape[0]
        rho_max = line_detected_accumulator.shape[1]
        # Compute the rho and theta values for the grids in our accumulator:
        ha = HoughAccumulator(theta_bins, phi_bins, phi_min=rho_min, phi_max=rho_max)
        y_coords, x_coords = np.where(line_detected_accumulator)
        ### Get the final accumulator with the curves
        # The first round of hough found lines, second round finds curves
        accumulator = ha.accumulatev2(x_coords, y_coords)
        ### Get the filtered version and then group them
        filtered_groups = accumulator > np.max(accumulator) - np.std(accumulator) * PARAMS['CURVE_ADJUST']
        y_coords, x_coords = np.where(filtered_groups)
        ### Get the number of connected components
        return self.group(y_coords, x_coords)

    def count_lines(self, line_detected_accumulator):
        filtered_groups = line_detected_accumulator > np.max(line_detected_accumulator) - \
                          np.median(line_detected_accumulator[np.nonzero(line_detected_accumulator)]) * PARAMS['LINE_ADJUST']

        y_coords, x_coords = np.where(filtered_groups)

        return self.group(y_coords, x_coords)

    def has_connection(self, group1, group2=None, x=None, y=None):
        if group2 is not None:
            for y, x in group2:
                if (x in [_[1] for _ in group1] or any([abs(_[1] - x) == 1 for _ in group1])) and \
                        (y in [_[0] for _ in group1] or any([abs(_[0] - y) == 1 for _ in group1])):
                    return True
        else:
            return (x in [_[1] for _ in group1] or any([abs(_[1] - x) == 1 for _ in group1])) and \
                   (y in [_[0] for _ in group1] or any([abs(_[0] - y) == 1 for _ in group1]))

    def group(self, y_coords: list, x_coords: list):
        groups = []
        # Group the lines together
        for y, x in zip(y_coords, x_coords):
            is_in = False
            # If the x and y are in a group (or close)
            for i, group in enumerate(groups):
                if self.has_connection(group, x=x, y=y):
                    groups[i].append([y, x])
                    is_in = True
                    break
            if not is_in:
                groups.append([[y, x]])

        ### Group the groups
        values = list(map(lambda x: x, groups))
        new_groups = [[y for y in groups if self.has_connection(y, x)][0] for x in values]
        new_groups = list(set(map(tuple, [set(map(tuple, _)) for _ in new_groups])))
        # Do a second pass
        values = list(map(lambda x: x, new_groups))
        new_groups = [[y for y in new_groups if self.has_connection(y, x)][0] for x in values]
        new_groups = list(set(map(tuple, [set(map(tuple, _)) for _ in new_groups])))

        return len(new_groups), new_groups

    def imshow_with_values(self, data: np.array):
        size = data.shape[0]
        import matplotlib.pyplot as plt
        # Limits for the extent
        x_start = 3.0
        x_end = 9.0
        y_start = 6.0
        y_end = 12.0

        extent = [x_start, x_end, y_start, y_end]

        # The normal figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        im = ax.imshow(data, extent=extent, origin='lower', interpolation='None', cmap='viridis')

        # Add the text
        jump_x = (x_end - x_start) / (2.0 * size)
        jump_y = (y_end - y_start) / (2.0 * size)
        x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
        y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = round(data[y_index, x_index], 2)
                text_x = x + jump_x
                text_y = y + jump_y
                ax.text(text_x, text_y, label, color='black', ha='center', va='center',
                        fontsize=20)
        fig.colorbar(im)
