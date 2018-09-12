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

notes:

look into itimeit
each image should take max 5 sec


"""

import numpy as np
from util import filters
from util import image

PARAMS = {
    'G_MAG_MAX_ADJUST': [.2],
    'G_MAG_AVERAGE_ADJUST': [0.0],  # , 0.5, 1],
    'G_MAG_MEDIAN_ADJUST': [0.0],  # , 0.5, 1],
    'G_MAG_ENTROPY_AVERAGE_ADJUST': [11.8],  # , 0.5, 1],
    'CURVE_ADJUST': [1],  # , -1.5, -.5, .5, 1.5],
    'LINE_ADJUST': [1.5]  # , -1.5, -.5, .5, 1.5]
}
# CENTROIDS = {'ball': np.array([1.11111111, 3.88888889, 42.42198876, 1.83979338]),
#              'brick': np.array([1.22222222, 2.11111111, 77.89421728, 1.79169806]), <- 27%
#              'cylinder': np.array([1., 3.22222222, 61.31741632, 1.85490538])}

CENTROIDS = {'brick': np.array([1.77777778, 0., 1.77776489, 0.99942041, 0.9991353,
                                0., 1.55555556]),
             'ball': np.array([0.55555556, 0.11111111, 1.66658852, 0.99966207, 0.99958828,
                               1.11111111, 0.]),
             'cylinder':
                 np.array([1.44444444, 0.66666667, 1.55545495, 0.99977806, 0.99931141,
                           0., 1.77777778])}


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
    clf = K_Means(labels=labels)
    clf.centroids = CENTROIDS
    #
    # gray_scale_image = image.convert_to_grayscale(im)
    #
    # random_integer = np.random.randint(low=0, high=3)
    return clf.predict(im)


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
                data.append([label, [0, 0, 0, 0, 0, 0, 0]])

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

    def entropy_of_im(self, im: np.array, region_size=30):
        """
        Code for this is found here:
        https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
        :param im:
        :return:
        """
        im = np.copy(im)
        N = region_size
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

    def entropy_fast(self, im: np.array):
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

    def mode(self, array: np.array):
        return max(set(array), key=array.count)

    def convert(self, im, type='') -> np.array:
        """ Curve, Line Tally image conversion method

        import matplotlib.pyplot as plt
        plt.imshow(im)
        plt.show()
        plt.imshow(G_magnitude_components)
        plt.show()
        plt.imshow(edge_im)
        plt.show()
        plt.imshow(accumulator)
        plt.show()
        """
        print('Converting...')
        ### Convert Image to gray
        gray_im = image.convert_to_grayscale(im / 255)
        gray_im = gray_im[::2, ::2]
        im = im[::2, ::2, :]
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
        G_magnitude_components = np.sqrt(Gx ** 2 + Gy ** 2)
        G_magnitude_original = np.copy(G_magnitude_components)
        # Arctan2 works a little better here, allowing us to avoid dividing by zero:
        G_direction = np.arctan2(Gy, Gx)

        ### Pre Process Image
        # Get the entropy
        region_size = 40
        entropy_im = self.entropy_of_im(G_magnitude_components, region_size)
        entropy_im = np.flip(entropy_im, axis=0)
        entropy_median = np.median([round(_, 2) for _ in entropy_im.flatten()])
        entropy_average = np.average([round(_, 2) for _ in entropy_im.flatten()])
        entropy_mode = self.mode([round(_, 1) for _ in entropy_im.flatten()])
        print(f'median: {entropy_median} avg: {entropy_average} mode: {entropy_mode}')
        # If the entropy is above a threshold, then do something different
        if entropy_average > PARAMS['G_MAG_ENTROPY_AVERAGE_ADJUST']:
            print('Performing Detailed Entropy...')
            # Get the region size that gets as close to the original resolution as possible
            region_grow = max([region_size - i for i in range(region_size)
                               if (region_size - i) * entropy_im.shape[0] <
                               G_magnitude_components.shape[0]])

            # Get all coords where the average entropy is above the average
            inflated = np.kron(entropy_im, np.ones([region_grow, region_grow], dtype=entropy_im.dtype))
            inflated = np.flip(inflated, axis=0)
            inflated = inflated[:,:]
            G_magnitude_components[np.where(inflated > entropy_mode - .13)] = 0
            G_magnitude_components[:, :region_grow] = 0  # Remove left boarder
            G_magnitude_components[:, -region_grow:-1] = 0  # Remove right boarder
            G_magnitude_components[:region_grow, :] = 0  # Remove top boarder
            G_magnitude_components[-region_grow:-1, :] = 0  # Remove bottom boarder
            region_size = region_grow
            adjuster = .2
        else:
            adjuster = .4

        # Dialate regions
        kernal = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        y_coords, x_coords = np.where(G_magnitude_components != 0)
        for y, x in zip(y_coords[::kernal.shape[0]], x_coords[::kernal.shape[0]]):
            G_magnitude_components[y - kernal.shape[0]:y + kernal.shape[1], \
            x - kernal.shape[0]: x + kernal.shape[1]] = G_magnitude_original[y - kernal.shape[0]:y + kernal.shape[1], \
                                                        x - kernal.shape[0]: x + kernal.shape[1]]

        ### Set up the accumulator
        # How many bins for each variable in parameter space?
        phi_bins = 64
        theta_bins = 64
        edge_im = G_magnitude_components > \
                  np.max(G_magnitude_components) * adjuster + \
                  np.average(np.array(G_magnitude_components).flatten()) * np.array(PARAMS['G_MAG_AVERAGE_ADJUST']) + \
                  np.median(np.array(G_magnitude_components).flatten()) * np.array(PARAMS['G_MAG_MEDIAN_ADJUST'])
        edge_direction_im = np.zeros(G_magnitude_components.shape) * np.NaN  # Create empty array o
        edge_direction_im[edge_im] = G_direction[edge_im]
        rho_min = -edge_im.shape[0]
        rho_max = edge_im.shape[1]

        # directions = G_direction[edge_im != 0]

        # Compute the rho and theta values for the grids in our accumulator:
        ha = HoughAccumulator(theta_bins, phi_bins, phi_min=rho_min, phi_max=rho_max)
        y_coords, x_coords = np.where(edge_im)
        accumulator = ha.accumulatev2(x_coords, y_coords)

        if type != '':
            pass
            import matplotlib.pyplot as plt
            # plt.imshow(gray_im)
            # plt.title(type)
            # plt.show()
            # plt.imshow(G_magnitude_components)
            # plt.title(type)
            # plt.show()
            plt.imshow(edge_im)
            plt.title(type)
            plt.show()
            # plt.imshow(accumulator)
            # plt.title(type)
            # plt.show()
            # self.imshow_with_values(entropy_im)
            # plt.title(type)
            # plt.show()

        ### Set the features we need
        # Ball: upper ten: std < 4, largest line < 40, avg upper < 30
        # Brick: upper ten: std > 9, largest line > 50, avg upper > 30
        # Cylinder: upper ten: std < 9, std > 3, std < 9, largest line: > 40, avg upper > 30
        upper_ten_percent = np.sort(accumulator.flatten())[int(len(accumulator.flatten()) * .99):]
        # upper_ten_percent_std = np.std(upper_ten_percent)
        # upper_ten_percent_average = np.average(upper_ten_percent)
        # upper_ten_percent_max = np.max(accumulator.flatten())

        # print(f' Largest Line: {np.max(accumulator.flatten())}'
        #       f' Average Upper Max {np.average(upper_ten_percent)}'
        #       f' Std of upper Max: {np.std(upper_ten_percent)}')
        # Set up curve and line detection
        # curves, curve_groups = self.count_curves(accumulator)
        # y_coords, x_coords = np.where(accumulator > CONSTANTS['LINE_ACCUMULATOR_THRESH'])
        lines, vert_pairs, line_groups = self.count_lines(accumulator, upper_ten_percent)

        """
        Notes:
        brick_3.jpg as classified as cylinder
        cylinder_1.jpg as classified as ball
        """

        accumulator_flattened = accumulator.flatten()
        accumulator_max_percent = np.max(accumulator_flattened) - np.std(accumulator_flattened)
        print(f'Max Percent: {accumulator_max_percent} Average {self.entropy(accumulator_flattened)}')

        # channel_dist = [np.sum(im[:, :, 0]), np.sum(im[:, :, 1]), np.sum(im[:, :, 2])] /\
        #                 max([np.sum(im[:, :, 0]), np.sum(im[:, :, 1]), np.sum(im[:, :, 2])])
        channel_dist = [np.sum(im[np.where(edge_im != 0), 0]), np.sum(im[np.where(edge_im != 0), 1]),
                        np.sum(im[np.where(edge_im != 0), 2])] / \
                       max([np.sum(im[np.where(edge_im != 0), 0]), np.sum(im[np.where(edge_im != 0), 1]),
                            np.sum(im[np.where(edge_im != 0), 2])])

        channel_dist[0] = channel_dist[0] if channel_dist[0] != 1 else channel_dist[0] * 2

        # features = [lines, channel_dist[0], channel_dist[1], channel_dist[2], np.average(im), upper_ten_percent_std < 4,
        #              upper_ten_percent_max < 40, upper_ten_percent_average < 30,
        #             upper_ten_percent_std > 9, upper_ten_percent_max > 50, upper_ten_percent_average > 30,
        #             3 < upper_ten_percent_std < 9, upper_ten_percent_max > 40, upper_ten_percent_average > 30]
        is_ball = accumulator_max_percent < 40
        not_a_ball = accumulator_max_percent > 56
        features = [lines, vert_pairs, channel_dist[0], channel_dist[1], channel_dist[2], int(is_ball) * 2,
                    not_a_ball * 2]

        # features = [self.entropy(np.array([round(_, 2) for _ in directions.flatten()])),
        #             accumulator_max_percent, np.std([round(_, 2) for _ in directions.flatten()]),
        #             len(set([round(_, 1) for _ in directions.flatten()])) /
        #             len([round(_, 1) for _ in directions.flatten()])]

        print(f'{str(features)}')
        print('Done!')
        return features

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
        phi_bins = 64
        theta_bins = 64
        rho_min = -line_detected_accumulator.shape[0]
        rho_max = line_detected_accumulator.shape[1]
        # Compute the rho and theta values for the grids in our accumulator:
        ha = HoughAccumulator(theta_bins, phi_bins, phi_min=rho_min, phi_max=rho_max)
        y_coords, x_coords = np.where(line_detected_accumulator)
        ### Get the final accumulator with the curves
        # The first round of hough found lines, second round finds curves
        accumulator = ha.accumulatev2(x_coords, y_coords)
        ### Get the filtered version and then group them
        filtered_groups = accumulator > np.max(accumulator) - np.std(accumulator) * np.array(PARAMS['CURVE_ADJUST'])
        y_coords, x_coords = np.where(filtered_groups)
        ### Get the number of connected components
        return self.group(y_coords, x_coords)

    def count_lines(self, line_detected_accumulator, upper_ten_percent):
        """
        Brick lines are more in the middle,
        circular detections / cylinder lines are on the corners.
        This is because:
        idk

        :param line_detected_accumulator:
        :param upper_ten_percent:
        :return:
        """

        max_min = min(upper_ten_percent[-3:])  # Keep the top most 3 lines (min lines to detect a brick)
        print(f'Max-Min: {max_min}')

        filtered_groups = line_detected_accumulator > max_min

        # import matplotlib.pyplot as plt
        # plt.imshow(filtered_groups)
        # plt.show()
        y_coords, x_coords = np.where(filtered_groups)

        # Interpret the points

        lines, line_groups = self.group(y_coords, x_coords)

        line_distances = [np.linalg.norm(np.array(line_groups[i]) - np.array(line_groups[i + 1]))
                          for i in range(len(line_groups) - 1)]

        # If the lines are too close to each other, then dont consider them lines, consider them noise
        if line_distances and min(line_distances) < 3 and len(line_distances) < 2:
            lines = 0

        # If they are far enough apart, then they might be vertical
        vert_pairs = 0
        if line_distances and min(line_distances) > 60 and min(line_distances) < 80:
            vert_pairs += 1

        # If the average locations of the points is on the edges then remove them
        center_points = [np.average(line_groups[i], axis=0) for i in range(len(line_groups))]
        center_point = np.average(center_points, axis=0)
        # Note: maybe decrease 80 to 70
        if center_point[1] > 60 or center_point[1] < 4 or (line_distances and min(line_distances) > 80):
            lines = 0

        return lines, vert_pairs, line_groups

    def has_connection(self, group1, group2=None, x=None, y=None, include_diagonal=True):
        if include_diagonal:
            if group2 is not None:
                for y, x in group2:
                    if (x in [_[1] for _ in group1] or any([abs(_[1] - x) == 1 for _ in group1])) and \
                            (y in [_[0] for _ in group1] or any([abs(_[0] - y) == 1 for _ in group1])):
                        return True
            else:
                return (x in [_[1] for _ in group1] or any([abs(_[1] - x) == 1 for _ in group1])) and \
                       (y in [_[0] for _ in group1] or any([abs(_[0] - y) == 1 for _ in group1]))
        else:
            if group2 is not None:
                for y, x in group2:
                    if (x in [_[1] for _ in group1] or any([abs(_[1] - x) == 1 for _ in group1])) and \
                            (y in [_[0] for _ in group1] or any([abs(_[0] - y) == 1 for _ in group1])) and \
                            not ((any([abs(_[0] - y) == 1 for _ in group1])) and
                                 any([abs(_[1] - x) == 1 for _ in group1])):
                        return True
            else:
                return (x in [_[1] for _ in group1] or any([abs(_[1] - x) == 1 for _ in group1])) and \
                       (y in [_[0] for _ in group1] or any([abs(_[0] - y) == 1 for _ in group1])) and \
                       not ((any([abs(_[0] - y) == 1 for _ in group1])) and
                            any([abs(_[1] - x) == 1 for _ in group1]))

    def group(self, y_coords: list, x_coords: list, include_diagonal=True):
        groups = []
        # Group the lines together
        for y, x in zip(y_coords, x_coords):
            is_in = False
            # If the x and y are in a group (or close)
            for i, group in enumerate(groups):
                if self.has_connection(group, x=x, y=y, include_diagonal=include_diagonal):
                    groups[i].append([y, x])
                    is_in = True
                    break
            if not is_in:
                groups.append([[y, x]])

        ### Group the groups
        values = list(map(lambda x: x, groups))
        new_groups = [[y for y in groups if self.has_connection(y, x, include_diagonal=include_diagonal)][0] for x in
                      values]
        new_groups = list(set(map(tuple, [set(map(tuple, _)) for _ in new_groups])))
        # Do a second pass
        values = list(map(lambda x: x, new_groups))
        new_groups = [[y for y in new_groups if self.has_connection(y, x, include_diagonal=include_diagonal)][0] for x
                      in values]
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


class HoughAccumulator(object):
    def __init__(self, theta_bins: int, phi_bins: int, phi_min: int, phi_max: int):
        '''
        Simple class to implement an accumalator for the hough transform.
        Args:
        theta_bins = number of bins to use for theta
        phi_bins = number of bins to use for phi
        phi_max = maximum phi value to accumulate
        phi_min = minimu phi value to acumulate
        '''
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.accumulator = np.zeros((self.phi_bins, self.theta_bins))

        # This covers all possible lines:
        theta_min = 0
        theta_max = np.pi

        # Compute the phi and theta values for the grids in our accumulator:
        self.rhos = np.linspace(phi_min, phi_max, self.accumulator.shape[0])
        self.thetas = np.linspace(theta_min, theta_max, self.accumulator.shape[1])

    def accumulate(self, x_coords: list, y_coords: list):
        '''
        Iterate through x and y coordinates, accumulate in hough space, and return.
        Args:
        x_coords = x-coordinates of points to transform
        y_coords = y-coordinats of points to transform

        Returns:
        accumulator = numpy array of accumulated values.
        '''

        for i in range(len(x_coords)):
            # Grab a single point
            x = x_coords[i]  # TODO This was originally called 'scaled-x'. Verify if a method needs to used to scale
            y = y_coords[i]  # TODO This was originally called 'scaled-y'. Verify if a method needs to used to scale

            # Actually do transform!
            curve_prho = x * np.cos(self.thetas) + y * np.sin(self.thetas)

            for j in range(len(self.thetas)):
                # Make sure that the part of the curve falls within our accumulator
                if np.min(abs(curve_prho[j] - self.rhos)) <= 1.0:
                    # Find the cell our curve goes through:
                    rho_index = np.argmin(abs(curve_prho[j] - self.rhos))
                    self.accumulator[rho_index, j] += 1

        return self.accumulator

    def accumulatev2(self, x_coords: list, y_coords: list):
        '''
        Iterate through x and y coordinates, accumulate in hough space, and return.
        Args:
        x_coords = x-coordinates of points to transform
        y_coords = y-coordinats of points to transform

        Returns:
        accumulator = numpy array of accumulated values.
        '''

        def accumulate_x(curve_prho):
            def j_is_valid(curve_prho) -> list:
                return [j for j in range(len(self.thetas)) if np.min(abs(curve_prho[j] - self.rhos)) <= 1.0]

            self.accumulator[[np.argmin(abs(curve_prho[j] - self.rhos)) for j in j_is_valid(curve_prho)],
                             j_is_valid(curve_prho)] += 1

        def get_curve_rho(index, x_coords, y_coords):
            return x_coords[index] * np.cos(self.thetas) + y_coords[index] * np.sin(self.thetas)

        for i in range(len(x_coords)):
            accumulate_x(get_curve_rho(i, x_coords, y_coords))

        return self.accumulator

    def clear_accumulator(self):
        '''
        Zero out accumulator
        '''
        self.accumulator = np.zeros((self.phi_bins, self.theta_bins))
