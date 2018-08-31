import copy
import os
# noinspection PyPackageRequirements
from scipy import ndimage
from pathlib import Path
from itertools import product
from util import image
from scipy import stats
import numpy as np
from challenge.josiah_laivins import classify, K_Means
from challenge.Logger import Logger
import challenge

PARAMS = {
    'G_MAG_MIN': [1.0, 0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2],
    'G_MAG_AVERAGE_ADJUST': [0.0, 0.2, 0.5, 0.8, 1],
    'G_MAG_MEDIAN_ADJUST': [0.0, 0.2, 0.5, 0.8, 1],
    'CURVE_ADJUST': [0, -5, -10, 5, 10, -15, 15],
    'LINE_ADJUST': [0, -5, -10, 5, 10, -15, 15]
}


class Tester(object):

    # noinspection PyPep8Naming
    def __init__(self):
        self.data_dir = str(Path(__file__).parents[1]) + os.sep + 'data'

        self.CATEGORIES = {'ball': [], 'brick': [], 'cylinder': []}
        self.params = {}

        # Specify file types that access
        self.CATEGORY_LIST = ['easy']
        self.OBJECT_TYPE = ['ball', 'brick', 'cylinder']
        self.FILE_NAME = ['ball_5.jpg', 'brick_3.jpg',  'cylinder_3.jpg']

        self.LOG = {
            'easy': copy.deepcopy(self.CATEGORIES),
            'hard': copy.deepcopy(self.CATEGORIES),
            'medium_1': copy.deepcopy(self.CATEGORIES),
            'medium_2': copy.deepcopy(self.CATEGORIES)
        }

        # self.query(CATEGORY_LIST, OBJECT_TYPE, FILE_NAME, self.eval_entropy, name='Entropy')
        # self.query(CATEGORY_LIST, OBJECT_TYPE, FILE_NAME, self.eval_average, name='Average')
        # self.query(CATEGORY_LIST, OBJECT_TYPE, FILE_NAME, self.eval_mode, name='Mode')
        # self.query(CATEGORY_LIST, OBJECT_TYPE, FILE_NAME, self.eval_median, name='Median')

    def run(self):
        # Set the parameters
        challenge.josiah_laivins.PARAMS = self.params
        clf = self.train(self.CATEGORY_LIST, self.OBJECT_TYPE, self.FILE_NAME)
        self.test(self.CATEGORY_LIST, self.OBJECT_TYPE, self.FILE_NAME, clf)

    def eval_entropy(self, im: np.array):
        # pixel * log (pixel)
        return stats.entropy(np.array(im).flatten())

    def eval_average(self, im: np.array):
        return np.average(np.array(im).flatten())

    def eval_mode(self, im: np.array):
        return stats.mode(np.array(im).flatten())[0][0]

    def eval_median(self, im: np.array):
        return np.median(np.array(im).flatten())

    def train(self, category_list, object_type, file_name):
        data = []
        # For each category: easy, medium_1, medium_2, hard
        category_list = os.listdir(self.data_dir) if not category_list else category_list
        for category in category_list:
            print(f'Working through: {category}')
            category_path = self.data_dir + os.sep + category
            object_type_list = os.listdir(category_path) if not object_type else object_type
            # For each type: ball, brick, cylinder:
            for image_type in object_type_list:
                print(f'Working through: {image_type}')
                type_path = category_path + os.sep + image_type
                image_list = os.listdir(type_path) if not file_name else file_name
                # For each image
                for file in image_list:
                    # Open an image
                    try:
                        im = ndimage.imread(type_path + os.sep + file)
                    except FileNotFoundError:
                        continue

                    data.append([image_type, im])

        clf = K_Means(labels=object_type)
        clf.fit(data)
        return clf

    def test(self, category_list, object_type, file_name, model: K_Means):
        # For each category: easy, medium_1, medium_2, hard
        category_list = os.listdir(self.data_dir) if not category_list else category_list
        for category in category_list:
            print(f'Working through: {category}')
            category_path = self.data_dir + os.sep + category
            object_type_list = os.listdir(category_path) if not object_type else object_type
            # For each type: ball, brick, cylinder:
            for image_type in object_type_list:
                print(f'Working through: {image_type}')
                type_path = category_path + os.sep + image_type
                image_list = os.listdir(type_path) if not file_name else file_name
                # For each image
                for file in image_list:
                    # Open an image
                    try:
                        im = ndimage.imread(type_path + os.sep + file)
                    except FileNotFoundError:
                        continue

                    # prediction = model.predict(np.histogram(im, bins=255)[0]) == image_type
                    predicted_label = model.predict(im)
                    prediction = model.predict(im) == image_type
                    if not prediction:
                        print(f'File: {file} as classified as {predicted_label}')
                    self.LOG[category][image_type].append(prediction)

    def query(self, category_list, object_type, file_name, evaluation_function, name):
        print(f'Getting the {name}')
        # For each category: easy, medium_1, medium_2, hard
        category_list = os.listdir(self.data_dir) if not category_list else category_list
        for category in category_list:
            print(f'Working through: {category}')
            category_path = self.data_dir + os.sep + category
            object_type_list = os.listdir(category_path) if not object_type else object_type
            # For each type: ball, brick, cylinder:
            for image_type in object_type_list:
                print(f'Working through: {image_type}')
                type_path = category_path + os.sep + image_type
                image_list = os.listdir(type_path) if not file_name else file_name
                # For each image
                for file in image_list:
                    # Open an image
                    try:
                        im = ndimage.imread(type_path + os.sep + file)
                    except FileNotFoundError:
                        continue

                    evaluation = evaluation_function(im)
                    self.LOG[category][image_type].append(evaluation)

        ### Show average of each category
        for category in self.LOG:
            average = 0
            print(f'{name} for {category} is: ')
            for label in self.LOG[category]:
                a = np.average(self.LOG[category][label])
                average += a
                print(f'\t{label} is ' + str(a))
            print(f'\t{name} is : {average/len(self.LOG[category])}')

    def set_params(self, params: dict):
        self.params = params


if __name__ == '__main__':
    logger = Logger({})
    logger.show_prev_batch()
    for combination in list(product(*PARAMS.values())):
        t = Tester()
        t.set_params({key: combination[i] for i, key in enumerate(PARAMS)})
        t.run()
        logger.set_log(t.LOG)
        logger.print_all()
        logger.show_accuracy_comparison()
        logger.set_batch(str({key: combination[i] for i, key in enumerate(PARAMS)}) +
                         str(challenge.josiah_laivins.CENTROIDS))
        logger.show_batch()
    logger.batch_save()
    logger.save()

    ## Normal:
    # t = Tester()
    # t.set_params(PARAMS)
    # logger = Logger(t.LOG)
    # logger.print_all()
    # logger.show_accuracy_comparison()
    # logger.save()
