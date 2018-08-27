import copy
import os
# noinspection PyPackageRequirements
from scipy import ndimage
from pathlib import Path

from util import image
import numpy as np
from challenge.josiah_laivins import classify, K_Means
from challenge.Logger import Logger


class Tester(object):

    # noinspection PyPep8Naming
    def __init__(self):
        self.data_dir = str(Path(__file__).parents[1]) + os.sep + 'data'

        self.CATEGORIES = {'ball': [], 'brick': [], 'cylinder': []}

        # Specify file types that access
        CATEGORY_LIST = ['easy']
        OBJECT_TYPE = ['ball', 'brick', 'cylinder']
        FILE_NAME = ['ball_1.jpg', 'ball_2.jpg', 'brick_1.jpg', 'brick_2.jpg', 'cylinder_1.jpg', 'cylinder_2.jpg']

        self.LOG = {
            'easy': copy.deepcopy(self.CATEGORIES),
            'hard': copy.deepcopy(self.CATEGORIES),
            'medium_1': copy.deepcopy(self.CATEGORIES),
            'medium_2': copy.deepcopy(self.CATEGORIES)
        }

        clf = self.train(CATEGORY_LIST, OBJECT_TYPE, FILE_NAME)
        self.test(CATEGORY_LIST, OBJECT_TYPE, FILE_NAME, clf)

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
                    prediction = model.predict(im) == image_type
                    self.LOG[category][image_type].append(prediction)


if __name__ == '__main__':
    t = Tester()
    logger = Logger(t.LOG)
    logger.print_all()
    logger.show_accuracy_comparison()
    logger.save()
