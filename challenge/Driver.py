import copy
import os
# noinspection PyPackageRequirements
from scipy import ndimage
from pathlib import Path

from challenge.josiah_laivins import classify
from challenge.Logger import Logger


class Tester(object):

    def __init__(self):
        self.data_dir = str(Path(__file__).parents[1]) + os.sep + 'data'

        self.CATEGORIES = {'ball': [], 'brick': [], 'cylinder': []}

        self.LOG = {
            'easy': copy.deepcopy(self.CATEGORIES),
            'hard': copy.deepcopy(self.CATEGORIES),
            'medium_1': copy.deepcopy(self.CATEGORIES),
            'medium_2': copy.deepcopy(self.CATEGORIES)
        }

        # For each category: easy, medium_1, medium_2, hard
        for category in ['hard']:  # os.listdir(self.data_dir):
            category_path = self.data_dir + os.sep + category
            # For each type: ball, brick, cylinder:
            for image_type in os.listdir(category_path):
                type_path = category_path + os.sep + image_type
                for file in os.listdir(type_path):
                    # Open an image
                    im = ndimage.imread(type_path + os.sep + file)
                    # State whether the classification is correct or not
                    self.LOG[category][image_type].append(classify(im) == image_type)


if __name__ == '__main__':
    t = Tester()
    logger = Logger(t.LOG)
    logger.print_all()
    logger.show_accuracy_comparison()
    logger.save()
