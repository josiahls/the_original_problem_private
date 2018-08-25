import pickle


class Logger(object):
    def __init__(self, log: dict):
        self.log = log
        self.accuracy = {}
        self.prev_accuracy = {}

        # Try loading a previous run. This will allow making comparisons
        try:
            self.prev_accuracy = pickle.load(open('prev_accuracy.pickle', 'rb'))
        except IOError:
            print('There is no previous run.')

    def print_all(self):
        self.show_overall_analysis()

    def show_full(self):
        print(str(self.log))

    def show_overall_analysis(self, show_as_percent=True):
        accuracy_easy = self.__get_accuracy('easy')
        accuracy_medium_1 = self.__get_accuracy('medium_1')
        accuracy_medium_2 = self.__get_accuracy('medium_2')
        accuracy_hard = self.__get_accuracy('hard')

        overall_accuracy = 0.5 * accuracy_easy + 0.2 * accuracy_medium_1 + 0.2 * accuracy_medium_2 + 0.1 * accuracy_hard

        if show_as_percent:
            accuracy_easy *= 100
            accuracy_medium_1 *= 100
            accuracy_medium_2 *= 100
            accuracy_hard *= 100
            overall_accuracy *= 100
            print('Current Run Accuracy Percents %')
        else:
            print('Current Run Accuracy')

        print(f'\taccuracy_easy: {accuracy_easy}')
        print(f'\taccuracy_medium_1: {accuracy_medium_1}')
        print(f'\taccuracy_medium_2: {accuracy_medium_2}')
        print(f'\taccuracy_hard: {accuracy_hard}')
        print(f'\toverall_accuracy: {overall_accuracy}')

        self.accuracy = {'accuracy_easy': accuracy_easy, 'accuracy_medium_1': accuracy_medium_1,
                         'accuracy_medium_2': accuracy_medium_2, 'accuracy_hard': accuracy_hard,
                         'overall_accuracy': overall_accuracy}

    def show_accuracy_comparison(self):
        if self.accuracy and self.prev_accuracy:
            print('Previous Run Comparison')
            print(f'\taccuracy_easy: {self.accuracy["accuracy_easy"] - self.prev_accuracy["accuracy_easy"]}')
            print(f'\taccuracy_medium_1: '
                  f'{self.accuracy["accuracy_medium_1"] - self.prev_accuracy["accuracy_medium_1"]}')
            print(f'\taccuracy_medium_2: '
                  f'{self.accuracy["accuracy_medium_2"] - self.prev_accuracy["accuracy_medium_2"]}')
            print(f'\taccuracy_hard: {self.accuracy["accuracy_hard"] - self.prev_accuracy["accuracy_hard"]}')
            print(f'\toverall_accuracy: {self.accuracy["overall_accuracy"] - self.prev_accuracy["overall_accuracy"]}')
        else:
            print("The current run or the pre run is not set.")

    def save(self):
        file = open('prev_accuracy.pickle', 'wb')
        pickle.dump(self.accuracy, file, pickle.HIGHEST_PROTOCOL)

    def __get_accuracy(self, category):
        total_true = sum([self.log[category][image_type].count(True) for image_type in [_ for _ in self.log[category]]])
        total_element_num = sum([len(self.log[category][image_type]) for image_type in [_ for _ in self.log[category]]])
        return total_true / total_element_num if total_element_num != 0 else 0
