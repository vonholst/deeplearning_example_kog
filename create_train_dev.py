import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from shutil import copyfile

seed = 143
np.random.seed(seed)


def train_dev_splits_from_directories(test_percentage, max_dev):
    hotdogs = np.array(glob.glob('./hotdog/**/*.jpg'))
    non_hotdogs = np.array(glob.glob('./non-hotdog/**/*.jpg'))
    total_data_count = len(hotdogs) + len(non_hotdogs)

    max_train_percentage = float(max_dev) / total_data_count
    test_size = min(max_train_percentage, test_percentage)

    hotdog_train, hotdog_test, hotdog_train_label, hotdog_test_label = train_test_split(hotdogs, np.ones(hotdogs.shape),
                                                                                        test_size=test_size,
                                                                                        random_state=seed)
    non_hotdog_train, non_hotdog_test, non_hotdog_train_label, non_hotdog_test_label = train_test_split(non_hotdogs,
                                                                                                        np.zeros(
                                                                                                            non_hotdogs.shape),
                                                                                                        test_size=test_size,
                                                                                                        random_state=seed)

    train_files = np.concatenate((hotdog_train, non_hotdog_train))
    dev_files = np.concatenate((hotdog_test, non_hotdog_test))

    train_dir = './train/'
    dev_dir = './dev/'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)

    for filename in train_files:
        dst_filename = train_dir + "".join(filename.split('/')[1:])
        copyfile(filename, dst_filename)

    for filename in dev_files:
        dst_filename = dev_dir + "".join(filename.split('/')[1:])
        copyfile(filename, dst_filename)


if __name__ == "__main__":
    train_dev_splits_from_directories(0.3, 1000.0)
