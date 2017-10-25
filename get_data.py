# -*- coding: utf-8 -*-

import requests
import cv2
import os
import numpy as np
import itertools
from multiprocessing import Pool
from pyprind import ProgBar
import sys

pic_num = 1


def store_raw_images(paths, links):
    global pic_num
    for link, path in zip(links, paths):
        print(u"Processing path {}".format(path))
        if not os.path.exists(path):
            os.makedirs(path)
        result = requests.get(link)
        image_urls = result.text.split('\n')

        pool = Pool(processes=128)
        inputs = zip(itertools.repeat(path), image_urls, itertools.count(pic_num))
        bar = ProgBar(len(inputs), stream=sys.stdout)
        for i in pool.imap(load_image, inputs):
            bar.update()


def load_image(*args, **kwargs):
    path, link, counter = args[0]
    global pic_num
    if pic_num < counter:
        pic_num = counter + 1
    try:
        assert isinstance(link, unicode)
        if not link.startswith(("http://", "https://")):
            link = "http://{}".format(link)
        file_name = path + "/" + str(counter) + ".jpg"
        r = requests.get(link, stream=True, timeout=10.0)
        if r.status_code == 200:
            with open(file_name, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        img = cv2.imread(file_name)
        if img is None:
            # print(u"File invalid: {}".format(counter))
            try:
                if os.path.isfile(file_name):
                    os.remove(file_name)

            except Exception, e:
                print(u"Unhandled exception")

    except Exception as e:
        print(str(e))


def remove_invalid(dir_paths):
    if not os.path.exists('invalid'):
        os.makedirs('invalid')

    for dir_path in dir_paths:
        for reference in os.listdir(dir_path):

            invalid_paths = os.listdir('invalid')
            for target in invalid_paths:
                try:
                    if target == ".DS_Store":
                        os.remove('invalid/' + str(target))
                        break
                    if reference == ".DS_Store":
                        os.remove('invalid/' + str(reference))
                        break

                    current_image_path = str(dir_path) + '/' + str(reference)
                    target_im = cv2.imread('invalid/' + str(target))
                    if target_im is not None:
                        reference_im = cv2.imread(current_image_path)
                        if reference_im is None:
                            print("Removed {}".format(current_image_path))
                            os.remove(current_image_path)

                        elif target_im.shape == reference_im.shape and not (
                        np.bitwise_xor(target_im, reference_im).any()):
                            print("Removed {}".format(current_image_path))
                            os.remove(current_image_path)
                            break

                except Exception as e:
                    print(str(e))


if __name__ == "__main__":
    links = [
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537']

    paths = ['pets', 'furniture', 'people', 'food', 'frankfurter', 'chili-dog', 'hotdog']

    store_raw_images(paths, links)
    remove_invalid(paths)
