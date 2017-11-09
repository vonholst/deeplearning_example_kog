# important!! keras==1.2.2

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from lib.helpers import get_training_parameters, plot_training_history
import json
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import keras

np.random.seed(123)  # for reproducibility

options = get_training_parameters(rows=224, cols=224)

validation_image_path = './dev'


val_datagen = ImageDataGenerator(
    rescale=options["image_scale"],
    # rotation_range=5.0,
    # shear_range=0.3,
    # zoom_range=0.3,
)

val_generator = val_datagen.flow_from_directory(
    validation_image_path,
    target_size=(options["img_rows"], options["img_cols"]),
    batch_size=options["image_gen_batch_size"],
    class_mode='categorical',
    shuffle=True)

# score = model.evaluate_generator(val_generator, 50)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def class_from_id(id, class_dictionary):
    for k, v in class_dictionary.iteritems():
        if v == id:
            return k
    return None


def show_result(im, target_class, result_class, result_confidence):
    plt.imshow(im)
    plt.title(u"{target}: {result_confidence}, max:({result_class})".format(target=target_class,
                                                                            result_confidence=result_confidence,
                                                                            result_class=result_class))
    plt.show()


if __name__ == "__main__":
    with open('./model/keras_model_training_history.json', 'r') as data_file:
        training_history = json.load(data_file)

    plot_training_history(training_history)

    with open('./model/keras_model_classes.json') as data_file:
        class_indices = json.load(data_file)

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model("./model/keras_model.h5")

    for _ in range(10):
        batch = val_generator.next()
        for i in range(batch[0].shape[0]):
            im = batch[0][i]
            target = batch[1][i]

            img = im.reshape(1, options["img_rows"], options["img_cols"], 3)
            result = model.predict(img)
            target_id = target.argmax()
            target_class = class_from_id(target_id, class_indices)
            result_class = class_from_id(result.argmax(), class_indices)
            result_confidence = result[0][target_id] * 100

            if target_class != result_class:
                show_result(im, target_class, result_class, result_confidence)
            else:
                print("Correct {}".format(i))
