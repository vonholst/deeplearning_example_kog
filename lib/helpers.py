from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import json
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.constraints import maxnorm
import matplotlib.pyplot as plt


def get_training_parameters(rows=128, cols=128):
    img_rows, img_cols = rows, cols
    input_shape = (img_rows, img_cols, 3)
    image_gen_batch_size = 128
    image_scale = 1. / 255.0
    epochs = 40
    samples_per_epoch = 1000

    options = dict(img_rows=img_rows,
                   img_cols=img_cols,
                   input_shape=input_shape,
                   image_gen_batch_size=image_gen_batch_size,
                   image_scale=image_scale,
                   epochs=epochs,
                   samples_per_epoch=samples_per_epoch,
                   )
    return options


def calculate_training_weights(image_generator):
    assert isinstance(image_generator, DirectoryIterator), 'Wrong class'
    training_examples = dict()
    max_training_examples = 0
    for class_name in image_generator.class_indices:
        class_identifier = image_generator.class_indices[class_name]
        number_of_class = np.sum(image_generator.classes == class_identifier)
        if number_of_class > max_training_examples:
            max_training_examples = number_of_class
        training_examples[class_identifier] = number_of_class

    training_weights = dict()
    for class_identifier in training_examples:
        training_weights[class_identifier] = float(max_training_examples) / training_examples[class_identifier]

    return training_weights


def generate_images(image_path, target_path):
    pass


def create_model(input_shape, number_of_classes):
    # Define model architecture
    # 1x100 -> (3) 32x100 ,(3) 32x100, [4] 25, (3) 25, (3) 25, [2] 12, (3) 12, (3) 12, [2] 6, FC...
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='sigmoid'))
    return model


def create_coreml_model(model, options, class_indices):
    import coremltools
    sorted_classes = sorted(class_indices.items(), key=lambda item: item[1])
    class_labels_sorted = [str(label) for label, index in sorted_classes]
    coreml_model = coremltools.converters.keras.convert(model, input_names='image',
                                                        image_input_names='image',
                                                        class_labels=class_labels_sorted,
                                                        image_scale=options["image_scale"])
    return coreml_model


def save_model(model, class_indices, training_history=None):
    model_json = model.to_json()
    with open("./model/keras_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model/keras_model.h5")
    print("Saved model to disk")
    with open("./model/keras_model_classes.json", 'w') as outfile:
        json.dump(class_indices, outfile)
    if training_history is not None:
        with open("./model/keras_model_training_history.json", 'w') as outfile:
            json.dump(training_history.history, outfile)


def load_model():
    # load json and create model
    json_file = open('./model/keras_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/keras_model.h5")

    with open('./model/keras_model_classes.json') as data_file:
        class_indices = json.load(data_file)
    return model, class_indices


def plot_training_history(history_dict):
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history_dict['acc'])
    plt.plot(history_dict['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

