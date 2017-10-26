
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from lib import helpers, model

from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(123)  # for reproducibility

options = helpers.get_training_parameters(rows=224, cols=224)
img_rows = options["img_rows"]
img_cols = options["img_cols"]
image_gen_batch_size = options["image_gen_batch_size"]
image_scale = options["image_scale"]
input_shape = options["input_shape"]
epochs = options["epochs"]

train_image_path = './train'
validation_image_path = './dev'
check_point_path = './model/checkpoints'

train_datagen = ImageDataGenerator(
    rescale=image_scale,
    shear_range=0.3,
    zoom_range=0.3,
    rotation_range=30.0,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_image_path,
    target_size=(img_rows, img_cols),
    batch_size=image_gen_batch_size,
    class_mode='categorical',
    shuffle=True,
)


validation_datagen = ImageDataGenerator(
    rescale=image_scale,
)

validation_generator = validation_datagen.flow_from_directory(
    validation_image_path,
    target_size=(options['img_rows'], options['img_cols']),
    batch_size=image_gen_batch_size,
    class_mode='categorical')

number_of_classes = len(train_generator.class_indices)
# Define model architecture
# model = helpers.create_model(input_shape, number_of_classes=number_of_classes)

model = model.retrained_mobilenet()

# opt = optimizers.SGD()
opt = optimizers.adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

training_weights = helpers.calculate_training_weights(train_generator)
print("using class weights: {}".format(training_weights))
steps_per_epoch = int(train_generator.n/image_gen_batch_size)  # unique samples / batch_size ?
validation_steps = int(validation_generator.n/image_gen_batch_size)

if not os.path.exists(check_point_path):
    os.makedirs(check_point_path)

check_point_file = 'weights.ep_{epoch:02d}-val_acc_{val_acc:.2f}_val_loss_{val_loss:.3f}.hdf5'
check_point_path_definition = check_point_path + '/' + check_point_file
checkpoint_callback = ModelCheckpoint(check_point_path_definition, monitor='acc', verbose=0, save_best_only=False,
                                      save_weights_only=False, mode='auto', period=3)

training_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    class_weight=training_weights,
    callbacks=[checkpoint_callback],
    validation_data=validation_generator,
    validation_steps=validation_steps,
)

# serialize model to JSON
helpers.save_model(model, class_indices=train_generator.class_indices, training_history=training_history)

# Evaluate model on test data
print('Evaluating model')
score = model.evaluate_generator(validation_generator, 2*validation_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
