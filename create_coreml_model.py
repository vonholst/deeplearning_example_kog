from lib.helpers import create_coreml_model, get_training_parameters, load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
import json
import keras

options = get_training_parameters()

with open('./model/keras_model_classes.json') as data_file:
    class_indices = json.load(data_file)

if True:
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model("./model/keras_model.h5")
else:
    model = load_model("./model/keras_model.h5")

coreml_model = create_coreml_model(model, options, class_indices)
coreml_model.author = 'Per Default'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Model for identifying hotdogs in images.'
coreml_model.save("./model/hotdog_classifier.mlmodel")
