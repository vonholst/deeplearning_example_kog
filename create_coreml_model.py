from lib.helpers import create_coreml_model, get_training_parameters, load_model

options = get_training_parameters()
model, class_indices = load_model()

coreml_model = create_coreml_model(model, options, class_indices)
coreml_model.author = 'Per Default'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Model for identifying hotdogs in images.'
coreml_model.save("./model/hotdog_classifier.mlmodel")
