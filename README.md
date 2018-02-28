# keras-eval
Evaluation abstraction for Keras models. 

## Evaluator Class

Easy predictions and evaluations for a single model or an ensemble of many models. 
You can specify all the following options. 

**Evaluator**

```
from keras_eval.eval import Evaluator

evaluator = Evaluator({
        'data_dir': None,
        'class_dictionaries': None,
        'ensemble_models_dir': None,
        'model_dir': None,
        'loss_function': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'batch_size': 1,
        'verbose': 0,
    }
```

## Evaluation Functions

Evaluate a set of images. 

Each sub-folder under `'data_dir/'` will be considered as a different class. E.g. `'data_dir/class_1/dog.jpg'` , `'data_dir/class_2/cat.jpg'`

**evaluate**
```
data_dir = ''tests/files/catdog/test/'
probs, labels = evaluator.evaluate(data_dir=None, K=[1], filter_indices=None, confusion_matrix=False, save_confusion_matrix_path=None, combination_mode=None)
# probs.shape = [n_models, n_samples, n_classes]
# labels.shape = [n_samples, n_classes]
```

Predict class probabilities of a set of images from a folder.

**predict**
```
folder_path = ''tests/files/catdog/test/cat/'
probs, image_paths = evaluator.predict(folder_path)
```

Predict class probabilities of a single image

**predict_image**
```
image_path = ''tests/files/catdog/test/cat/cat-1.jpg'
probs = evaluator.predict_image(image_path)
```

## Additional options

You can also add more options once the Evaluator object has been created, like loading more models
or setting class names and abbreviations to show in the confusion matrix.

**add_model**

```
model_path = '/your_model_path/model.h5
# Also supports model custom objects
custom_objects = None
evaluator.add_model(model_path, custom_objects)
```

**add_model_ensemble**

```
model_path = '/your_model_ensemble_path/'
evaluator.add_model_ensemble(model_path)
```

**set_class_dictionaries**

```
dict_classes = [{'abbrev':'PSO', 'class_name': 'psoriasis'},
           {'abbrev':'SL', 'class_name': 'skin-lesion'},
           {'abbrev':'HS', 'class_name': 'healthy-skin'},
           {'abbrev':'NS', 'class_name': 'not_skin'}]
           
evaluator.set_class_dictionaries(dict_classes)
```


