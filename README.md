# keras-eval

[![Build Status](https://travis-ci.org/triagemd/keras-eval.svg?branch=master)](https://travis-ci.org/triagemd/keras-eval)
[![PyPI version](https://badge.fury.io/py/keras-eval.svg)](https://badge.fury.io/py/keras-eval)
[![codecov](https://codecov.io/gh/triagemd/keras-eval/branch/master/graph/badge.svg)](https://codecov.io/gh/triagemd/keras-eval)

Evaluation abstraction for Keras models. [Example Notebook](https://github.com/triagemd/keras-eval/blob/master/example.ipynb)

Requires [keras-model-specs](https://github.com/triagemd/keras-model-specs).

# Example Evaluation

`probs, labels = evaluator.evaluate(data_dir=data_dir, top_k=2, confusion_matrix=True, save_confusion_matrix_path='cm.png')`

![Confusion_matrix](https://github.com/triagemd/keras-eval/blob/master/figs/confusion_matrix.png)

`evaluator.show_results('average')`

model | f1_score | fdr | positives | sensitivity | specificity | auroc | negatives | precision | accuracy
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
mobilenet_v1.h5 | 0.934 | 0.064 | 1869 | 0.934 | 0.934 | 0.807 | 131 | 0.936 | 0.934

`evaluator.show_results('individual')`

class | sensitivity | precision | f1_score | specificity | FDR | AUROC | TP | FP | FN | % of samples
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
cats | 0.907 | 0.960 | 0.933 | 0.962 | 0.040 | 1.0 | 907 | 38 | 93 | 50.0
dogs | 0.962 | 0.912 | 0.936 | 0.907 | 0.088 | 1.0 | 962 | 93 | 38 | 50.0

# Use the code

Clone the repository

`git clone https://github.com/triagemd/keras-eval.git`

To install project dependencies, inside the root folder run

`script/up`

## Evaluator Class

Easy predictions and evaluations for a single model or an ensemble of many models.

The format to load models is the following:

**For a single model**

```
# The input is the model path
model_path = '/model_folder/resnet_50/model.h5'

# Inside model folder, there should be a '/model_folder/resnet_50/model_spec.json' file
```

**For an ensemble of models**
```
# The input is the parent folder
ensemble_models_dir = '/model_folder'

# That folder must contain several folders containing the models we want to load with their respective specs
# e.g. '/model_folder/resnet_50/model.h5', '/model_folder/resnet_50/model_spec.json', '/model_folder/densenet201/model.h5', '/model_folder/densenet201/model_spec.json'

```

You can specify all the following options.

**Evaluator**

```
from keras_eval.eval import Evaluator

evaluator = Evaluator(
                data_dir=None,
                concepts=None,
                ensemble_models_dir=None,
                model_path=model_path,
                loss_function='categorical_crossentropy',
                metrics=['accuracy'],
                batch_size=32,
                verbose=0)
```

## Evaluation Functions

Evaluate a set of images.

Each sub-folder under `'data_dir/'` will be considered as a different class. E.g. `'data_dir/class_1/dog.jpg'` , `'data_dir/class_2/cat.jpg'`
If you are evaluating an ensemble of models, we currently allow for these probability combination modes: `'maximum'`, `'arithmetic'`, `'geometric'`, `'harmonic'`

**evaluate**
```
data_dir = 'tests/files/catdog/test/'
probabilities, labels = evaluator.evaluate(data_dir=None, top_k=2, filter_indices=None, confusion_matrix=False, save_confusion_matrix_path=None, verbose=1)
# probabilities.shape = [n_models, n_samples, n_classes]
# labels.shape = [n_samples, n_classes]
```

Predict class probabilities of a set of images from a folder.

**predict**
```
folder_path = 'tests/files/catdog/test/cat/'
probs = evaluator.predict(folder_path)
```

Predict class probabilities of a single image

**predict**
```
image_path = 'tests/files/catdog/test/cat/cat-1.jpg'
probs = evaluator.predict(image_path)
```

## Evaluator attributes

After making predictions you can access to `evaluator.image_paths` to get a list of the files forwarded.

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

**set_concepts**

```
concepts = [{'label':'Dog', 'id': 'dogs'},
           {'label':'Cat', 'id': 'cats'}]

evaluator.set_concepts(concepts)
```

## Extra

For more information check the [Example Notebook](https://github.com/triagemd/keras-eval/blob/master/example.ipynb) and the source code. 

## Contact

This library is mantained by [@triagemd](https://github.com/triagemd).
To report any problem or issue, please use the [Issues](https://github.com/triagemd/keras-eval/issues) section. 
