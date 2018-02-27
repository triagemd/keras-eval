import os
import json
import platform
import keras

from keras_eval.eval import Evaluator


def check_evaluate_on_catdog_datasets(eval_args={}):
    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test'),
        batch_size=1,
        **eval_args
    )

    probs, labels = evaluator.evaluate()

    # n_models x n_samples x n_classes
    assert len(probs.shape) == 3

    # n_samples x n_classes
    assert len(labels.shape) == 2


def check_predict_on_catdog_datasets(eval_args={}):
    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test/cat/'),
        batch_size=1,
        **eval_args
    )

    probs, images_path = evaluator.predict()

    # n_models x n_samples x n_classes
    assert len(probs.shape) == 3

    # 2 images in the folder
    assert len(images_path) == 2


def test_evaluator_mobilenet_v1_on_catdog_dataset():
    model = keras.applications.mobilenet.MobileNet()
    model.save('/tests/mobilenet.h5')

    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
    }

    with open('mobilenet_model_specs.txt', 'w') as outfile:
        json.dump(specs, outfile)

    check_eval_on_catdog_datasets({model_dir: '/tests/mobilenet.h5'})

    check_predict_on_catdog_datasets({model_dir: '/tests/mobilenet.h5'})
