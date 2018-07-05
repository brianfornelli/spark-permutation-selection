from bidarka import spark
import atexit
import os
import shutil
import tempfile
from sklearn.externals import joblib
import numpy as np
import logging


def _delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


class TempMemmap(object):
    def __init__(self, model, X, y):
        self.temp_folder = tempfile.mkdtemp(prefix="spark_permutation_")
        self.model = model
        self.X = X
        self.y = y
        self.model_path = os.path.join(self.temp_folder, 'model.pkl')
        self.data_path = os.path.join(self.temp_folder, 'data.npz')

    def __enter__(self):
        joblib.dump(self.model, self.model_path)
        np.savez(self.data_path, data=self.X.data, indptr=self.X.indptr, indices=self.X.indices,
                 shape=self.X.shape, y=self.y)
        atexit.register(lambda : _delete_folder(self.temp_folder))
        return self.model_path, self.data_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        _delete_folder()


def _get_spark_path(classifier=True):
    if classifier:
        stub = "spark_explore_classifier.py"
    else:
        stub = "spark_explore_regression.py"
    return os.path.join(os.path.dirname(__file__), stub)


def run(fitted_model, sparse_x, y, iterations, result_path, classifier):
    with TempMemmap(model=fitted_model, X=sparse_x, y=y) as (model_path, data_path):
        logging.info("model_path = '{}', data_path = '{}'".format(model_path, data_path))
        spark_script = _get_spark_path(classifier)
        logging.info("script_path = '{}'".format(spark_script))
        spark(script_path=spark_script, payload=dict(model_path=model_path, data_path=data_path,
                                                     iterations=iterations, result_path=result_path))
