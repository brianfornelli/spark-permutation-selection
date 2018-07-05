from pyspark import SparkContext
import argparse
import json
from sklearn.metrics import mean_absolute_error
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def _load_data(path):
    loader = np.load(path)
    X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    y = loader['y']
    return X, y


def run(sc, model_path, data_path, iterations, result_path):
    X, y = _load_data(data_path)
    model = joblib.load(model_path)
    pred = model.predict(X)
    base_score = mean_absolute_error(y, pred)
    # broadcast to workers
    data_bc = sc.broadcast(dict(data=X.data, indptr=X.indptr, indices=X.indices, shape=X.shape, y=y))
    model_bc = sc.broadcast(model)
    iterations_bc = sc.broadcast(iterations)
    base_score_bc = sc.broadcast(base_score)
    scorer_bc = sc.broadcast(mean_absolute_error)

    def _parallel_score(c_i):
        X = csr_matrix((data_bc.value['data'], data_bc.value['indices'], data_bc.value['indptr']),
                       shape=data_bc.value['shape'])
        y = data_bc.value['y']
        model = model_bc.value
        iterations = iterations_bc.value
        base_score = base_score_bc.value
        fx = scorer_bc.value
        x = X.copy()
        tmp_scores = np.zeros(iterations)
        for i in range(iterations):
            idx = np.random.permutation(y.shape[0])
            arr = x[:, c_i].todense()
            x[:, c_i] = arr[idx]
            pred = model.predict(x)
            tmp_scores[i] = -(base_score - fx(y, pred))
        mu = np.mean(tmp_scores)
        return (c_i, mu)

    results = sc.parallelize(range(X.shape[1])).map(_parallel_score).collect()
    c_is, scores = zip(*results)
    df = pd.DataFrame({'indx': c_is, 'mu': scores})
    df.sort_values('indx', inplace=True)
    df.to_csv(result_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=json.loads, required=True)
    args = parser.parse_args()
    config = args.config
    spark_context = SparkContext()
    run(spark_context, config['model_path'], config['data_path'], config['iterations'], config['result_path'])
