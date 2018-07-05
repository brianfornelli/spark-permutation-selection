from _permsel import run
import pandas as pd


def explore(fitted_model, sparse_x, y, iterations, result_path, classifier=True):
    run(fitted_model, sparse_x, y, iterations, result_path, classifier)
    df = pd.read_csv(result_path)
    return df
