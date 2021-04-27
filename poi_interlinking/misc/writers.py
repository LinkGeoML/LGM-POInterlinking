import os
import csv
import numpy as np

from poi_interlinking import helpers
from poi_interlinking import config


def save_features(fpath, data, delimiter=',', cols=None):
    h = helpers.StaticValues(config.MLConf.classification_method)
    col_names = h.final_cols + ['class'] if cols is None else cols
    col_format = ['%1.3f'] * (len(col_names) - 1) + ['%i']
    # TODO: transform to metric (temporal for saving)
    # data[:, 1] -= 1
    # data[:, 1] *= -1
    # data[:, -2] -= 1
    # data[:, -2] *= -1

    np.savetxt(
        fpath, data, header=f'{delimiter}'.join(['index'] + col_names), comments='',
        fmt=f'{delimiter}'.join(['%i'] + col_format)
    )


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def write_results(fpath, results, delimiter='&'):
    """
    Writes full and averaged experiment results.

    Args:
        fpath (:obj:`str`): Path to write.
        results (dict): Contains metrics as keys and the corresponding values values.
        delimiter (str): Field delimiter to use.
    """
    file_exists = True
    if not os.path.exists(fpath): file_exists = False

    results = without_keys(results, 'fimportances')
    with open(fpath, 'a+') as file:
        writer = csv.writer(file, delimiter=delimiter)
        if not file_exists:
            writer.writerow(results.keys())
        writer.writerow(results.values())
