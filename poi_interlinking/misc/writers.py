import os
import csv
import numpy as np

from poi_interlinking import helpers
from poi_interlinking import config


def save_features(fpath, data, delimiter=','):
    h = helpers.StaticValues(config.MLConf.classification_method)
    cols = ['index'] + h.final_cols + ['Class']
    # TODO: transform to metric (temporal for saving)
    data[:, 1] -= 1
    data[:, 1] *= -1
    data[:, -2] -= 1
    data[:, -2] *= -1

    np.savetxt(
        fpath, data, header=f'{delimiter}'.join(cols), comments='',
        fmt=f'{delimiter}'.join(['%i'] + ['%1.3f']*len(h.final_cols) + ['%i'])
    )


def write_results(fpath, results, delimiter='&'):
    """
    Writes full and averaged experiment results.

    Args:
        fpath (str): Path to write
        results (dict): Contains metrics as keys and the corresponding values \
            values

    Returns:
        None
    """
    file_exists = True
    if not os.path.exists(fpath): file_exists = False

    with open(fpath, 'a+') as file:
        writer = csv.writer(file, delimiter=delimiter)
        if not file_exists:
            writer.writerow(results.keys())
        writer.writerow(results.values())
