# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import time
import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
from datetime import datetime

from poi_interlinking import config
from poi_interlinking.learning import hyperparam_tuning
from poi_interlinking.helpers import StaticValues
from poi_interlinking.processing.features import Features
from poi_interlinking.processing.sim_measures import LGMSimVars
from poi_interlinking.misc import writers


class StrategyEvaluator:
    """
    This class implements the pipeline for various strategies.
    """
    def __init__(self, encoding='latin'):
        self.encoding = encoding

    def hyperparamTuning(self, train_data, test_data):
        """A complete process of distinct steps in figuring out the best ML algorithm with best hyperparameters to
        toponym interlinking problem.

        :param train_data: Relative path to the train dataset.
        :type train_data: str
        :param test_data: Relative path to the test dataset.
        :type test_data: str
        """
        tot_time = time.time()

        LGMSimVars.per_metric_optValues = StaticValues.opt_values[self.encoding.lower()]
        assert (os.path.isfile(os.path.join(config.default_data_path, train_data))), \
            f'{train_data} dataset does not exist'
        assert (os.path.isfile(os.path.join(config.default_data_path, test_data))), \
            f'{test_data} dataset does not exist'

        f = Features()
        pt = hyperparam_tuning.ParamTuning()

        start_time = time.time()
        f.load_data(os.path.join(config.default_data_path, train_data), self.encoding)
        fX, y = f.build()
        print("Loaded train dataset and build features for {} setup; {} sec.".format(
            config.MLConf.classification_method, time.time() - start_time))

        start_time = time.time()
        # 1st phase: find out best classifier from a list of candidate ones
        best_clf = pt.fineTuneClassifiers(fX, y)
        print("Best classifier {} with hyperparams {} and score {}; {} sec.".format(
            best_clf['classifier'], best_clf['hyperparams'], best_clf['score'], time.time() - start_time)
        )

        start_time = time.time()
        # 2nd phase: train the fine tuned best classifier on the whole train dataset (no folds)
        estimator = pt.trainClassifier(fX, y, best_clf['estimator'])
        print("Finished training model on the dataset; {} sec.".format(time.time() - start_time))

        start_time = time.time()
        f.load_data(os.path.join(config.default_data_path, test_data), self.encoding)
        fX, y = f.build()
        print("Loaded test dataset and build features; {} sec".format(time.time() - start_time))

        start_time = time.time()
        # 3th phase: test the fine tuned best classifier on the test dataset
        metrics = pt.testClassifier(fX, y, estimator)

        res = dict(Classifier=best_clf['classifier'], **metrics, time=time.time() - start_time)
        self._print_stats(res)

        print("The whole process took {} sec.".format(time.time() - tot_time))

    def evaluate(self, dataset):
        """Train and evaluate selected ML algorithms with custom hyper-parameters on dataset.
        """
        tot_time = time.time()

        # Create folder to store experiments
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        exp_folder = os.path.join('experiments', f'exp_{date_time}')
        os.makedirs(exp_folder)

        copyfile('poi_interlinking/config.py', os.path.join(exp_folder, 'config.py'))

        LGMSimVars.per_metric_optValues = StaticValues.opt_values[self.encoding.lower()]
        assert (os.path.isfile(os.path.join(config.default_data_path, dataset))), \
            f'{os.path.join(config.default_data_path, dataset)} dataset does not exist!!!'

        f = Features()
        pt = hyperparam_tuning.ParamTuning()

        start_time = time.time()
        f.load_data(os.path.join(config.default_data_path, dataset), self.encoding)
        fX, y = f.build()
        print("Loaded dataset and build features for {} setup; {} sec.".format(
            config.MLConf.classification_method, time.time() - start_time))

        if config.save_intermediate_results:
            writers.save_features(os.path.join(exp_folder, 'features_build.csv'), fX)

        fX_train, fX_test, y_train, y_test = train_test_split(
            fX, y, stratify=y, test_size=config.test_size, random_state=config.seed_no)

        for clf in config.MLConf.clf_custom_params:
            print('Method {}'.format(clf))
            print('=======', end='')
            print(len(clf) * '=')

            start_time = time.time()
            # 1st phase: train each classifier on the whole train dataset (no folds)
            estimator = pt.clf_names[clf][0](**config.MLConf.clf_custom_params[clf])
            estimator = pt.trainClassifier(fX_train, y_train, estimator)
            print("Finished training model on dataset; {} sec.".format(time.time() - start_time))

            start_time = time.time()
            # 2nd phase: test each classifier on the test dataset
            metrics = pt.testClassifier(fX_test, y_test, estimator)

            res = dict(Classifier=clf, **metrics, time=time.time() - start_time)
            self._print_stats(res)
            writers.write_results(os.path.join(exp_folder, 'output.csv'), res)

        print("The whole process took {} sec.\n".format(time.time() - tot_time))

    @staticmethod
    def _print_stats(params):
        print("| Method\t& Accuracy\t& Precision\t& Prec-weighted\t& Recall\t& Rec-weighted"
              "\t& F1-Score\t& F1-weighted\t& Time (sec)")
        print("||{}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}".format(
            params['Classifier'],
            params['Accuracy'],
            params['Precision'], params['Precision_weighted'],
            params['Recall'], params['Recall_weighted'],
            params['F1_score'], params['F1_score_weighted'],
            params['time']))

        # if params['feature_importances'] is not None:
        #     importances = np.ma.masked_equal(params['feature_importances'], 0.0)
        #     if importances.mask is np.ma.nomask: importances.mask = np.zeros(importances.shape, dtype=bool)
        #
        #     indices = np.argsort(importances.compressed())[::-1][
        #               :min(importances.compressed().shape[0], self.max_features_toshow)]
        #     headers = ["name", "score"]
        #
        #     fcols = StaticValues.featureCols if config.MLConf.extra_features is False \
        #         else StaticValues.featureCols + StaticValues.extra_featureCols
        #     print(tabulate(zip(
        #         np.asarray(fcols, object)[~importances.mask][indices], importances.compressed()[indices]
        #     ), headers, tablefmt="simple"))

        print()
