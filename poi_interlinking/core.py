# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import time
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from shutil import copyfile
from datetime import datetime
import numpy as np
from collections import defaultdict
from beautifultable import BeautifulTable

from poi_interlinking import config, helpers
from poi_interlinking.learning import hyperparam_tuning
from poi_interlinking.processing.features import Features
from poi_interlinking.processing.sim_measures import LGMSimVars
from poi_interlinking.misc import writers


class StrategyEvaluator:
    """
    This class implements the pipeline for various strategies.
    """
    def __init__(self, encoding='latin'):
        self.encoding = encoding

    def hyperparamTuning(self, dataset):
        """A complete process of distinct steps in figuring out the best ML algorithm with optimal hyperparameters that
        fit the ``dataset`` for the toponym interlinking problem.

        :param dataset: Name of the dataset to use for training and evaluating various classifiers.
        :type dataset: str
        """
        tot_time = time.time()

        LGMSimVars.per_metric_optValues = config.MLConf.sim_opt_params[self.encoding.lower()]

        f = Features()
        pt = hyperparam_tuning.ParamTuning()

        start_time = time.time()
        assert (os.path.isfile(os.path.join(config.default_data_path, dataset))), \
            f'{dataset} dataset does not exist'
        f.load_data(os.path.join(config.default_data_path, dataset), self.encoding)
        fX, y = f.build()
        print("Loaded dataset and build features for {} setup; {} sec.".format(
            config.MLConf.classification_method, time.time() - start_time))

        skf = StratifiedShuffleSplit(n_splits=1, random_state=config.seed_no, test_size=config.test_size)
        for train_idxs, test_idxs in skf.split(fX, y):
            fX_train, fX_test = fX[train_idxs], fX[test_idxs]
            y_train, y_test = y[train_idxs], y[test_idxs]

            start_time = time.time()
            # 1st phase: find out best classifier from a list of candidate ones
            best_clf = pt.fineTuneClassifiers(fX_train, y_train)
            print("Best classifier {} with hyperparams {} and score {}; {} sec.".format(
                best_clf['classifier'], best_clf['hyperparams'], best_clf['score'], time.time() - start_time)
            )

            # start_time = time.time()
            # # 2nd phase: train the fine tuned best classifier on the whole train dataset (no folds)
            # estimator = pt.trainClassifier(fX_train, y_train, best_clf['estimator'])
            # print("Finished training model on the dataset; {} sec.".format(time.time() - start_time))

            # start_time = time.time()
            # 3th phase: test the fine tuned best classifier on the test dataset
            metrics = pt.testClassifier(fX_test, y_test, best_clf['estimator'])

            res = dict(
                Classifier=best_clf['classifier'], **metrics,
                fimportances=best_clf['importances'] if 'importances' in best_clf else None,
                time=time.time() - start_time
            )
            self._print_stats(res)

        print("The whole process took {} sec.".format(time.time() - tot_time))

    def evaluate(self, dataset, is_build=False):
        """Train and evaluate supported ML algorithms with custom hyper-parameters on dataset.

        :param dataset: Name of the dataset to use for training and evaluating various classifiers.
        :type dataset: str
        """
        tot_time = time.time()

        # Create folder to store experiments
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        exp_folder = os.path.join('experiments', f'exp_{date_time}')
        os.makedirs(exp_folder)
        copyfile('poi_interlinking/config.py', os.path.join(exp_folder, 'config.py'))

        f = Features()
        pt = hyperparam_tuning.ParamTuning()

        LGMSimVars.per_metric_optValues = config.MLConf.sim_opt_params[self.encoding.lower()]

        start_time = time.time()
        assert (os.path.isfile(os.path.join(config.default_data_path, dataset))), \
            f'{os.path.join(config.default_data_path, dataset)} dataset does not exist!!!'
        f.load_data(os.path.join(config.default_data_path, dataset), self.encoding)
        if not is_build:
            fX, y = f.build()
            print("Loaded dataset and build features for {} setup; {} sec.".format(
                config.MLConf.classification_method, time.time() - start_time))
        else:
            tmp_df = f.get_loaded_data()
            y = tmp_df[config.use_cols['status']].to_numpy()
            tmp_df.drop(columns=[config.use_cols['status'], 'index'], inplace=True)
            fX = tmp_df.to_numpy()
            print("Loaded dataset with pre-built features; {} sec.".format(time.time() - start_time))

        # fX_train, fX_test, y_train, y_test, train_set_df, test_set_df = train_test_split(
        #     fX, y, f.get_loaded_data(), stratify=y, test_size=config.test_size, random_state=config.seed_no)
        skf = StratifiedShuffleSplit(n_splits=config.MLConf.kfold_no, random_state=config.seed_no,
                                     test_size=config.test_size)
        fold = 1

        res = dict()
        for train_idxs, test_idxs in skf.split(fX, y):
            print(f'Evaluating models on fold {fold}...')
            fX_train, fX_test, train_set_df = fX[train_idxs], fX[test_idxs], f.get_loaded_data().iloc[train_idxs]
            y_train, y_test, test_set_df = y[train_idxs], y[test_idxs], f.get_loaded_data().iloc[test_idxs]

            if config.save_intermediate_results:
                fold_path = os.path.join(exp_folder, f'fold_{fold}')
                os.makedirs(fold_path)

                writers.save_features(
                    os.path.join(fold_path, 'train_features_build.csv'),
                    np.concatenate((
                        (train_idxs + 1)[:, np.newaxis], fX_train, y_train[:, np.newaxis]
                    ), axis=1))
                writers.save_features(
                    os.path.join(fold_path, 'test_features_build.csv'),
                    np.concatenate((
                        (test_idxs + 1)[:, np.newaxis], fX_test, y_test[:, np.newaxis]
                    ), axis=1))

                # if not is_build:
                #     train_set_df.reset_index(drop=True).to_csv(os.path.join(fold_path, 'train.csv'), index=True,
                #                                                index_label='index')
                #     test_set_df.reset_index(drop=True).to_csv(os.path.join(fold_path, 'test.csv'), index=True,
                #                                               index_label='index')

            for clf in config.MLConf.clf_custom_params:
                start_time = time.time()
                # 1st phase: train each classifier on the whole train dataset (no folds)
                estimator = pt.clf_names[clf][0](**config.MLConf.clf_custom_params[clf])
                estimator = pt.trainClassifier(fX_train, y_train, estimator)
                print(f"Finished training {clf} model; {time.time() - start_time} sec.")

                # start_time = time.time()
                # 2nd phase: test each classifier on the test dataset
                metrics = pt.testClassifier(fX_test, y_test, estimator)

                if config.save_intermediate_results:
                    writers.save_features(
                        os.path.join(fold_path, f'train_proba_{clf}.csv'),
                        np.concatenate((
                            (train_idxs + 1)[:, np.newaxis], estimator.predict_proba(fX_train),
                            estimator.predict(fX_train)[:, np.newaxis]  # , y_train[:, np.newaxis]
                        ), axis=1),
                        cols=['prob_class_0', 'prob_class_1', 'pred_class']
                    )
                    writers.save_features(
                        os.path.join(fold_path, f'test_proba_{clf}.csv'),
                        np.concatenate((
                            (test_idxs + 1)[:, np.newaxis], estimator.predict_proba(fX_test),
                            estimator.predict(fX_test)[:, np.newaxis]  # , y_test[:, np.newaxis]
                        ), axis=1),
                        cols=['prob_class_0', 'prob_class_1', 'pred_class']
                    )

                if clf not in res: res[clf] = defaultdict(list)
                for m, v in metrics.items():
                    res[clf][m].append(v)
                res[clf]['time'].append(time.time() - start_time)

                if hasattr(estimator, 'feature_importances_'):
                    res[clf]['fimportances'].append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    res[clf]['fimportances'].append(estimator.coef_)

            fold += 1

        for clf, metrics in res.items():
            print('Method {}'.format(clf))
            print('=======', end='')
            print(len(clf) * '=')

            output = dict()
            for m, v in metrics.items():
                if m == 'fimportances': output[m] = np.mean(v, axis=0)
                else: output[m] = np.mean(v)

            # res = dict(
            #     Classifier=clf, **output,
            #     fimportances=clf['importances'] if 'importances' in best_clf else None,
            #     # time=time.time() - start_time
            # )
            self._print_stats(dict(Classifier=clf, **output))
            writers.write_results(os.path.join(exp_folder, 'output.csv'), dict(Classifier=clf, **output))

        print("The whole process took {} sec.\n".format(time.time() - tot_time))

    def evaluate_on_pre_split(self, dtrain, dtest, is_build=False):
        """Train and evaluate supported ML algorithms with custom hyper-parameters on dataset.

        :param dataset: Name of the dataset to use for training and evaluating various classifiers.
        :type dataset: str
        """
        tot_time = time.time()

        # Create folder to store experiments
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        exp_folder = os.path.join('experiments', f'exp_{date_time}')
        os.makedirs(exp_folder)
        copyfile('poi_interlinking/config.py', os.path.join(exp_folder, 'config.py'))

        LGMSimVars.per_metric_optValues = config.MLConf.sim_opt_params[self.encoding.lower()]

        f = Features()
        pt = hyperparam_tuning.ParamTuning()

        start_time = time.time()
        assert (os.path.isfile(os.path.join(config.default_data_path, dtrain))), \
            f'{os.path.join(config.default_data_path, dtrain)} dataset does not exist!!!'
        f.load_data(os.path.join(config.default_data_path, dtrain), self.encoding)
        if not is_build:
            fX_train, y_train = f.build()
            print("Loaded train dataset {} and build features for {} setup; {} sec.".format(
                dtrain, config.MLConf.classification_method, time.time() - start_time))
        else:
            tmp_df = f.get_loaded_data()
            y_train = tmp_df[config.use_cols['status']].to_numpy()
            tmp_df.drop(columns=[config.use_cols['status'], 'index'], inplace=True)
            fX_train = tmp_df.to_numpy()
            print("Loaded train dataset {} with pre-built features; {} sec.".format(dtrain, time.time() - start_time))

        print(f'Using {config.train_size * 100}% of the training dataset.')
        skf = StratifiedShuffleSplit(n_splits=1, random_state=config.seed_no, train_size=config.train_size)
        for train_idxs, test_idxs in skf.split(fX_train, y_train):
            fX_train, y_train = fX_train[train_idxs], y_train[train_idxs]

        start_time = time.time()
        assert (os.path.isfile(os.path.join(config.default_data_path, dtest))), \
            f'{os.path.join(config.default_data_path, dtest)} dataset does not exist!!!'
        f.load_data(os.path.join(config.default_data_path, dtest), self.encoding)
        if not is_build:
            fX_test, y_test = f.build()
            print("Loaded test dataset {} and build features for {} setup; {} sec.".format(
                dtest, config.MLConf.classification_method, time.time() - start_time))
        else:
            tmp_df = f.get_loaded_data()
            y_test = tmp_df[config.use_cols['status']].to_numpy()
            tmp_df.drop(columns=[config.use_cols['status'], 'index'], inplace=True)
            fX_test = tmp_df.to_numpy()
            print("Loaded test dataset {} with pre-built features; {} sec.".format(dtest, time.time() - start_time))

        res = dict()
        for clf in config.MLConf.clf_custom_params:
            start_time = time.time()
            # 1st phase: train each classifier on the whole train dataset (no folds)
            estimator = pt.clf_names[clf][0](**config.MLConf.clf_custom_params[clf])
            estimator = pt.trainClassifier(fX_train, y_train, estimator)
            print(f"Finished training {clf} model; {time.time() - start_time} sec.")

            # start_time = time.time()
            # 2nd phase: test each classifier on the test dataset
            metrics = pt.testClassifier(fX_test, y_test, estimator)

            if clf not in res: res[clf] = defaultdict(list)
            for m, v in metrics.items():
                res[clf][m].append(v)
            res[clf]['time'].append(time.time() - start_time)

            if hasattr(estimator, 'feature_importances_'):
                res[clf]['fimportances'].append(estimator.feature_importances_)
            elif hasattr(estimator, 'coef_'):
                res[clf]['fimportances'].append(estimator.coef_)

        for clf, metrics in res.items():
            print('Method {}'.format(clf))
            print('=======', end='')
            print(len(clf) * '=')

            output = dict()
            for m, v in metrics.items():
                if m == 'fimportances': output[m] = np.mean(v, axis=0)
                else: output[m] = np.mean(v)

            self._print_stats(dict(Classifier=clf, **output))
            writers.write_results(os.path.join(exp_folder, 'output.csv'), dict(Classifier=clf, **output))

        print("The whole process took {} sec.\n".format(time.time() - tot_time))

    @staticmethod
    def _print_stats(params):
        print("| Method\t& Accuracy\t& Precision\t& Prec-weighted\t& Recall\t& Rec-weighted"
              "\t& F1-Score\t& F1-weighted"
              "\t& Roc-AUC\t& ROC-AUC-weighted"
              "\t& Time (sec)")
        print("||{}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}".format(
            params['Classifier'],
            params['Accuracy'],
            params['Precision'], params['Precision_weighted'],
            params['Recall'], params['Recall_weighted'],
            params['F1_score'], params['F1_score_weighted'],
            params['roc_auc'], params['roc_auc_weighted'],
            params['time']))

        if 'fimportances' in params and params['fimportances'] is not None:
            importances = np.ma.masked_equal(params['fimportances'], 0.0)
            if importances.mask is np.ma.nomask: importances.mask = np.zeros(importances.shape, dtype=bool)

            indices = np.argsort(importances.compressed())[::-1][
                      :min(importances.compressed().shape[0], config.MLConf.max_features_to_show)]
            headers = ["name", "score"]

            table = BeautifulTable()
            fcols = helpers.StaticValues(config.MLConf.classification_method).final_cols
            table.column_headers = headers

            for feature_name, val in zip(np.asarray(fcols, object)[~importances.mask][indices],
                                         importances.compressed()[indices]):
                table.append_row([feature_name, val])

            table.set_style(BeautifulTable.STYLE_RST)
            print(table)

        print()
