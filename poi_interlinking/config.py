# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm


default_data_path = 'data'
freq_term_size = 400

# fieldnames = ["s1", "s2", "status", "c1", "c2", "a1", "a2", "cc1", "cc2"]
fieldnames = None
use_cols = dict(
    ID1='ID1', ID2='ID2',
    s1='Name1', s2='Name2', addr1='Address1', addr2='Address2',
    lon1='st_x1', lat1='st_y1', lon2='st_x2', lat2='st_y2',
    status='Class'
)
delimiter = ','

# #: Relative path to the train dataset. This value is used only when the *dtrain* cmd argument is None.
# train_dataset = 'data/dataset-string-similarity_global_1k.csv'
# # train_dataset = 'data/dataset-string-similarity_latin_EU_NA_1k.txt'
# # train_dataset = 'data/dataset-string-similarity-100.csv'
#
# #: Relative path to the test dataset. This value is used only when the *dtest* cmd argument is None.
# test_dataset = 'data/dataset-string-similarity.txt'

#: float: Similarity threshold on whether sorting on toponym tokens is applied or not. It is triggered on a score
#: below the assigned threshold.
sort_thres = 0.55

#: int: Seed used by each of the random number generators.
seed_no = 13

test_size = 0.2

save_intermediate_results = False


class MLConf:
    """
    This class initializes parameters that correspond to the machine learning part of the framework.

    These variables define the parameter grid for GridSearchCV:

    :cvar SVM_hyperparameters: Defines the search space for SVM.
    :vartype SVM_hyperparameters: :obj:`list`
    :cvar MLP_hyperparameters: Defines the search space for MLP.
    :vartype MLP_hyperparameters: :obj:`dict`
    :cvar DecisionTree_hyperparameters: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters: :obj:`dict`
    :cvar RandomForest_hyperparameters: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters: :obj:`dict`
    :cvar XGBoost_hyperparameters: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters: :obj:`dict`

    These variables define the parameter grid for RandomizedSearchCV where continuous distributions are used for
    continuous parameters (whenever this is feasible):

    :cvar SVM_hyperparameters_dist: Defines the search space for SVM.
    :vartype SVM_hyperparameters_dist: :obj:`dict`
    :cvar MLP_hyperparameters_dist: Defines the search space for MLP.
    :vartype MLP_hyperparameters_dist: :obj:`dict`
    :cvar DecisionTree_hyperparameters_dist: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters_dist: :obj:`dict`
    :cvar RandomForest_hyperparameters_dist: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters_dist: :obj:`dict`
    :cvar XGBoost_hyperparameters_dist: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters_dist: :obj:`dict`
    """

    kfold_no = 5
    """int: The number of outer folds that splits the dataset for the k-fold cross-validation.
    """

    #: int: The number of inner folds that splits the dataset for the k-fold cross-validation.
    kfold_inner_parameter = 4

    n_jobs = 4  #: int: Number of parallel jobs to be initiated. -1 means to utilize all available processors.

    classification_method = 'lgm'
    """str: The classification group of features to use. (*basic* | *basic_sorted* | *lgm*).

    See Also
    --------
    :class:`~poi_interlinking.processing.features.Features` : Details on the supported groups.
    """

    # accepted values: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'grid'
    """str: Search Method to use for finding best hyperparameters. (*randomized* | *grid*).
    
    See Also
    --------     
    :meth:`~poi_interlinking.learning.hyperparam_tuning.ParamTuning.fineTuneClassifiers` : Details on the
        supported methods.        
    """
    #: int: Number of iterations that RandomizedSearchCV should execute. It applies only when
    #: :attr:`hyperparams_search_method` equals to 'randomized'.
    max_iter = 300

    #: int: Number of ranked features to print
    max_features_to_show = 10

    classifiers = [
        # 'SVM',
        # 'DecisionTree',
        'RandomForest',
        # 'ExtraTrees',
        # 'XGBoost',
        # 'MLP'
    ]
    """list of str: Define the classifiers to apply on code execution. Accepted values are: 

    - SVM 
    - DecisionTree
    - RandomForest
    - ExtraTrees
    - XGBoost
    - MLP.
    """

    # score = 'roc_auc_ovr_weighted'
    score = 'roc_auc'
    """str: The metric to optimize on hyper-parameter tuning. Possible valid values presented on `Scikit predefined values`_. 

    .. _Scikit predefined values:
        https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    """

    clf_custom_params = {
        'SVM': {
            # default
            # 'C': 1.0, 'max_iter': 3000,
            # best
            'C': 100, 'class_weight': 'balanced', 'gamma': 0.01, 'kernel': 'sigmoid', 'max_iter': 10000,
            'random_state': seed_no
        },
        'DecisionTree': {
            # default
            # 'max_depth': 100, 'max_features': 'auto',
            # best
            'class_weight': {0: 1, 1: 3}, 'max_depth': 50, 'max_features': 8, 'min_samples_leaf': 10,
            'min_samples_split': 10, 'splitter': 'best',
            'random_state': seed_no,
        },
        'RandomForest': {
            # default
            # 'n_estimators': 300, 'max_depth': 100, 'oob_score': True, 'bootstrap': True,
            # best
            'class_weight': {0: 1, 1: 5}, 'criterion': 'gini', 'max_depth': 1700, 'max_features': 'sqrt',
            'min_samples_split': 8, 'n_estimators': 50,
            'random_state': seed_no, 'n_jobs': n_jobs,  # 'oob_score': True,
        },
        'ExtraTrees': {
            # default
            # 'n_estimators': 300, 'max_depth': 100,
            # best
            'class_weight': {0: 1, 1: 3}, 'criterion': 'entropy', 'max_depth': 1200, 'max_features': 'sqrt',
            'min_samples_split': 8, 'n_estimators': 10,
            'random_state': seed_no, 'n_jobs': n_jobs
        },
        'XGBoost': {
            # default
            # 'n_estimators': 3000,
            # best
            'max_delta_step': 2, 'max_depth': 5, 'n_estimators': 15, 'subsample': 0.5,
            'seed': seed_no, 'nthread': n_jobs
        },
        'MLP': {
            # default
            # 'tol': 0.0001, 'learning_rate_init': 0.06794912926673598, 'max_iter': 1000, 'activation': 'logistic',
            # 'solver': 'lbfgs',
            # best
            'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'learning_rate_init': 0.05, 'max_iter': 10000,
            'solver': 'sgd', 'tol': 0.0001,
            'random_state': seed_no,
        },
    }

    sim_opt_params = {
        'latin': {
            # Only latin dataset 100k lines
            'damerau_levenshtein': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.5, 0.2, 0.3]]},
            'jaro': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.7, 0.1, 0.2]]},
            'jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
            # 'permuted_winkler': [],
            # 'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.9, [0.4, 0.2, 0.4]]},
            'jaccard': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.9, [0.3, 0.3, 0.4]]},
            'strike_a_match': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.9, [0.5, 0.1, 0.4]]},
            'skipgram': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.9, [0.3, 0.3, 0.4]]},
            'monge_elkan': {'simple': [0.6, [0.7, 0.2, 0.1]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'soft_jaccard': {'simple': [0.8, [0.6, 0.1, 0.3]], 'avg': [0.9, [0.5, 0.1, 0.4]]},
            'davies': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'tuned_jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'tuned_jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
        },
        'global': {
            'damerau_levenshtein': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            # 'permuted_winkler': [],
            # 'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'strike_a_match': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.65, [0.4, 0.5, 0.1]]},
            'skipgram': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'monge_elkan': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'soft_jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'davies': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'tuned_jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'tuned_jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        }
    }

    # These parameters constitute the search space for GridSearchCV in our experiments.
    SVM_hyperparameters = [
        {
            'kernel': ['rbf', 'sigmoid'],
            'gamma': [1e-2, 1e-3, 1, 5, 10, 'scale'],
            'C': [0.01, 0.1, 1, 10, 25, 50, 100, 300],
            'max_iter': [10000],
            'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
        },
        {
            'kernel': ['poly'],
            'degree': [1, 2, 3],
            'gamma': ['scale', 'auto'],
            'C': [0.01, 0.1, 1, 10, 25, 50, 100],
            'max_iter': [30000],
            'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
        },
    ]
    DecisionTree_hyperparameters = {
        'max_depth': [2, 3, 5, 10, 30, 50, 60, 80, 100],
        'min_samples_split': [2, 5, 10, 20, 50, 100],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': list(np.arange(2, 11, 2)) + ["sqrt", "log2"],
        'splitter': ['best', 'random'],
        'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
    }
    RandomForest_hyperparameters = {
        # 'bootstrap': [True, False],
        'max_depth': [500, 1000, 1200, 1500, 1700, 1800, 2000],
        "n_estimators": [20, 30, 50, 80, 100, 250, 500],
        'criterion': ['gini', 'entropy'],
        'max_features': ['log2', 'sqrt'],  # auto is equal to sqrt
        # 'min_samples_leaf': [1, 2, 4, 10],
        'min_samples_split': [3, 5, 6, 8, 10],
        'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
    }
    XGBoost_hyperparameters = {
        # "n_estimators": [50, 70, 100, 500, 1000, 3000],
        # 'max_depth': [3, 5, 10, 30, 50, 70, 100],
        "n_estimators": [5, 10, 15, 20, 50, 70, 100],
        'max_depth': [2, 3, 5, 7, 8, 10, 20, 30],
        # hyperparameters to avoid overfitting
        # 'eta': list(np.linspace(0.01, 0.2, 10)),  # 'learning_rate'
        # 'gamma': [0, 1, 5],
        'subsample': [0.4, 0.5, 0.6, 0.7],
        # # Values from 0.3 to 0.8 if you have many columns (especially if you did one-hot encoding),
        # # or 0.8 to 1 if you only have a few columns
        # 'colsample_bytree': list(np.linspace(0.8, 1, 3)),
        # 'min_child_weight': [1, 5, 10],
        # 'scale_pos_weight': [1, 2, 3, 5],
        'max_delta_step': [1, 2, 3, 5],
    }
    MLP_hyperparameters = {
        'hidden_layer_sizes': [(100,), (50, 50,)],
        'learning_rate_init': [0.005, 0.01, 0.05, 0.1],
        'max_iter': [10000],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'tol': [1e-3, 1e-4],
    }

    # These parameters constitute the search space for RandomizedSearchCV in our experiments.
    SVM_hyperparameters_dist = {
        'C': expon(scale=100), 'gamma': expon(scale=.1),
        'kernel': ['rbf'],
        'class_weight': ['balanced'],
        'max_iter': [10000]
    }
    DecisionTree_hyperparameters_dist = {
        'max_depth': sp_randint(10, 200),
        'min_samples_split': sp_randint(2, 200),
        'min_samples_leaf': sp_randint(1, 10),
        'max_features': sp_randint(1, 11),
        'class_weight': ['balanced'] + [{0: 1, 1: w} for w in range(1, 5)],
    }
    RandomForest_hyperparameters_dist = {
        # 'bootstrap': [True, False],
        'max_depth': sp_randint(3, 200),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 10),
        'min_samples_split': sp_randint(2, 30),
        "n_estimators": sp_randint(200, 1000),
        'class_weight': ['balanced'] + [{0: 1, 1: w} for w in range(1, 5)],
    }
    XGBoost_hyperparameters_dist = {
        "n_estimators": sp_randint(50, 4000),
        'max_depth': sp_randint(3, 200),
        # 'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        # hyperparameters to avoid overfitting
        'gamma': sp_randint(0, 5),
        # 'subsample': truncnorm(0.7, 1),
        'subsample': truncnorm(0.4, 0.7),
        # 'colsample_bytree': truncnorm(0.8, 1),
        # 'min_child_weight': sp_randint(1, 10),
        # 'scale_pos_weight': sp_randint(1, 5),
        'max_delta_step': sp_randint(1, 5),
        "reg_alpha": truncnorm(0, 2),
        'reg_lambda': sp_randint(1, 20),
    }
    MLP_hyperparameters_dist = {
        'hidden_layer_sizes': [(100,), (50, 50,)],
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': [10000],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'tol': [1e-3, 1e-4],
    }
