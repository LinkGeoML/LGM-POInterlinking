|MIT|

==================
LGM-POInterlinking
==================
This Python code implements and evaluates the proposed LinkGeoML models for POI classification-based interlinking.

In this setting, we consider several of the attributes connected to each spatio-textual entity to decide whether two
POIs refer to the same physical spatial entity. Specifically, we utilize the following attributes which can be
categorized as: (i) spatial, i.e., its coordinates, (ii) textual, i.e., its name and address name and (iii) number,
i.e., its address number. For the textual ones, we utilize and adapt the meta-similarity function, called **LGM-Sim**,
that was developed for the `LGM-Interlinking <https://github.com/LinkGeoML/LGM-Interlinking.git>`__ problem.
Consequently, we utilize the available richer spatial information to derive a set of informative training features to
capture domain specificities and, thus, more accurately describe each POI. The proposed method and its derived features
are used in various classification models to address the POI interlinking problem. Additionally, we perform a full machine
learning workflow that involves the grid-search and cross-validation functionality, based on the `scikit-learn <https
://scikit-learn.org/>`_ toolkit, to optimize and optimally fit each examined model to the data at hand.

The source code was tested using Python 3 (>=3.7) and Scikit-Learn 0.23.1 on a Linux server.

Setup procedure
---------------
Download the latest version from the `GitHub repository <https://github.com/LinkGeoML/LGM-POInterlinking.git>`_, change to
the main directory and run:

.. code-block:: bash

   pip install -r pip_requirements.txt

It should install all the required libraries automatically (*scikit-learn, numpy, pandas etc.*).

Usage
------
The input dataset need to be in CSV format. Specifically, a valid dataset should have at least the following
fields/columns:

* The names for each of the candidate POI pairs.
* The addresses for each of the candidate POI pairs.
* The coordinates, i.e., latitude and longitude, for each of the candidate POI pairs.
* The label, i.e., {True, False}, assigned to each POI pair.

The library implements the following distinct processes:

#. Features extraction
    The `build <https://linkgeoml.github.io/POI-Interlinking/process.html#poi_interlinking.processing.features.Features>`_
    function constructs a set of training features to use within classifiers for toponym interlinking.

#. Algorithm and model selection
    The functionality of the
    `fineTuneClassifiers <https://linkgeoml.github.io/POI-Interlinking/learning.html#poi_interlinking.learning.hyperparam_tuning.
    ParamTuning.fineTuneClassifiers>`_ function is twofold.
    Firstly, it chooses among a list of supported machine learning algorithms the one that achieves the highest average
    accuracy score on the examined dataset. Secondly, it searches for the best model, i.e., the optimal hyper-parameters
    for the best identified algorithm in the first step.

#. Model training
    The `trainClassifier <https://linkgeoml.github.io/POI-Interlinking/learning.html#poi_interlinking.learning.hyperparam_tuning.
    ParamTuning.trainClassifier>`_ trains the best selected model on previous
    process, i.e., an ML algorithm with tuned hyperparameters that best fits data, on the whole train dataset, without
    splitting it in folds.

#. Model deployment
    The `testClassifier <https://linkgeoml.github.io/POI-Interlinking/learning.html#poi_interlinking.learning.hyperparam_tuning.
    ParamTuning.testClassifier>`_ applies the trained model on new untested data.

A complete pipeline of the above processes, i.e., features extraction, training and evaluating state-of-the-art
classifiers, for toponym interlinking can be executed with the following command:

.. code-block:: bash

    $ python -m poi_interlinking.cli tune --train_set <path/to/train-dataset> --test_set <path/to/test-dataset>

Additionally, *help* is available on the command line interface (*CLI*). Enter the following to list all supported
commands or options for a given command with a short description.

.. code-block:: bash

    $ python -m poi_interlinking.cli -h
    Usage: cli.py [OPTIONS] COMMAND [ARGS]...

    Options:
      -h, --help  Show this message and exit.

    Commands:
      evaluate                evaluate the effectiveness of the proposed methods
      extract_frequent_terms  create a file with ranked frequent terms found in corpus
      hyperparam_tuning       tune various classifiers and select the best hyper-parameters on a train dataset
      learn_sim_params        learn parameters, i.e., weights/thresholds, on a train dataset for similarity metrics

Documentation
-------------
Source code documentation is available from `linkgeoml.github.io`__.

__ https://linkgeoml.github.io/LGM-POInterlinking/

License
-------
POI-Interlinking is available under the `MIT <https://opensource.org/licenses/MIT>`_ License.

..
    .. |Documentation Status| image:: https://readthedocs.org/projects/coala/badge/?version=latest
       :target: https://linkgeoml.github.io/POI-Interlinking/

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
