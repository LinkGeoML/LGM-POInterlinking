# coding= utf-8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import os
import re

import numpy as np
import pandas as pd
from text_unidecode import unidecode
import unicodedata
import __main__
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pycountry
from langdetect import detect, lang_detect_exception
import multiprocessing as mp

from poi_interlinking import config
from poi_interlinking.processing import sim_measures


punctuation_regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`ʿ,.-]')


def strip_accents(s):
    """Transliterate any unicode string into the closest possible representation in ascii text.

    Parameters
    ----------
    s : str
        Input string

    Returns
    -------
    str
        The transliterated string.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def ascii_transliteration_and_punctuation_strip(s):
    # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
    # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
    # equivalents.

    s = unidecode(strip_accents(s.lower()))
    s = punctuation_regex.sub('', s)
    return s


def transform(s1, s2, sorting=False, canonical=False, delimiter=' ', simple_sorting=False):
    """Perform normalization processes to input strings such as lowercasing, transliteration and punctuation/accentuation
    alignment.

    Parameters
    ----------
    s1: str
        The first string.
    s2: str
        The second string.
    sorting: bool
        A boolean flag whether to perform a custom mechanism of sorting or not. Specifically, an alphanumerical
        sorting applies only when the strings similarity is below the :attr:`~poi_interlinking.config.sort_thres`.
        If ``True`` and ``simple_sorting`` is ``False``, then perform the custom type of sorting.
    canonical: bool
        A boolean flag whether to perform canonical decomposition, i.e., translates each character into its decomposed
        form, and, afterwards, apply the compatibility decomposition, i.e. replace all compatibility characters with
        their equivalents.sorting or not.
    delimiter: str
        Character used to split s1 and s2.
    simple_sorting: bool
        If ``True`` apply alphanumeric sorting on s1 and s2.

    Returns
    -------
    s1, s2: str
        The transformed strings according to the selected parameters, e.g., `canonical`, `sorting` or `simple_sorting`.
    """
    # a = six.text_type(s1) #.lower()
    a = s1
    b = s2

    thres = config.sort_thres

    if canonical:
        a = ascii_transliteration_and_punctuation_strip(a)
        b = ascii_transliteration_and_punctuation_strip(b)

    if simple_sorting:
        a = " ".join(sorted_nicely(a.split(delimiter)))
        b = " ".join(sorted_nicely(b.split(delimiter)))
    elif sorting:
        tmp_a = a.replace(' ', '')
        tmp_b = b.replace(' ', '')

        if getattr(sim_measures, 'damerau_levenshtein')(tmp_a, tmp_b) < thres:
            a = " ".join(sorted_nicely(a.split(delimiter)))
            b = " ".join(sorted_nicely(b.split(delimiter)))
        elif getattr(sim_measures, 'damerau_levenshtein')(tmp_a, tmp_b) > \
                getattr(sim_measures, 'damerau_levenshtein')(a, b):
            a = tmp_a
            b = tmp_b

    return a, b


def sorted_nicely(l):
    """Sort the given iterable in the way that is expected.

    Parameters
    ----------
    l: :obj:`list` or :obj:`set` of str
        The iterable to be sorted.

    Returns
    --------
    :obj:`list`
        A sorted list of strs
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_langnm(s, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(s)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname


# Clean the string from stopwords based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(s, lang_detect=False):
    lname = get_langnm(s, lang_detect)
    stemmer = SnowballStemmer(lname)
    tokens = word_tokenize(s)
    # words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(lname))

    stopped_words = set(filter(lambda token: token in stopwords_set, tokens))
    filtered_words = list(filter(lambda token: token not in stopped_words, tokens))
    filtered_stemmed_words = list(map(lambda token: stemmer.stem(token), filtered_words))

    return filtered_words, filtered_stemmed_words, stopped_words


def getBasePath():
    return os.path.abspath(os.path.dirname(__main__.__file__))


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(*x)))


def parmap(f, X, nprocs=min(config.n_cores, mp.cpu_count() - 1)):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.progress_apply(func, **kwargs)


def apply_df_by_multiprocessing(df, func, workers=min(config.n_cores, mp.cpu_count() - 1), **kwargs):
    # workers = kwargs.pop('workers')
    pool = mp.Pool(processes=workers)
    result = pool.imap(_apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))])

    pool.close()
    pool.join()

    result = sorted(result, key=lambda x: x[0])
    return pd.concat([i[1] for i in result])


class Printing:
    cols = {
        'Method': 'Classifier',
        'Accuracy': 'Accuracy',
        'Precision': 'Precision',
        # 'Prec - weighted': 'Precision_weighted',
        'Recall': 'Recall',
        # 'Rec - weighted': 'Recall_weighted',
        'F1-Score': 'F1_score',
        'F1-std': 'F1_score_std',
        # 'F1-weighted': 'F1_score_weighted',
        'Roc-AUC': 'Roc_auc',
        # 'ROC-AUC-weighted': 'roc_auc_weighted',
        'Time (sec)': 'time'
    }


class StaticValues:

    sim_features_cols = [
        "Damerau_Levenshtein",
        "Jaro",
        "Jaro_Winkler",
        "Jaro_Winkler_reversed",
        "Jaro_Winkler_sorted",
        # "Permuted Jaro-Winkler",
        "Cosine_ngrams",
        "Jaccard_ngrams",
        "Dice_bigrams",
        "Jaccard_skipgrams",
        "Monge_Elkan",
        "Soft_Jaccard",
        "Davis_and_De_Salles",
        "Tuned_Jaro_Winkler",
        "Tuned_Jaro_Winkler_reversed",
    ]

    individual_lgm_feature_cols = [
        "LGM_baseScore",
        "LGM_mismatchScore",
        "LGM_specialScore",

    ]

    spatial_feature_cols = [
        'point_dist'
    ]

    address_feature_cols = [
        'street_numbers_diff',
    ] + [f'{x}_on_street_names' for x in sim_features_cols]

    extra_feature_cols = [
        'SimName',
        'SimAddress',
        'SimSemantic'
    ]

    final_cols = []

    sim_metrics = {
        'damerau_levenshtein': ['basic', 'sorted', 'lgm'],
        'jaro': ['basic', 'sorted', 'lgm'],
        'jaro_winkler': ['basic', 'sorted', 'lgm'],
        'jaro_winkler_reversed': ['basic', 'sorted', 'lgm'],
        'sorted_winkler': ['basic'],
        'permuted_winkler': [],
        'cosine': ['basic', 'sorted', 'lgm'],
        'jaccard': ['basic', 'sorted', 'lgm'],
        'strike_a_match': ['basic', 'sorted', 'lgm'],
        'skipgram': ['basic', 'sorted', 'lgm'],
        'monge_elkan': ['basic', 'sorted', 'lgm'],
        'soft_jaccard': ['basic', 'sorted', 'lgm'],
        'davies': ['basic', 'sorted', 'lgm'],
        'tuned_jaro_winkler': ['basic', 'sorted', 'lgm'],
        'tuned_jaro_winkler_reversed': ['basic', 'sorted', 'lgm'],
    }

    def __init__(self, sim_type='basic'):
        del self.final_cols[:]

        if sim_type == 'lgm':
            # self.final_cols += self.address_feature_cols + [
            #     f'{x}_{y}_on_street_names' for x in ['Sorted', 'LGM']
            #     for y in self.sim_features_cols if y.lower() != 'jaro_winkler_sorted'
            # ]
            # self.final_cols += [f'{x}_on_street_names' for x in self.individual_lgm_feature_cols]

            self.final_cols += self.address_feature_cols
            self.final_cols += self.sim_features_cols + [
                f'{x}_{y}' for x in ['Sorted', 'LGM']
                for y in self.sim_features_cols if y.lower() != 'jaro_winkler_sorted'
            ]
            self.final_cols += self.individual_lgm_feature_cols

            self.final_cols += self.spatial_feature_cols
            if config.all_cols: self.final_cols += self.extra_feature_cols
        elif sim_type == 'sorted':
            # self.final_cols += self.address_feature_cols + [
            #     f'{x}_{y}_on_street_names' for x in ['Sorted']
            #     for y in self.sim_features_cols if y.lower() != 'jaro_winkler_sorted'
            # ]

            self.final_cols += self.address_feature_cols
            self.final_cols += self.sim_features_cols + [
                f'{x}_{y}' for x in ['Sorted']
                for y in self.sim_features_cols if y.lower() != 'jaro_winkler_sorted'
            ]

            # self.final_cols += self.spatial_feature_cols
            if config.all_cols: self.final_cols += self.extra_feature_cols
        else:  # basic or whatever
            self.final_cols = self.address_feature_cols + self.sim_features_cols + self.spatial_feature_cols
            if config.all_cols: self.final_cols += self.extra_feature_cols
