# coding= utf-8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import os
import re
from text_unidecode import unidecode
import unicodedata
import __main__
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pycountry
from langdetect import detect, lang_detect_exception

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


class StaticValues:
    # sim_features_cols = [
    #     "Damerau-Levenshtein",
    #     "Jaro",
    #     "Jaro-Winkler",
    #     "Jaro-Winkler reversed",
    #     "Sorted Jaro-Winkler",
    #     # "Permuted Jaro-Winkler",
    #     "Cosine N-grams",
    #     "Jaccard N-grams",
    #     "Dice bigrams",
    #     "Jaccard skipgrams",
    #     "Monge-Elkan",
    #     "Soft-Jaccard",
    #     "Davis and De Salles",
    #     "Damerau-Levenshtein Sorted",
    #     "Jaro Sorted",
    #     "Jaro-Winkler Sorted",
    #     "Jaro-Winkler reversed Sorted",
    #     # "Sorted Jaro-Winkler Sorted",
    #     # "Permuted Jaro-Winkler Sorted",
    #     "Cosine N-grams Sorted",
    #     "Jaccard N-grams Sorted",
    #     "Dice bigrams Sorted",
    #     "Jaccard skipgrams Sorted",
    #     "Monge-Elkan Sorted",
    #     "Soft-Jaccard Sorted",
    #     "Davis and De Salles Sorted",
    #     "LinkGeoML Jaro-Winkler",
    #     "LinkGeoML Jaro-Winkler reversed",
    #     # "LSimilarity",
    #     "LSimilarity_wavg",
    #     # "LSimilarity_davies",
    #     # "LSimilarity_skipgram",
    #     # "LSimilarity_soft_jaccard",
    #     # "LSimilarity_strike_a_match",
    #     # "LSimilarity_cosine",
    #     # "LSimilarity_monge_elkan",
    #     # "LSimilarity_jaro_winkler",
    #     # "LSimilarity_jaro",
    #     # "LSimilarity_jaro_winkler_reversed",
    #     "LSimilarity_davies_wavg",
    #     "LSimilarity_skipgram_wavg",
    #     "LSimilarity_soft_jaccard_wavg",
    #     "LSimilarity_strike_a_match_wavg",
    #     "LSimilarity_cosine_wavg",
    #     "LSimilarity_jaccard_wavg",
    #     "LSimilarity_monge_elkan_wavg",
    #     "LSimilarity_jaro_winkler_wavg",
    #     "LSimilarity_jaro_wavg",
    #     "LSimilarity_jaro_winkler_reversed_wavg",
    #     "LSimilarity_l_jaro_winkler_wavg",
    #     "LSimilarity_l_jaro_winkler_reversed_wavg",
    #     # "LSimilarity_baseScore",
    #     # "LSimilarity_mismatchScore",
    #     # "LSimilarity_specialScore",
    #     "Avg LSimilarity_baseScore",
    #     "Avg LSimilarity_mismatchScore",
    #     "Avg LSimilarity_specialScore",
    #     # non metric features
    #     # "contains_str1",
    #     # "contains_str2",
    #     # "WordsNo_str1",
    #     # "WordsNo_str2",
    #     # "dashed_str1",
    #     # "dashed_str2",
    #     # "hasFreqTerm_str1",
    #     # "hasFreqTerm_str2",
    #     # "posOfHigherSim_str1_start",
    #     # "posOfHigherSim_str1_middle",
    #     # "posOfHigherSim_str1_end",
    #     # "posOfHigherSim_str2_start",
    #     # "posOfHigherSim_str2_middle",
    #     # "posOfHigherSim_str2_end",
    # ]

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

            self.final_cols += self.spatial_feature_cols
            if config.all_cols: self.final_cols += self.extra_feature_cols
        else:  # basic or whatever
            self.final_cols = self.address_feature_cols + self.sim_features_cols + self.spatial_feature_cols
            if config.all_cols: self.final_cols += self.extra_feature_cols
