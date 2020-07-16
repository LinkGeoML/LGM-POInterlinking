"""Command line interface for operation management"""
import click

from poi_interlinking.processing.spatial.osm_utilities import download_osm_polygons
from poi_interlinking.pre_processing import frequent_terms as ft
from poi_interlinking.learning import parameters as pm
from poi_interlinking import core


@click.group(context_settings=dict(max_content_width=120, help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command(help='download POIs into a defined bounding box over Overpass-turbo API')
@click.option('--bbox', default='38.02,23.77,38.07,23.84', show_default=True,
              help='Coordinates of the bounding box for an area in a comma-separated format as: lat1,lon1,lat2,lon2')
def download(bbox):
    click.echo('Downloading POIs with Overpass-turbo API')
    download_osm_polygons(bbox)


@cli.command('extract_frequent_terms', help='create a file with ranked frequent terms found in corpus')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to extract frequent terms from. '
                   'It requires once to run: python -m nltk.downloader \'punkt\'')
@click.option('--encoding', default='latin', show_default=True, type=click.Choice(['latin', 'global']),
              help='specify the alphabet encoding of toponyms in dataset.')
@click.option('--exp_path', help='Prefix to be used in naming the file with the extracted frequent terms.')
def freq_terms(train_set, encoding, exp_path):
    ft.extract_freqterms(train_set, encoding, exp_path)


@cli.command('learn_sim_params', help='learn parameters, i.e., weights/thresholds, on a train dataset for '
                                      'similarity metrics')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to learn parameters.')
@click.option('--sim_type', default='lgm', show_default=True, type=click.Choice(['basic', 'sorted', 'lgm']),
              help='Group of similarities to train.')
@click.option('--encoding', default='latin', show_default=True, type=click.Choice(['latin', 'global']),
              help='Specify the alphabet encoding of toponyms in dataset.')
def learn_params(train_set, sim_type, encoding):
    if sim_type == 'lgm':
        pm.learn_params_for_lgm(train_set, encoding)
    else: pm.learn_thres(train_set, sim_type)


@cli.command('tune', help='tune various classifiers and select the best hyper-parameters on a train dataset')
@click.option('--dataset', default='', help='the dataset to train/evaluate the models.')
@click.option('--encoding', default='latin', show_default=True, type=click.Choice(['latin', 'global']),
              help='Specify the alphabet encoding of toponyms in dataset.')
def hyperparams_learn(dataset, encoding):
    core.StrategyEvaluator(encoding).hyperparamTuning(dataset)


@cli.command('eval', help='evaluate the effectiveness of the proposed methods')
@click.option('--dataset', default='', help='the dataset to train/evaluate the models.')
@click.option('--train_set', default='manual_complete.csv', show_default=True, help='the dataset to train the models.')
@click.option('--test_set', default='auto_full_complete.csv', show_default=True,
              help='the dataset to evaluate the models.')
@click.option('--encoding', default='latin', show_default=True, type=click.Choice(['latin', 'global']),
              help='Specify the encoding of toponyms in dataset.')
@click.option('--is_build', is_flag=True, help='Whether loaded datasets contain raw data or already built features.')
def eval_classifiers(dataset, train_set, test_set, is_build, encoding):
    if train_set and test_set:
        core.StrategyEvaluator(encoding).evaluate_on_pre_split(train_set, test_set, is_build)
    else:
        core.StrategyEvaluator(encoding).evaluate(dataset, is_build)


cli.add_command(download)


if __name__ == '__main__':
    cli()
