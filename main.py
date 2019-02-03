#!/usr/bin/env python3

import os
import os.path
import pathlib
import sys
import types
import warnings

import funcy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.common import SettingWithCopyWarning

from explorer import UnsignedS3PipelineExplorer


sns.set(style='white')

root = '/tmp'
outputdir = os.path.join(root, 'output')
datadir = os.path.join(root, 'data')

warnings.simplefilter('ignore', SettingWithCopyWarning)
warnings.simplefilter('ignore', FutureWarning)



# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

test_id_start = '20181024200501872083'
bucket = 'ml-pipelines-2018'
ex = UnsignedS3PipelineExplorer(bucket)


@funcy.memoize
def _load_pipelines_df(ex=None):
    """Get all pipelines, passing the analysis-specific test_id filter"""
    path = pathlib.Path(datadir, 'cache', 'pipelines.pkl.gz')
    if not path.exists():
        if ex is None:
            raise ValueError
        path.parent.mkdir(parents=True, exist_ok=True)
        df = ex.get_pipelines()
        df.to_pickle(path, compression='gzip')
    else:
        df = pd.read_pickle(path)

    assert df['test_id'].min() >= test_id_start

    return df


def _load_baselines_df():
    df = pd.read_table(os.path.join(datadir, 'baselines.tsv'))
    df['problem'] = df['problem'].str.replace('_problem', '')
    df = df.set_index('problem')
    _add_tscores(df, score_name='baselinescore')
    return df


# ------------------------------------------------------------------------------
# Saving results
# ------------------------------------------------------------------------------

def _savefig(fig, name, figdir=outputdir):
    for ext in ['.png', '.pdf', '.eps']:
        fig.savefig(figdir + '/' + name + ext,
                    bbox_inches='tight', pad_inches=0)

# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------

_METRIC_TYPES = {
    'f1': 'zero_one_score',
    'f1Macro': 'zero_one_score',
    'accuracy': 'zero_one_score',
    'normalizedMutualInformation': 'zero_one_score',
    'meanSquaredError': 'zero_inf_cost',
    'meanAbsoluteError': 'zero_inf_cost',
    'rootMeanSquaredError': 'zero_inf_cost',
}


def _normalize(metric_type, min_value=None, max_value=None):
    def f(raw):
        if metric_type == 'zero_one_score':
            return raw
        elif metric_type == 'zero_one_cost':
            return 1 - raw
        elif metric_type == 'ranged_score':
            return (raw - min_value) / (max_value - min_value)
        elif metric_type == 'real_score':
            return 1 / (1 + np.exp(-raw))
        elif metric_type == 'real_cost':
            return 1 - (1 / (1 + np.exp(-raw)))
        elif metric_type == 'zero_inf_score':
            return 1 / (1 + np.exp(-np.log10(raw)))
        elif metric_type == 'zero_inf_cost':
            return 1 - 1 / (1 + np.exp(-np.log10(raw)))
        else:
            raise ValueError('Unknown metric type')

    return f


def _normalize_df(s, score_name='cv_score'):
    return _normalize(_METRIC_TYPES[s['metric']])(s[score_name])


def _add_tscores(df, score_name='score'):
    if 't-score' not in df:
        df['t-score'] = df.apply(_normalize_df, score_name=score_name, axis=1)


@funcy.memoize
def _get_tuning_results_df():
    df = _load_pipelines_df()

    def default_score(group):
        return group.sort_values(by='ts', ascending=True)['score'].iloc[0]

    def min_max_score(group):
        return group['score'].agg(['min', 'max'])

    default_scores = (
        df
        .groupby(['dataset', 'name'])
        .apply(default_score)
        .to_frame('default_score')
        .reset_index()
        .rename(columns = {'name': 'template'})
        [lambda _df: ~_df['template'].str.contains('trivial')]
        .groupby('dataset')
        ['default_score']
        .mean()
        .to_frame('default_score')
    )

    min_max_scores = (
        df
        .groupby('dataset')
        .apply(min_max_score)
        .rename(columns = {'min': 'min_score', 'max': 'max_score'})
    )

    sds = (
        df
        .groupby('dataset')
        ['score']
        .std()
        .to_frame('sd')
    )

    # adjust for error vs reward-style metrics (make errors negative)
    # adjustment == -1 if the metric is an Error (lower is better)
    adjustments = (
        df
        .groupby('dataset')
        ['metric']
        .first()
        .str
        .contains('Error')
        .map({True: -1, False: 1})
        .to_frame('adjustment')
    )

    data = min_max_scores.join(default_scores).join(sds).join(adjustments)

    # compute best score, adjusting for min/max
    data['best_score'] = data['max_score']
    mask = data['adjustment'] == -1
    data.loc[mask, 'best_score'] = data.loc[mask, 'min_score']

    data['delta'] = data.eval(
        expr='adjustment * (best_score - default_score) / sd')

    return data


@funcy.memoize
def _get_test_results_df():
    results_df = ex.get_test_results()
    results_df = results_df.query('test_id >= @test_id_start')
    _add_tscores(results_df, score_name='cv_score')
    return results_df


# ------------------------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------------------------

def make_table_1():
    df = _load_pipelines_df(ex=ex)
    datasets = df['dataset'].unique()

    _all_datasets = ex.get_datasets()
    _all_datasets['dataset_id'] = _all_datasets['dataset'].apply(
        ex.get_dataset_id)

    datasets_df = pd.merge(
        pd.DataFrame(datasets, columns=['dataset_id']),
        _all_datasets,
        left_on='dataset_id',
        right_on='dataset_id',
    )
    assert datasets_df.shape[0] == len(datasets)

    modality_type_count = datasets_df.groupby(
        ['data_modality', 'task_type']).size().to_frame('Tasks')
    assert modality_type_count['Tasks'].sum() == 431

    result = (
        modality_type_count
        .rename(columns={'data_modality': 'Data Modality',
                         'task_type': 'Problem Type'})
        .sort_index()
    )

    result.to_latex(os.path.join(outputdir, 'table1.tex'))


def make_figure_6():
    baselines_df = _load_baselines_df()

    problems = list(baselines_df.index)
    best_pipelines = [ex.get_best_pipeline(problem) for problem in problems]
    mlz_pipelines_df = pd.DataFrame.from_records(
        [
            pipeline.to_dict()
            for pipeline in best_pipelines
            if pipeline is not None
        ]
    )
    mlz_pipelines_df = mlz_pipelines_df.query('test_id >= @test_id_start')
    mlz_pipelines_df['problem'] = mlz_pipelines_df['dataset'].str.replace(
        '_dataset_TRAIN', '')
    mlz_pipelines_df = mlz_pipelines_df.set_index('problem')
    _add_tscores(mlz_pipelines_df)

    combined_df = baselines_df.join(
        mlz_pipelines_df, lsuffix='_ll', rsuffix='_mlz')

    data = (
        combined_df[['t-score_ll', 't-score_mlz']]
        .dropna()
        .rename(columns={'t-score_ll': 'baseline',
                         't-score_mlz': 'ML Bazaar'})
        .sort_values('baseline')
        .stack()
        .to_frame('score')
        .reset_index()
        .rename(columns={'level_1': 'system'})
    )

    with sns.plotting_context('paper', font_scale=2.0):
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x='problem', y='score', hue='system', data=data)

        # remove x labels
        ax.set_xlabel('task')
        ax.set_xticklabels([])

        sns.despine()
        plt.tight_layout()
        ax.get_legend().set_title('')

        _savefig(fig, 'figure6', figdir=outputdir)
        plt.close(fig)

    # Compute performance vs human baseline (Section 5.3)
    result = (
        combined_df
        [['t-score_ll', 't-score_mlz']]
        .dropna()
        .apply(np.diff, axis=1)
        .agg(['mean', 'std'])
    )

    fn = os.path.join(outputdir, '5_3_performance_vs_baseline')
    result.to_csv(fn)



def make_figure_7():
    data = _get_tuning_results_df()
    delta = data['delta'].dropna()

    with sns.plotting_context('paper', font_scale=2.0):
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.distplot(delta)

        ax.set_xlabel('standard deviations')
        ax.set_ylabel('density')
        ax.set_xlim(left=0, right=5)

        sns.despine()
        plt.tight_layout()

        _savefig(fig, 'figure7', figdir=outputdir)
        plt.close(fig)


def compute_5_2_pipelines_node_second():
    test_results = ex.get_test_results()
    test_results = test_results.query('test_id >= @test_id_start')
    nodes = test_results['hostname'].nunique()
    n_pipelines = test_results['iterations'].sum()
    total_seconds_elapsed = (
                test_results['iterations'] * test_results['cv_time']).sum()
    average_seconds_elapsed = total_seconds_elapsed / nodes
    result = n_pipelines / average_seconds_elapsed / nodes

    fn = os.path.join(outputdir, '5_2_pipelines_node_second.txt')
    with open(fn, 'w') as f:
        f.write('{} pipelines/node/second'.format(result))


def compute_5_3_performance_vs_baseline():
    """Compute performance vs human baseline (Section 5.3)"""
    # see make_figure_6 for implementation
    pass


def compute_5_3_tuning_improvement_sds():
    """Compute average improvement during tuning, in sds"""
    data = _get_tuning_results_df()
    delta = data['delta'].dropna()
    result = delta.mean()

    fn = os.path.join(outputdir, '5_3_tuning_improvement_sds.txt')
    with open(fn, 'w') as f:
        f.write(
            '{} standard deviations of improvement during tuning'
            .format(result))


def compute_5_3_tuning_improvement_pct_of_tasks():
    """Compute pct of tasks that improve by >1sd during tuning"""
    data = _get_tuning_results_df()
    delta = data['delta'].dropna()
    result = 100 * (delta > 1.0).mean()

    fn = os.path.join(outputdir, '5_3_tuning_improvement_pct_of_tasks.txt')
    with open(fn, 'w') as f:
        f.write(
            '{:.2f}% of tasks improve by >1 standard deviation'
            .format(result))


def compute_5_4_npipelines():
    """Compute the total number of XGB/RF pipelines evaluated"""
    df = _load_pipelines_df()
    npipelines_rf = np.sum(df['pipeline'].str.contains('random_forest'))
    npipelines_xgb = np.sum(df['pipeline'].str.contains('xgb'))
    total = npipelines_rf + npipelines_xgb
    result = pd.DataFrame(
        [npipelines_rf, npipelines_xgb, total],
        index = ['RF', 'XGB', 'total'],
        columns = ['pipelines']
    )

    fn = os.path.join(outputdir, '5_4_npipelines.csv')
    result.to_csv(fn)


def compute_5_4_xgb_wins_pct():
    """Compute the pct of tasks for which XGB pipelines beat RF pipelines"""
    results_df = _get_test_results_df()

    rf_results_df = (
        results_df
        [lambda _df: _df['pipeline'].str.contains('random_forest').fillna(False)]
        .groupby('dataset')
        ['t-score']
        .max()
        .to_frame('RF')
    )

    xgb_results_df = (
        results_df
        [lambda _df: _df['pipeline'].str.contains('xgb').fillna(False)]
        .groupby('dataset')
        ['t-score']
        .max()
        .to_frame('XGB')
    )

    result = (
        rf_results_df
        .join(xgb_results_df)
        .fillna(0)
        .apply(np.argmax, axis=1)
        .value_counts()
        .to_frame('wins')
        .assign(percent = lambda _df: _df['wins'] / np.sum(_df['wins']))
    )

    fn = os.path.join(outputdir, '5_4_xgb_wins_pct.csv')
    result.to_csv(fn)


def compute_5_5_matern_wins_pct():
    """Compute the pct of tasks for which the best pipeline as tuned by GP-Matern52-EI beats the best pipeline as tuned by GP-SE-EI"""
    results_df = _get_test_results_df()

    gp_se_ei_results_df = (
        results_df
        [lambda _df: _df['tuner_type'].str.contains('gpei').fillna(False)]
        .groupby('dataset')
        ['t-score']
        .max()
        .to_frame('GP-SE-EI')
    )

    gp_matern52_ei_results_df = (
        results_df
        [lambda _df: _df['tuner_type'].str.contains('gpmatern52ei').fillna(False)]
        .groupby('dataset')
        ['t-score']
        .max()
        .to_frame('GP-Matern52-EI')
    )

    result = (
        gp_se_ei_results_df
        .join(gp_matern52_ei_results_df)
        .fillna(0)
        .apply(np.argmax, axis=1)
        .value_counts()
        .to_frame('wins')
        .assign(percent = lambda _df: _df['wins'] / np.sum(_df['wins']))
    )

    fn = os.path.join(outputdir, '5_5_matern_wins_pct.csv')
    result.to_csv(fn)


def main():
    """Call all of the results generating functions defined here"""
    this = sys.modules[__name__]
    names = set(dir(this)) - {'main'}
    for name in sorted(names):
        if not name.startswith('_'):
            obj = getattr(this, name)
            if isinstance(obj, types.FunctionType):
                print('Calling {}...'.format(name))
                obj()

    print('Done.')


if __name__ == '__main__':
    main()
