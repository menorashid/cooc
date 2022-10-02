import os

import argparse
import numpy
import omegaconf
import pandas

import dataset_prepper
import cooc


def main(config):
    dp = dataset_prepper.DatasetPrepper(config.dataset)
    dataset = dp.get_ows_dataset(config.cooc.ows)

    pain_level_field = config.dataset.fields.pain_level
    pain_dataset = dataset.loc[dataset[pain_level_field].isin(config.cooc.pain_level)]
    no_pain_dataset = dataset.loc[dataset[pain_level_field].isin(config.cooc.no_pain_level)]

    pain_graph = cooc.get_normalized_adjacency_matrix(pain_dataset, dp.header_aus)
    no_pain_graph = cooc.get_normalized_adjacency_matrix(no_pain_dataset, dp.header_aus)
    selected_aus = cooc.select_aus(pain_graph, no_pain_graph, config.cooc.alpha, dp.header_aus)

    p_values = cooc.get_p_values(
        no_pain_dataset, pain_dataset, config.dataset.fields.subject, dp.header_aus
    )
    selected_p_values = p_values[numpy.in1d(dp.header_aus, selected_aus)]

    os.makedirs(config.out_dir, exist_ok=True)
    out_file = os.path.join(config.out_dir, 'aus_and_p_values.csv')
    pandas.DataFrame({'codes': selected_aus, 'p-values': selected_p_values}).to_csv(out_file)

    print(
        f'Pain Levels: {config.cooc.pain_level}. No Pain Levels: {config.cooc.no_pain_level}\n'
        f'Selected Codes: {selected_aus}\n'
        f'P-values: {selected_p_values}\n'
        f'Output written to: {out_file}\n'
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file path', type=str, required=True)
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    main(config)
