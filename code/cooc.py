import itertools

import numpy
import scipy


def get_normalized_adjacency_matrix(ows_dataset, all_aus):
    adjacency_matrix = numpy.zeros((len(all_aus),len(all_aus)))
    features = ows_dataset[all_aus].values
    for instance in features:
        au_idx = numpy.where(instance > 0)[0]
        for au_i_idx, au_j_idx in itertools.combinations(au_idx, 2):
            adjacency_matrix[au_i_idx, au_j_idx] += 1
            adjacency_matrix[au_j_idx, au_i_idx] += 1
    num_instances_per_au = numpy.sum(features, axis=0, keepdims=True).T
    num_instances_per_au[num_instances_per_au == 0] = 1
    return adjacency_matrix / num_instances_per_au

def select_aus(pain_graph, no_pain_graph, alpha, all_aus):
    difference_graph = pain_graph - no_pain_graph
    difference_graph[difference_graph < 0] = 0
    au_importance = numpy.sum(difference_graph, axis=0)
    min_au_importance = numpy.min(au_importance[au_importance > 0])
    threshold = alpha * (numpy.max(au_importance) - min_au_importance) + min_au_importance
    return all_aus[au_importance >= threshold]

def get_p_values(pain_dataset, no_pain_dataset, subject_field_name, all_aus):
    unique_subjects = numpy.unique(pain_dataset[subject_field_name].values)

    pain_features = []
    no_pain_features = []
    for subject in unique_subjects:
        pain_values = pain_dataset.loc[pain_dataset[subject_field_name] == subject]
        pain_values = pain_values[all_aus].values

        no_pain_values = no_pain_dataset.loc[no_pain_dataset[subject_field_name] == subject]
        no_pain_values = no_pain_values[all_aus].values

        num_instances = min(pain_values.shape[0], no_pain_values.shape[0])
        pain_features.append(pain_values[:num_instances, :])
        no_pain_features.append(no_pain_values[:num_instances, :])

    pain_features = numpy.concatenate(pain_features)
    no_pain_features = numpy.concatenate(no_pain_features)
    p_values = []
    for au_idx in range(pain_features.shape[1]):
        p_values.append(
            scipy.stats.ttest_rel(pain_features[:, au_idx], no_pain_features[:, au_idx]).pvalue
        )
    return numpy.array(p_values)