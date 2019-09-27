from model import Max_IS_Network
from csp_utils import CSP_Instance, is_language

import data_utils
import argparse
import os
import numpy as np
from tqdm import tqdm


def evaluate_boosted(network, eval_instances, t_max, attempts=64):
    """
    Evaluate Independent Set Network with boosted predictions
    :param network: A Max_IS_Network
    :param eval_instances: A list of CSP instances for evaluation
    :param t_max: Number of RUN_CSP iterations on each instance
    :param attempts: Number of parallel attempts for each instance
    """

    conflict_ratios = []
    is_sizes = []
    for i, instance in enumerate(eval_instances):

        # get boosted and corrected predictions
        output_dict = network.predict_boosted_and_corrected(instance, iterations=t_max, attempts=attempts)
        
        conflicts = output_dict['conflicts']
        conflict_ratio = output_dict['conflict_ratio']
        conflict_ratios.append(conflict_ratio)
        is_size = output_dict['is_size']
        is_sizes.append(is_size)
        print(f'Conflicts for instance {i}: {conflicts}, IS Size: {is_size}')

    mean_conflict_ratio = np.mean(conflict_ratios)
    mean_is_size = np.mean(is_sizes)
    print(f'Mean ratio of conflicting edges: {mean_conflict_ratio}, Mean Corrected Independent Set size: {mean_is_size}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='The model directory of a trained network')
    parser.add_argument('-t', '--t_max', type=int, default=40, help='Iterations to perform on each graph')
    parser.add_argument('-a', '--attempts', type=int, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='A path to a training set of graphs in the NetworkX adj_list format. If left unspecified, random instances are used.')
    parser.add_argument('-v', '--n_variables', type=int, default=400, help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-c', '--n_clauses', type=int, default=1000, help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=5000, help='Number of instances for training. Only used when --data_path is not specified.')
    args = parser.parse_args()

    if args.data_path is not None:
        graphs = data_utils.load_graphs(os.path.join(args.data_path, '**'))
        instances = [CSP_Instance.graph_to_csp_instance(g, is_language, 'NAND') for g in graphs]
    else:
        print(f'Generating {args.n_instances} training instances')
        instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, is_language) for _ in tqdm(range(args.n_instances))]

    network = Max_IS_Network.load(args.model_dir)
    evaluate_boosted(network, instances, args.t_max, args.attempts)


if __name__ == '__main__':
    main()
