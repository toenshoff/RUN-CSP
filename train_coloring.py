from model import Coloring_Network
from csp_utils import CSP_Instance, coloring_language
from train import train

import data_utils

import argparse
import os
import random
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('-t', '--t_max', type=int, default=25, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-m', '--model_dir', type=str, help='Model directory in which the trained model is stored')
    parser.add_argument('-d', '--data_path', default=None, help='A path to a training set of graphs in the NetworkX adj_list format. If left unspecified, random instances are used.')
    parser.add_argument('-v', '--n_variables', type=int, default=100, help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-c', '--n_clauses', type=int, default=300, help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=5000, help='Number of instances for training. Only used when --data_path is not specified.')
    args = parser.parse_args()

    if args.data_path is not None:
        print('loading graphs...')
        graphs = data_utils.load_graphs(os.path.join(args.data_path, '**'))
        random.shuffle(graphs)
        print('Converting graphs to CSP Instances')
        instances = [CSP_Instance.graph_to_csp_instance(g, coloring_language, 'NEQ') for g in graphs]
    else:
        print(f'Generating {args.n_instances} training instances')
        instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, coloring_language) for _ in tqdm(range(args.n_instances))]
        
    # combine instances into batches
    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)

    # construct and train new network
    network = Coloring_Network(args.model_dir)
    train(network, train_batches, epochs=args.epochs, t_max=args.t_max)


if __name__ == '__main__':
    main()
