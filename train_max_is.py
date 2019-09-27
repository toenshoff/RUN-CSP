from model import Max_IS_Network
from csp_utils import CSP_Instance, is_language

import data_utils
import argparse
import os
import random
from tqdm import tqdm


def train(network, train_data, t_max, epochs):
    '''
    Trains an Independent Set Network on the given data
    :param network: The Max_IS_Network instance
    :param train_data: A list of CSP instances that are used for training
    :param t_max: Number of RUN_CSP iterations on each instance
    :param epochs: Number of training epochs
    '''

    # only save network state when conflicts are below this threshold
    conflict_threshold = 0.02

    best_is_ratio = 0.0
    for e in range(epochs):
        print('Epoch: {}'.format(e))

        # train one epoch
        output_dict = network.train(train_data, iterations=t_max)

        # Get average percentage of conflicting edges and relative size of independent set
        conflict_ratio = output_dict['conflict_ratio']
        is_ratio = output_dict['is_ratio']
        print(f'Training Conflict Probability: {conflict_ratio}, IS Ratio {is_ratio}')

        # if network improved, save new model
        if conflict_ratio < conflict_threshold and is_ratio > best_is_ratio:
            network.save_checkpoint('best')
            best_is_ratio = is_ratio
            

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
        instances = [CSP_Instance.graph_to_csp_instance(g, is_language, 'NAND') for g in graphs]
    else:
        print(f'Generating {args.n_instances} training instances')
        instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, is_language) for _ in tqdm(range(args.n_instances))]
        
    # combine instances into batches
    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)

    # construct new network
    network = Max_IS_Network(args.model_dir)
    train(network, train_batches, t_max=args.t_max, epochs=args.epochs)


if __name__ == '__main__':
    main()
