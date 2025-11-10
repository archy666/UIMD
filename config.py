import argparse
import os

def get_args():
    root = os.getcwd() + '/datasets/'
    parser = argparse.ArgumentParser(description='hyper-parameter in deterministic IBCI model')
    parser.add_argument('--missing_rate', type=float, nargs='+', default=[0.1], help='missing rate (default = 0.5)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default=0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma parameter (default=0.1)')
    parser.add_argument('--beta', type=float, nargs='+', default=[0.001], help='beta parameters (default=[0.001])')
    parser.add_argument('--lamb', type=float, nargs='+', default=[1], help='lambda parameters (default=[1])')
    parser.add_argument('--rho', type=float, nargs='+', default=[1], help='rho parameters (default=[1])')
    parser.add_argument('--scheduler', type=bool, default=True, help='use scheduler or not (default=True)')
    parser.add_argument('--step_size', type=int, default=200, help='step_size for scheduler (default=200)')
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size for training (default: 32)')
    parser.add_argument('--loss_type', type=str, default='loss_IMCI_hsic', help='loss function type (default=loss_IMCI_hsic)')
    parser.add_argument('--data_path', type=str, default=root+'/cub_googlenet_doc2vec_c10.mat', help='data path')
    return parser.parse_args()