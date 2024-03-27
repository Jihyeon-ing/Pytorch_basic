import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str, help='directory for saving results')
    parser.add_argument('--ex_name', default='ver_1', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default="./dataset")
    parser.add_argument('--dataname', default='./data')
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_n', default=10, type=int,nargs='*') 
    parser.add_argument('--hidden_n', default=64, type=int)
    parser.add_argument('--out_n', default=1, type=int)
  
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--is_train', default=True, type=bool)

    # Test parameters
    parser.add_argument('--test_epoch', default=100, type=int, help='Epoch of model which you test')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    if args.is_train:
        exp.train(args)
    else:
        exp.test(args)
