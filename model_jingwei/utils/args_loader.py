import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    # data
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='CIFAR-10 imagenet')
    parser.add_argument('--out-datasets', default=['SVHN', ], nargs="*", type=str, help="['SVHN', 'LSUN', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument('--name', default="resnet18-supcon", type=str, help='neural network name and training set')
    parser.add_argument('--model-arch', default='resnet18-supcon', type=str, help='model architecture')
    parser.add_argument('--save-path', default='./cache/', type=str, help="the cache for model's results")

    parser.add_argument('--p', default=0, type=float, help='sparsity level')
    parser.add_argument('--imagenet-root', default='./datasets/imagenet/', type=str, help='imagenet root')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    
    # model
    parser.add_argument('--method', default='', type=str, help='')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--epochs-clf', default=2, type=int, help='number of total epochs for classifiers')
    parser.add_argument('--epochs-ft', default=2, type=int, help='number of total epochs for fine-tuning')
    parser.add_argument('-b', '--batch-size', default=3000, type=int, help='mini-batch size')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save-epoch', default=10, type=int,
                        help='save the model every save_epoch')
    parser.add_argument('--cal-metric', help='calculatse metric directly', action='store_true')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
    parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int, help='depth of resnet')
    parser.add_argument('--width', default=4, type=int, help='width of resnet')
    parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
    parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
    
    # optimization
    parser.add_argument('--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        help='weight decay (default: 0.0001)')

    parser.set_defaults(argument=True)
    args = parser.parse_args()
    
    return args