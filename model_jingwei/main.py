import faiss
import numpy as np
from exp.exp_OWL import Exp_OWL
from utils.data_loader import get_loader_in, get_loader_out
import argparse

from utils.args_loader import get_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args()
    exp = Exp_OWL(args)  # set experiments
    print('>>>>>>>start feature extraction on in-distribution data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
        args.in_dataset))
    # exp.id_feature_extract()
    exp.id_feature_extract(exp.model, args.in_dataset)

    print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.out_datasets))
    exp.ns_feature_extract('SVHN')
    loader_out = get_loader_out(args, dataset=(None, 'SVHN'), split=('val'))
    val_loader_out = loader_out.val_ood_loader  # take the val/test batch of the ood data
    exp.ns_feature_extract(exp.model, val_loader_out, 'SVHN')

    print(
        '>>>>>>>start ood detection on new-coming data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.out_datasets))
    unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels = exp.ood_detection('SVHN', K=50)
    # print('>>>>>>>start ood detection on new-coming data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.out_datasets))
    # unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels = exp.ood_detection('SVHN', K=50)

    # print(f'Total new samples: {len(bool_ood)} \nNumber of correctly detected ood samples: {len(unknown_idx)}')

    print(
        '>>>>>>>start incremental learning on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.out_datasets))
    ood_class = [0, 1]  # select two classes in ood data as unrecognized/new classes
    n_ood = 50  # take 50 ood samples
    exp.train_global('SVHN', True, ood_class, n_ood)

    print(f'Total new samples: {len(bool_ood)} \nNumber of correctly detected ood samples: {len(unknown_idx)}')
    torch.cuda.empty_cache()
