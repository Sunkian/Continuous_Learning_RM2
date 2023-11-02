import os
import sys
import time
import warnings
import math
import random
import json
import faiss
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from seaborn import heatmap

from ..utils import metrics
from ..utils.data_loader import get_loader_in, get_loader_out
from ..utils.model_loader import get_model, get_classifier
from ..utils.losses import SupConLoss
from ..utils.util import AverageMeter
from ..utils.util import adjust_learning_rate, warmup_learning_rate, accuracy
from ..utils.util import set_optimizer
from ..models.resnet_supcon import SupConResNet, LinearClassifier
from ..exp.exp_OWLbasic import Exp_OWLbasic

warnings.filterwarnings('ignore')

np.random.seed(1)


class Exp_OWL(Exp_OWLbasic):
    def __init__(self, args):
        super(Exp_OWL, self).__init__(args)  ## init device

        # save path for checkpoints and classifiers
        # self.save_path = "./checkpoints/CIFAR-10/resnet18-supcon/"
        self.save_path = "/Users/apagnoux/Downloads/Continuous_Learning_RM2-master/model_jingwei/checkpoints/{}/{}/".format(
            self.args.in_dataset,
            self.args.model_arch
        )
        self.cache_path = self.args.save_path
        self.ft_model_path = self.save_path + "checkpoint_finetuned.pth.tar"
        self.init_classifier_path = self.save_path + "classifier_{flag}.pth.tar".format(flag='init')
        self.ft_classifier_path = self.save_path + "classifier_{flag}.pth.tar".format(flag='ft')

        # dataloaders
        self.loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
        self.trainloaderIn, self.testloaderIn, self.num_classes = self.loader_in_dict.train_loader, self.loader_in_dict.val_loader, self.loader_in_dict.num_classes
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        self.model = get_model(self.args, self.num_classes, load_ckpt=True)  # load pre-trained model
        self.classifier_init = get_classifier(self.args, self.num_classes, load_ckpt=False, classifier_flag='init')
        return self.model

    # def _select_optimizer(self):
    #     model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    #     return model_optim

    def id_feature_extract(self, model, id_name):

        """
            extract features from in-distribution samples

        :param model: the adopted model for feature extraction
        :param id_name: the ID dataset name, e.g., CIFAR-10
        #id_samples: (n, 3, 32, 32), a couple of new-coming normalized images

        :return None
        :save files: feat_log, score_log, label;
        """

        batch_size = self.args.batch_size
        dummy_input = torch.zeros((1, 3, 32, 32)).to(self.device)
        feat = model(dummy_input)
        featdims = feat.shape[1]

        begin = time.time()

        all_metadata = []

        # 'val'/'test' loader: to get the threshold distance in ID data and check if a sample is OOD
        for split, in_loader in [('train', self.trainloaderIn), ('val', self.testloaderIn), ]:
            # why testing data in in-distribution data?

            #  A ENLEVER, CHARGER DEPUIS LA BDD
            cache_name = f"{self.cache_path}/{id_name}_{split}_{self.args.name}.npz"
            if not os.path.exists(cache_name):
                # feat_log: the last layer features of ResNet
                # label: the ground truth labels
                feat_log = np.zeros((len(in_loader.dataset), featdims))
                # score_log = np.zeros((len(in_loader.dataset), self.num_classes))
                label = np.zeros(len(in_loader.dataset))

                model.eval()
                for batch_idx, (inputs, targets) in enumerate(in_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    start_ind = batch_idx * batch_size
                    end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))
                    out = model(inputs)
                    # score = self.classifier_init(out)

                    feat_log[start_ind:end_ind, :] = out.detach().cpu().numpy()
                    # score_log[start_ind:end_ind, :] = score.detach().cpu().numpy()
                    label[start_ind:end_ind] = targets.detach().cpu().numpy()


                    for idx in range(start_ind, end_ind):
                        metadata = {
                            'data_name': f'data_{idx}',  # Replace with actual data name if available
                            'dataset_split': split,
                            'feat_log': feat_log[idx].tolist(),
                            'label': int(label[idx]),
                            'repr_flag': None  # Or whatever logic you use to set this
                        }


                        # Send a POST request to the FastAPI route
                        # response = requests.post("http://127.0.0.1:8000/push_feature_data/", json=metadata)
                        response = requests.post("http://127.0.0.1:8000/update_feature_data/", json=metadata)

                        # if response.status_code == 200:
                        #     print(f"Pushed data for data_{idx} successfully!")
                        # else:
                        #     print(f"Failed to push data for data_{idx}!")
                        all_metadata.append(metadata)


                    if batch_idx % 100 == 0:
                        print(f"id batches: {batch_idx}/{len(in_loader)}")
                print("feature shape, feat_log: {}, label: {}".format(feat_log.shape, label.shape))

                # with open(f'/metadata_id_{split}.json', 'w') as json_file:  # The filename is now dynamic based on split
                #     json.dump(all_metadata, json_file)
                np.savez(cache_name, feat_log=feat_log, label=label)
                # np.savez(cache_name, feat_log = feat_log, score_log = score_log, label = label)
            else:
                print(f"Features for {id_name} already extracted and cached in {cache_name}")
                # data = np.load(cache_name, allow_pickle=True)
                # feat_log = data['feat_log']
                # score_log = data['score_log']
                # label = data['label']
                continue

        print(f"Time for Feature extraction over ID training/validation set: {time.time() - begin}")

        return None

    def ns_feature_extract(self, model, dataloader, ood_name):

        """
            extract features from new-coming samples

        # REMARKs
            - For testing purpose, the target groundtruths are retrieved and cached
            - For GUI, the target groundtruths should be labeled by users

        #:param ood_samples: (n, 3, 32, 32), a couple of new-coming normalized images
        :param model: the backbone model used for feature extraction
        :param dataloader: the dataloader for ood data
        :param ood_name: name of the ood dataset, e.g., "SVHN"

        :return ood_feat_log, ood_label
        :save files: feat_log; (score_log) is removed

        """

        batch_size = self.args.batch_size
        dummy_input = torch.zeros((1, 3, 32, 32)).to(self.device)
        feat = model(dummy_input)
        featdims = feat.shape[1]

        begin = time.time()

        cache_name = f"{self.cache_path}/{ood_name}vs{self.args.in_dataset}_{self.args.name}.npz"
        if not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(dataloader.dataset), featdims))
            ood_label = np.zeros(len(dataloader.dataset))

            model.eval()
            processed_samples = 0  # Add this before the for loop
            response = requests.get(f"http://127.0.0.1:8000/list_files/{ood_name}/")
            filenames_list = response.json().get("files", [])

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                current_filenames = filenames_list[processed_samples: processed_samples + len(inputs)]
                print('CURRENT_FILENAMES:', current_filenames)

                inputs = inputs.to(self.device)
                actual_batch_size = inputs.shape[0]

                start_ind = processed_samples
                end_ind = start_ind + actual_batch_size

                out = model(inputs)
                ood_feat_log[start_ind:end_ind, :] = out.detach().cpu().numpy()
                ood_label[start_ind:end_ind] = targets.detach().cpu().numpy()

                for idx, filename in enumerate(current_filenames):
                    update_data = {
                        'file_name': filename,
                        'ood_feat_log': ood_feat_log[start_ind + idx].tolist(),
                        'ood_label': int(ood_label[start_ind + idx])
                    }
                    response = requests.post("http://127.0.0.1:8000/update_ood_data/", json=update_data)
                    print(response.content)
                    if response.status_code == 200:
                        print(f"Updated data for {filename} successfully!")
                    else:
                        print(f"Failed to update data for {filename}!")

                processed_samples += actual_batch_size  # Update the total processed samples

                if batch_idx % 100 == 0:
                    print(f"ood batches: {batch_idx}/{len(dataloader)}")
            # save features and ground truth (if any)
            np.savez(cache_name, ood_feat_log=ood_feat_log, ood_label=ood_label)
        else:
            print(f"Features for {self.args.out_datasets} already extracted and cached in {cache_name}")
            data = np.load(cache_name, allow_pickle=True)
            ood_feat_log = data['ood_feat_log']
            ood_label = data['ood_label']

            # print(ood_feat_log)
            # print(ood_label)

        print(f"Time for Feature extraction over OOD dataset: {time.time() - begin}")

        return ood_feat_log, ood_label

    def read_id(self, id_name):
        """
            read feature log from in-distribution (id) cached files

        :param id_name: the ID dataset name, e.g., CIFAR-10

        :return caches: dict, key-value of 'id_feat', 'id_score' and 'id_label'

        """

        caches = {}

        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(
            normalizer(x))  # Last Layer only ## x is changed from multi-layer features to single-layer features


        for split in ['train', 'val']:
            # cache_name = self.cache_path + f"{id_name}_{split}_{self.args.name}.npz"
            # data = np.load(cache_name, allow_pickle=True)
            if split == 'train':
                response = requests.get('http://127.0.0.1:8000/get_train_data/')
                data = response.json()
            elif split == 'val':
                response = requests.get('http://127.0.0.1:8000/get_test_data/')
                data = response.json()

            feat_logs_list = []
            labels_list = []
            names = []
            for item in data['data']:
                feat_logs_list.append(item['feat_log'])
                labels_list.append(item['label'])
                names.append(item['data_name'])

            caches["id_feat_" + split] = prepos_feat(np.array(feat_logs_list))
            caches["id_label_" + split] = np.array(labels_list)
            caches["names"] = names
            # caches["names_" + split] = names

        return caches

    def read_ood(self, ood_name):
        """
            read feature log from out-of-distribution (ood) cached files

        :param ood_dataset: name of the ood dataset, e.g., "SVHN"
        :return caches: dict, key-value of 'ood_feat' and 'ood_score'

        """

        caches = {}
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(
            normalizer(x))  # Last Layer only ## x is changed from multi-layer features to single-layer features

        # Fetch data from the route instead of loading from cache_name
        response = requests.get('http://127.0.0.1:8000/get_ood_data/')
        data = response.json()

        feat_logs_list = []
        labels_list = []
        names = []
        for item in data['data']:
            feat_logs_list.append(item['ood_feat_log'])
            labels_list.append(item['ood_label'])
            names.append(item['file_name'])

        caches['ood_feat'] = prepos_feat(np.array(feat_logs_list))
        caches["ood_label"] = np.array(labels_list)
        caches["names"] = names

        # print(caches)

        return caches

    def ood_detection(self, ood_name, K=50):

        """
            ood detection for new-coming samples

        :param ood_name: name of the ood dataset, e.g., "SVHN"
        :param K: the KNNs

        # required elements for ood detection:
        ## ood_samples: (n, 3, 32, 32), a couple of new-coming normalized images
        ## feat_id_train: the feature logits of ID training set
        ## feat_id_val: the feature logits of ID validation set
        ## feat_ood: the feature logits of new-coming samples

        :return unknown_idx: list, the indices of un-recognized samples
        :return bool_ood: list, ood detection results
        :return scores_conf: the confidence scores of the detection (to be defined)
        :return pred_scores: the class prediction scores
        :return pred_labels: the class predictions

        """
        # id_name = self.args.in_dataset
        id_name = self.args.in_dataset + "_ft" #### For fine-tuning !!

# Lire les features directement à partir de la base de données updatée
        caches_id = self.read_id(id_name)
        caches_ood = self.read_ood(ood_name)


        feat_id_train = caches_id["id_feat_train"]
        feat_id_val = caches_id["id_feat_val"]
        feat_ood = caches_ood['ood_feat']

        names = caches_ood['names']

        print(names)



        print('SHAPE', feat_id_train.shape)
        # Out-of-distribution(OOD) detection
        index = faiss.IndexFlatL2(feat_id_train.shape[1])
        index.add(feat_id_train)
        for k in [K]:  # K = 50 for CIFAR
            # 'index.search' returns (n_feat_test, k), i.e., the knn vectors of each testing sample
            # e.g., D: (10000, 50), containing the distances between (test_sample, kNN), k in range(1, K)

            # Calculate the threshold so that e.g., 95% of ID data is correctly classified
            D, _ = index.search(feat_id_val, k)
            scores_known = -D[:, -1]  # e.g., shape (10000), the L2 distance to the k-th neighbor
            scores_known.sort()
            num_k = scores_known.shape[0]
            threshold = scores_known[round(0.05 * num_k)]

            # evaluation metrics for ood detection
            all_results = []

            # ood detection for new-coming samples
            D, _ = index.search(feat_ood, k)
            scores_ns = -D[:, -1]

            # save indices of detected ood samples
            unknown_idx = []
            for idx, score in enumerate(scores_ns):
                # print(f"new sample {idx} ood detection result is {bool(score > threshold)}")
                if score < threshold:
                    unknown_idx.append(idx)
                else:
                    continue

            # save ood detection results for all new samples
            bool_ood = scores_ns < threshold

            # summary results via metrics, for batch of data
            results = metrics.cal_metric(scores_known, scores_ns)
            all_results.append(results)
            metrics.print_all_results(all_results, self.args.out_datasets, f'KNN k={k}')

            # scores_conf: the confidence scores for ood sample recognition, 1: ood sample, 0: non ood sample
            # score_ns >= threshold: score_conf <= 0.5;
            # score_ns < threshold: score_conf > 0.5;
            # condition: score_ns < 0
            # TODO: to check the max value of score_conf when score_ns = 0
            scores_conf = 1 / (1 + np.exp(-(scores_ns - threshold)))

            for idx, filename in enumerate(names):
                update_data = {
                    'file_name': filename,  # use the actual filename from the dataset
                    'bool_ood': bool(bool_ood[idx]),
                    'scores_conf': float(scores_conf[idx])
                }
                response = requests.post("http://127.0.0.1:8000/update_ood_data2/", json=update_data)
                print(response.content)
                if response.status_code == 200:
                    print(f"Updated data for {filename} successfully!!!!")
                else:
                    print(f"Failed to update data for {filename}!")


            # load the class prediction scores by the base model
        # caches_ood["ood_score"] = caches_ood['ood_score']  # shape: (N, C)
        # pred_scores = np.max(caches_ood["ood_score"], axis=1)  # (N)
        # pred_labels = np.argmax(caches_ood["ood_score"], axis=1)  # (N)

        # return unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels
        return unknown_idx, bool_ood, scores_conf

    def sample_instances(self, y, num_samples=5):
        """
        For each unique label in y, sample 'num_samples' instances and return their indices.

        Parameters:
        - y: 1D array-like, labels vector
        - num_samples: int, number of samples to retrieve for each class

        Returns:
        - samples: dict, keys are unique labels in y, values are lists of sampled indexes for each class
        """
        # Get unique labels
        unique_labels = np.unique(y)

        samples = {}

        for label in unique_labels:
            # Get indexes of all instances of the current class
            label_indexes = np.where(y == label)[0]

            # Randomly sample 'num_samples' indexes
            sampled_indexes = np.random.choice(label_indexes, num_samples, replace=False)

            samples[label] = sampled_indexes

        print('SAMPLES_IND', sampled_indexes)

        return samples

    def build_ft_dataloader(self, ood_name, batch_size, shuffle, ood_class=[0], n_ood=200):
        """
            Build a new dataloader for fine-tuning, with sampled old-class data and new-class data
            The new-class data is labeled by users

        #:param unknown_samples: (n, 3, 32, 32), a couple of new-coming normalized images
        :param ood_name: e.g., 'SVHN'
        :param batch_size: set another batch_size for fine-tuning
        :param shuffle
        :param ood_class: a set of ood classes to be used for fine-tuning
        :param n_ood: number of ood samples for testing

        :return dataloader: (x, y)

        """

        # Key idea:
        # - apply all unknown samples (one/multiple classes) for re-building the embedding space,
        # - select the most representative samples for active labeling

        ############################## Read raw data ##############################
        # read id data
        x_train_id = np.array(torch.stack([x for x, _ in self.trainloaderIn.dataset]))
        y_train_id = np.array([y for _, y in self.trainloaderIn.dataset])
        # read ood data
        loader_out = get_loader_out(self.args, dataset=(None, ood_name), split=('val'))
        self.val_loader_out = loader_out.val_ood_loader  # take the val/test batch of the ood data

        # print('TESTTTTT',self.val_loader_out.dataset[0])
        x_ood = np.array(torch.stack([x for x, _ in self.val_loader_out.dataset]))  # (N, H, W, C)
        y_ood = np.array([y for _, y in self.val_loader_out.dataset])  # (N)

        # sampling ood/new coming data with one/multiple classes
        # samples_ood/id_idx : {class: indices}, the representatfive instances are randomly sampled
        # TODO caching mechanism: cache the most representative samples for fine-tuning
        #       e.g., representatiove sample selection with kNNs of the cluster centroids
        samples_ood_idx = self.sample_instances(y_ood, num_samples=n_ood)

        print('Samples_OOD indexes', samples_ood_idx)

        # select samples from target ood classes
        target_samples_ood_idx = {k: samples_ood_idx[k] for k in ood_class if k in samples_ood_idx}
        ood_idx = [index for indices in target_samples_ood_idx.values() for index in indices]
        x_repr_ood, y_repr_ood = x_ood[ood_idx], y_ood[ood_idx]

        # select samples from all id classes
        ratio_old_new = 1
        n_old_per_class = ratio_old_new * n_ood
        samples_id_idx = self.sample_instances(y_train_id, num_samples=n_old_per_class)
        print('Samples_ID indexes', samples_id_idx)
        id_idx = [index for indices in samples_id_idx.values() for index in indices]
        x_repr_id, y_repr_id = x_train_id[id_idx], y_train_id[id_idx]

        # manuelly modify 'y_repr_ood' by adding the max value of 'y_repr_id'
        y_repr_ood += self.num_classes
        # build a new dataloader for fine-tuning
        x_repr = np.concatenate([x_repr_id, x_repr_ood], axis=0)
        y_repr = np.concatenate([y_repr_id, y_repr_ood], axis=0)
        print(f"test 4: x_repr.shape is {x_repr.shape}, y_repr.shape is {y_repr.shape}")
        dataset = TensorDataset(torch.tensor(x_repr), torch.tensor(y_repr))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    # option 1: use sampled data for training
    # option 2: use all data for training
    def train(self, train_loader, classifier, optimizer, epochs):

        """
            train the classifer with embeddings as input

        :param train_loader: the dataloader for embeddings
        :param classifier: the classifier to be trained
        :param optimizer: name of the ood dataset, e.g., "SVHN"
        :param epochs: number of training epoch

        :return classifier, losses.avg, top1.avg

        """

        self.model.eval()
        classifier.train()

        for epoch in range(epochs):
            criterion = torch.nn.CrossEntropyLoss()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()

            end = time.time()
            for idx, (features, labels) in enumerate(train_loader):
                data_time.update(time.time() - end)

                bsz = labels.shape[0]
                output = classifier(features.float())
                loss = criterion(output, labels.long())

                # update metric
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            # print info
            if (epoch + 1) % 10 == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()

        return classifier, losses.avg, top1.avg

    def fine_tune(self, dataloader, optimizer, epochs, method='SupCon'):
        """
            Fine tune the base model for embedding learning

        :param dataloader: data used to fine-tune the base model
        :param optimizer
        :param epochs: training epochs
        :param method: the loss function

        :return losses.avg

        """
        # freeze all layer except the (last) head projection layer
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.head.requires_grad = True

        begin = time.time()
        for epoch in range(epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            # fine-tuning 'model'
            # model: SupConResNet, encoder + head for feature projection
            end = time.time()
            # (N, C, H, W), (N)
            for idx, (batch_x, labels) in enumerate(dataloader):
                bsz = labels.shape[0]
                batch_x = torch.autograd.Variable(batch_x, requires_grad=True)
                features = self.model(batch_x)  # (N, 128)

                # TODO: to check the data augmentations for multiple views
                # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                features = torch.cat([features.unsqueeze(1), features.unsqueeze(1)], dim=1)
                if method == 'SupCon':
                    criterion = SupConLoss()
                    loss = criterion(features, labels)
                elif method == 'SimCLR':
                    loss = criterion(features)
                else:
                    raise ValueError('contrastive method not supported: {}'.
                                     format(method))

                # update metric
                losses.update(loss.item(), bsz)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # print info
                if (idx + 1) % 10 == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                        epoch, idx + 1, len(dataloader), batch_time=batch_time,
                        data_time=data_time, loss=losses))
                    sys.stdout.flush()
        print(f"Time for fine-tuning over restructured dataset: {time.time() - begin}")

        return losses.avg

    def inference(self, model, dataloader, metrics=True):
        """

        :param self:
        :param model: classifier
        :param metrics:
        :return:
        """

        model.eval()
        preds, ground_truths = [], []

        # (N, H, W, C), (N)
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            out = model(batch_x.float()).detach().cpu().numpy()
            preds.append(out)
            ground_truths.append(batch_y)
            if batch_idx % 100 == 0:
                print(f"inference batches: {batch_idx}/{len(dataloader)}")

        preds = np.concatenate(preds, axis=0)  # (N, C)
        preds = np.argmax(preds, axis=1)  # (N)
        ground_truths = np.concatenate(ground_truths, axis=0)

        print('inference dataset info: ', np.unique(ground_truths, return_counts=True))

        # ticks = ['']
        if metrics:
            # plt.figure()
            # cfm = confusion_matrix(target, predictions)
            # htm = heatmap(cfm, cmap='twilight', annot=True, xticklabels=ticks, yticklabels=ticks)
            # figure = htm.get_figure()
            # figure.savefig('./results/conf_matrix.pdf', dpi=400)
            print(classification_report(ground_truths, preds, digits=5))

        return accuracy_score(ground_truths, preds)


    def train_global(self, ood_name, shuffle, ood_class, n_ood):
        base_model_name = self.args.name.split('_')[0]

        n_new_class = len(ood_class)
        batch_size = self.args.batch_size
        # Initiate dataloaders for (embeddings, targets)

        caches_id = self.read_id(id_name=self.args.in_dataset)
        print('CAAAAAACHE IIIIID', caches_id.keys())
        print('CAAAAAACHE IIIIID', caches_id)
        # filename_id_train = caches_id.get("data_name", "Default_ID_Filename")
        # names of the files : caches_id['names']

        caches_ood = self.read_ood(ood_name)
        print('CAAAAAACHE OOOOOD', caches_ood.keys())
        print('CAAAAAACHE OOOOOD', caches_ood)
        # filename_ood = caches_ood.get("data_name", "Default_OOD_Filename")
        # names of the files : caches_ood['names']

        # Now, you have the filenames stored in filename_id_train and filename_ood
        # print(f"ID training filename: {filename_id_train}")
        # print(f"OOD filename: {filename_ood}")
        feat_id_train, y_id_train = caches_id["id_feat_train"], caches_id["id_label_train"]
        feat_id_val, y_id_val = caches_id["id_feat_val"], caches_id["id_label_val"]
        feat_ood, y_ood = caches_ood['ood_feat'], caches_ood['ood_label']

        print("test 1: ", feat_id_train.shape, y_id_train.shape)
        dataset_train = TensorDataset(torch.tensor(feat_id_train), torch.tensor(y_id_train))
        dataset_val = TensorDataset(torch.tensor(feat_id_val), torch.tensor(y_id_val))
        dataset_ood = TensorDataset(torch.tensor(feat_ood), torch.tensor(y_ood))

        # train and save the init classifier (SGD)
        print("Train and save the init classifier \n")
        if not os.path.exists(self.init_classifier_path):
            dataloader_train = DataLoader(dataset_train, batch_size, shuffle)
            self.classifier_init, loss_avg, top1_avg = self.train(dataloader_train, self.classifier_init,
                                                                  set_optimizer(self.args, self.classifier_init),
                                                                  epochs=self.args.epochs_clf)
            torch.save({"state_dict": self.classifier_init.state_dict()}, self.init_classifier_path)
        else:
            print(f"Init classifier exist in {self.init_classifier_path}\n")
            self.classifier_init = get_classifier(self.args, self.num_classes, load_ckpt=True, classifier_flag='init')

            # evaluation of the init model
        print("Evaluation of the init model \n")
        print("ID data")
        dataloader_val = DataLoader(dataset_val, batch_size, shuffle)
        accu_score_id = self.inference(self.classifier_init, dataloader_val, shuffle)
        print("OOD data")
        dataloader_ood = DataLoader(dataset_ood, batch_size, shuffle)
        accu_score_ood = self.inference(self.classifier_init, dataloader_ood, shuffle)
        print("Initial model's accuracy on IN data: {}, on OOD data: {}".format(accu_score_id, accu_score_ood))

        # Fine-tuning with ood data
        ft_dataloader = self.build_ft_dataloader(ood_name, batch_size, shuffle, ood_class, n_ood)
        loss_avg_ft = self.fine_tune(ft_dataloader, set_optimizer(self.args, self.model), epochs=self.args.epochs_ft,
                                     method='SupCon')

        torch.save({"state_dict": self.model.state_dict()}, self.ft_model_path)

        classifier_ft = LinearClassifier(name=base_model_name, num_classes=self.num_classes + n_new_class)
# HERE, update la bdd
        # re-extract features for ID and OOD data using fine-tuned model
        self.id_feature_extract(self.model, self.args.in_dataset + "_ft")
        caches_id_ft = self.read_id(self.args.in_dataset + "_ft")
        feat_id_train, y_id_train = caches_id_ft["id_feat_train"], caches_id_ft["id_label_train"]
        feat_id_val, y_id_val = caches_id_ft["id_feat_val"], caches_id_ft["id_label_val"]
        feat_ood, y_ood = self.ns_feature_extract(self.model, ft_dataloader, ood_name + "_ft")

        # assert len(caches_id_ft['names']) == len(feat_id_train) == len(y_id_train)
        # assert len(caches_id_ft['names']) == len(feat_id_val) == len(y_id_val)

        response = requests.get(f"http://127.0.0.1:8000/list_files/{ood_name}/")
        if response.status_code == 200:
            file_data = response.json()
            filenames = file_data.get("files", [])
        else:
            print("Failed to retrieve filenames!")
            filenames = []  # Empty list as a fallback

        print('NEW CACHES ID FT', caches_id_ft)
        print('NEW CACHES ID FT', caches_id_ft.keys())
        #Filenames for id : caches_id_ft['names']
        print('Filenames for OOD:', filenames)


        # for fname, f_ood, y in zip(caches_id_ft['names'], feat_id_train, y_id_train):
        #     print(f"Sending {fname} with features {f_ood[:5]} and label {y}")
        #     response = requests.post("http://127.0.0.1:8000/update_train_data/", json={
        #         "data_name": fname,
        #         "feat_log": f_ood.tolist(),
        #         "id_label_train": int(y)
        #     })
        #
        #     if response.status_code != 200:
        #         print(f"Error updating ID Train data for {fname}:", response.content)
        #     else:
        #         print('YES ID TRAIN')

        for fname, f_ood, y in zip(caches_id_ft['names'], feat_id_val, y_id_val):
            print(f"Sending VAL data {fname} with features {f_ood[:5]} and label {y}")

            response = requests.post("http://127.0.0.1:8000/update_test_data/", json={
                "data_name": fname,
                "feat_log": f_ood.tolist(),
                "label": int(y)
            })

            if response.status_code != 200:
                print(f"Error updating ID Val data for {fname}:", response.content)
            else:
                print('YES ID VAL ')


        #
        # for fname, f_ood, y in zip(filenames, feat_ood, y_ood):
        #     response = requests.post("http://127.0.0.1:8000/update_data_collection/", json={
        #         "file_name": fname,
        #         "ood_feat_log": f_ood.tolist(),
        #         "ood_label": int(y)
        #     })
        #
        #     if response.status_code != 200:
        #         print(f"Error updating OOD data for {fname}:", response.content)
        #     else:
        #         print('YES OOD')

        dataset_train = TensorDataset(torch.tensor(feat_id_train), torch.tensor(y_id_train))
        dataset_val = TensorDataset(torch.tensor(feat_id_val), torch.tensor(y_id_val))
        dataset_ood = TensorDataset(torch.tensor(feat_ood), torch.tensor(y_ood))
        # train and save the fine-tuned classifier (SGD)
        dataloader_train = DataLoader(dataset_train, batch_size, shuffle)
        self.classifier_ft, loss_avg, top1_avg = self.train(dataloader_train, classifier_ft,
                                                            set_optimizer(self.args, self.classifier_init),
                                                            epochs=self.args.epochs_clf)
        # Change the line above to :
        # self.classifier_ft, loss_avg, top1_avg = self.train(dataloader_train, classifier_ft,
        #                                                     set_optimizer(self.args, self.classifier_ft),
        #                                                     epochs=self.args.epochs_clf)
        torch.save({"state_dict": self.classifier_ft.state_dict()}, self.ft_classifier_path)




        # evaluation of the fine-tuned model 
        print("Evaluation of the fine-tuned model \n")
        print("ID data")
        dataloader_val = DataLoader(dataset_val, batch_size, shuffle)
        accu_score_id = self.inference(self.classifier_ft, dataloader_val, shuffle)
        print("OOD data")
        dataloader_ood = DataLoader(dataset_ood, batch_size, shuffle)
        accu_score_ood = self.inference(self.classifier_ft, dataloader_ood, True)
        print("Fine-tuned model's accuracy on IN data: {}, on OOD data: {}".format(accu_score_id, accu_score_ood))