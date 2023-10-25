import os
import time
import warnings
import math
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from pathlib import Path
from torchvision import transforms
from .exp_OWLbasic import Exp_OWLbasic
from ..utils import metrics
from ..utils.data_loader import get_loader_in
from ..utils.model_loader import get_model
from ..utils.custom_loader import GenericImageDataset
from torch.utils.data import DataLoader
import json

warnings.filterwarnings('ignore')

np.random.seed(1)

# from API.api_helper import push_results_to_db, push_embeddings_to_db, push_scores_to_db
class Exp_OWL(Exp_OWLbasic):
    def __init__(self, args):
        super(Exp_OWL, self).__init__(args)  ## init device

        # result save path
        # testing_info = "model_{}_{}".format(
        #     self.args.name,
        #     self.args.in_dataset,
        #     self.args.out_datasets
        #     )
        # self.save_path = Path(self.args.save_path + testing_info)

        # dataloaders
        self.loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
        self.trainloaderIn, self.testloaderIn, self.num_classes = self.loader_in_dict.train_loader, self.loader_in_dict.val_loader, self.loader_in_dict.num_classes
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        self.model = get_model(self.args, self.num_classes, load_ckpt=True)  # load pre-trained model
        return self.model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def id_feature_extract(self):

        """
            extract features from in-distribution samples

        #id_samples: (n, 3, 32, 32), a couple of new-coming normalized images

        :return None
        :save files: feat_log, score_log, label_log
        """

        batch_size = self.args.batch_size
        dummy_input = torch.zeros((1, 3, 32, 32)).to(self.device)
        # score is the last layer output, i.e., after FC layer
        score, feature_list = self.model.feature_list(dummy_input)
        featdims = [feat.shape[1] for feat in feature_list]

        begin = time.time()

        # 'val' loader: to get the threshold distance in ID data, to check if a sample is OOD

        for split, in_loader in [('train', self.trainloaderIn), ('val', self.testloaderIn), ]:
            # why testing data in in-distribution data?
            cache_name = f"{self.args.save_path}/{self.args.in_dataset}_{split}_{self.args.name}_in_alllayers.npz"
            if not os.path.exists(cache_name):
                # feat_log: the concatenated features of each layer out in ResNet
                # score_log: the prediction scores over classes
                # label_log: the ground truth labels
                feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))
                score_log = np.zeros((len(in_loader.dataset), self.num_classes))
                label_log = np.zeros(len(in_loader.dataset))

                # Update metadata with those features feat_log, score_log, label_log for CIFAR10
                all_metadata = []

                self.model.eval()
                for batch_idx, (inputs, targets) in enumerate(in_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    start_ind = batch_idx * batch_size
                    end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                    score, feature_list = self.model.feature_list(inputs)
                    out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list],
                                    dim=1)

                    feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                    label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                    score_log[start_ind:end_ind] = score.data.cpu().numpy()

                    # Creating metadata dictionary and appending to the all_metadata list
                    for idx in range(start_ind, end_ind):
                        metadata = {
                            'data_name': f'data_{idx}',  # Replace with actual data name if available
                            'dataset_split': split,
                            'feat_log': feat_log[idx].tolist(),
                            'score_log': score_log[idx].tolist(),
                            'label_log': int(label_log[idx])
                        }
                        all_metadata.append(metadata)

                    if batch_idx % 100 == 0:
                        print(f"id batches: {batch_idx}/{len(in_loader)}")

                # Writing the entire metadata list to a single JSON file after the loop
                with open(f'metadata_id_{split}.json', 'w') as json_file:  # The filename is now dynamic based on split
                    json.dump(all_metadata, json_file)
                # print("feature shape, feat_log.T: {}, score_log.T: {}, label_log: {}".format(feat_log.T.shape, score_log.T.shape, label_log.shape))
                np.savez(cache_name, feat_log=feat_log, score_log=score_log, label_log=label_log)
            else:
                print(f"Features for {self.args.in_dataset} already extracted and cached in {cache_name}")
                continue
                # data = np.load(cache_name, allow_pickle=True)
                # feat_log = data['feat_log']
                # score_log = data['score_log']
                # label_log = data['label_log']

        print(f"Time for Feature extraction over ID training/validation set: {time.time() - begin}")

    def ns_feature_extract(self, ood_dataset):

        """
            extract features from new-coming samples

        #:param ood_samples: (n, 3, 32, 32), a couple of new-coming normalized images
        :param ood_dataset: name of the ood dataset, e.g., "SVHN"

        :return None
        :save files: feat_log, score_log, label_log
        """
        imagesize = 32
        transform_test = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
            #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        batch_size = self.args.batch_size
        dummy_input = torch.zeros((1, 3, 32, 32)).to(self.device)
        # score is the last layer output, i.e., after FC layer
        score, feature_list = self.model.feature_list(dummy_input)
        featdims = [feat.shape[1] for feat in feature_list]

        # ood data handling
        # TODO: the adjust the input ood data
        begin = time.time()

        # loader_test_dict et out_loader
        #         loader_test_dict = get_loader_out(self.args, ood_dataset)
        #         out_loader = loader_test_dict.val_ood_loader

        all_metadata_ood = []
        ood_data = GenericImageDataset(source=ood_dataset, mode='external', transform=transform_test)
        out_loader = DataLoader(ood_data, batch_size=self.args.batch_size, shuffle=False)
        cache_name = f"{self.args.save_path}/{ood_dataset}vs{self.args.in_dataset}_{self.args.name}_out_alllayers.npz"
        if not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
            ood_score_log = np.zeros((len(out_loader.dataset), self.num_classes))

            self.model.eval()
            for batch_idx, (inputs, image_name) in enumerate(out_loader):
                inputs = inputs.to(self.device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                score, feature_list = self.model.feature_list(inputs)
                out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
                embeddings_batch = out.data.cpu().numpy()
                # for idx in range(embeddings_batch.shape[0]):
                #     embeddings = embeddings_batch[idx].tolist()
                #     push_embeddings_to_db(image_name[idx], ood_dataset, embeddings)

                for idx in range(embeddings_batch.shape[0]):
                    metadata_ood = {
                        'name': image_name[idx],
                        'dataset': ood_dataset,
                        'class_ground_truth': "",
                        'embeddings': embeddings_batch[idx].tolist()
                    }
                    all_metadata_ood.append(metadata_ood)

                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"ood batches: {batch_idx}/{len(out_loader)}")
            np.savez(cache_name, ood_feat_log=ood_feat_log, ood_score_log=ood_score_log)
        else:
            print(f"Features for {self.args.out_datasets} already extracted and cached in {cache_name}")
            data = np.load(cache_name, allow_pickle=True)
            ood_feat_log = data['ood_feat_log']
            ood_score_log = data['ood_score_log']
        with open('metadata_ood.json', 'w') as json_file:
            json.dump(all_metadata_ood, json_file)

        print(f"Time for Feature extraction over OOD dataset: {time.time() - begin}")

    def ood_detection(self, ood_dataset, K=50):

        """
            ood detection new-coming samples

        :param ood_dataset: name of the ood dataset, e.g., "SVHN"
        :param K: the KNNs

        # required elements for ood detection:
        ## ood_samples: (n, 3, 32, 32), a couple of new-coming normalized images
        ## train_feat_log: the feature logs of ID training set
        ## val_feat_log: the feature logs of ID validation set
        ## ns_feat_log: the feature logs of new-coming samples

        :return unknown_idx: list, the indices of un-recognized samples
        :return bool_ood: list, ood detection results
        :return scores_conf: the confidence scores of the detection (to be defined)
        :return pred_scores: the class prediction scores
        :return pred_labels: the class predictions

        """

        caches = {}

        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448, 960)]))  # Last Layer only

        # read feature log from in-distribution (id) cached files
        for split in ['train', 'val']:
            cache_name = os.path.join(self.args.save_path,
                                      f"{self.args.in_dataset}_{split}_{self.args.name}_in_alllayers.npz")

            # cache_name = self.args.save_path + f"{self.args.in_dataset}_{split}_{self.args.name}_in_alllayers.npz"
            data = np.load(cache_name, allow_pickle=True)
            caches["id_feat_log_" + split] = data['feat_log']
            caches["id_score_log" + split] = data['score_log']
            caches["id_label_log" + split] = data['label_log']
            caches["feat_" + split] = prepos_feat(caches["id_feat_log_" + split])  # (N, 960) -> (N, 512)

        # read feature logs from out-of-distribution (ood) cached files
        cache_name = f"{self.args.save_path}/{ood_dataset}vs{self.args.in_dataset}_{self.args.name}_out_alllayers.npz"
        data = np.load(cache_name, allow_pickle=True)
        feat_ood = prepos_feat(data['ood_feat_log'])  # (N, 960) -> (N, 512)

        # Out-of-distribution(OOD) detection
        index = faiss.IndexFlatL2(caches["feat_train"].shape[1])
        index.add(caches["feat_train"])
        for k in [K]:  # K = 50 for CIFAR
            # 'index.search' returns (n_feat_test, k), i.e., the knn vectors of each testing sample
            # e.g., D: (10000, 50), containing the distances between (test_sample, kNN), k in range(1, K)

            # Calculate the thresold so that e.g., 95% of ID data is correctly classified
            D, _ = index.search(caches["feat_val"], k)
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

            # load the class prediction scores by the base model
        caches["ood_score_log"] = data['ood_score_log']  # shape: (N, C)
        pred_scores = np.max(caches["ood_score_log"], axis=1)  # (N)
        pred_labels = np.argmax(caches["ood_score_log"], axis=1)  # (N)

        with open('metadata_ood.json', 'r') as json_file:
            metadata_ood = json.load(json_file)

        assert len(metadata_ood) == len(bool_ood), "Mismatch between metadata_ood and bool_ood"
        assert len(metadata_ood) == len(scores_conf), "Mismatch between metadata_ood and scores_conf"
        assert len(metadata_ood) == len(pred_scores), "Mismatch between metadata_ood and pred_scores"
        assert len(metadata_ood) == len(pred_labels), "Mismatch between metadata_ood and pred_labels"

        # Step 2: Update the metadata_ood with new fields
        for idx, data in enumerate(metadata_ood):
            data["unknown_idx"] = idx in unknown_idx
            data["bool_ood"] = bool_ood[idx].item()
            data["scores_conf"] = scores_conf[idx].item()
            data["pred_scores"] = pred_scores[idx].item()
            data["pred_labels"] = pred_labels[idx].item()

        # Step 3: Write back the updated metadata_ood to JSON file
        with open('metadata_ood.json', 'w') as json_file:
            json.dump(metadata_ood, json_file, indent=4)

        # for idx in range(len(pred_scores)):
        #     push_scores_to_db(
        #         idx=idx,
        #         dataset_name=ood_dataset,
        #         bool_ood=bool_ood[idx],
        #         pred_scores=pred_scores[idx],
        #         pred_labels=pred_labels[idx],
        #         scores_confidence=scores_conf[idx]
        #     )



        # metadata_file = Path("data/metadata") / "metadata.json"  # Make sure to specify the correct path
        # with metadata_file.open("r") as mf:
        #     metadata = json.load(mf)
        # metadata = metadata[:len(bool_ood)]
        # print(f"Length of metadata: {len(metadata)}")
        # print(f"Length of bool_ood: {len(bool_ood)}")
        # print(f"Length of scores_conf: {len(scores_conf)}")
        # print(f"Length of pred_scores: {len(pred_scores)}")
        # print(f"Length of pred_labels: {len(pred_labels)}")
        #
        # assert len(metadata) == len(bool_ood), "Mismatch between metadata and bool_ood"
        # assert len(metadata) == len(scores_conf), "Mismatch between metadata and scores_conf"
        # assert len(metadata) == len(pred_scores), "Mismatch between metadata and pred_scores"
        # assert len(metadata) == len(pred_labels), "Mismatch between metadata and pred_labels"
        #
        # # Assuming that the order of entries in metadata matches with the order of results
        # for idx, image_data in enumerate(metadata):
        #     if image_data["dataset"] == ood_dataset:  # Matching based on the dataset name
        #         image_data["bool_ood"] = bool_ood[idx].item()  # Convert numpy boolean to native Python boolean
        #         image_data["scores_conf"] = scores_conf[idx].item()
        #         image_data["pred_scores"] = pred_scores[idx].item()
        #         image_data["pred_labels"] = pred_labels[idx].item()
        # with metadata_file.open("w") as mf:
        #     json.dump(metadata, mf, indent=4)

        return unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels

    def fine_tune(self, unknown_samples):

        """
            fine tune the model with unknown samples labeled by users

        :param ood_samples: (n, 3, 32, 32), a couple of new-coming normalized images

        :return model

        """

        # caching mechanism: cache the most representative samples for fine-tuning

        return

    def vali(self, vali_loader, new_graph):

        self.model.eval()
        return

    def test(self, new_graph=True):

        self.model.eval()

        preds = []
        trues = []

        return
