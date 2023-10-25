import os, json
import torch


class Exp_OWLbasic(object):
    def __init__(self, args):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _select_optimizer(self):
        pass

    def _get_data(self):
        pass

    def id_feature_extract(self):
        pass

    def ns_feature_extract(self):
        pass
    
    # the communication with Databse, e.g., inference on the coming instances (features, score, label) 
    # DB_1: the entire database with all coming instances
    # DB_2: the partial databse with representative samples
    def interact_databse(self):
        pass 
    
    def vali(self):
        pass

    def ood_detection(self):
        pass

    def fine_tune(self):
        # Questions: is it possible to use only supervised contrastive loss to fine-tune the model?
        pass

    def test(self):
        pass
