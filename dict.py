import numpy as np

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
#
# print(unpickle('/Users/apagnoux/Downloads/Continuous_Learning_RM2-master/cache/CIFAR-10_ft_val_resnet18-supcon.npz').keys())


with np.load('cache/CIFAR-10_ft_val_resnet18-supcon.npz') as data:
    # feat_log et label
    # Print the keys in the .npz file
    for i in data['feat_log']:
        print(i[0])
    print(list(data.keys()))