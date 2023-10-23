def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


print(unpickle('/Users/apagnoux/PycharmProjects/pythonProject2/datasets/data/cifar-10-batches-py/data_batch_1').keys())