# import os
# import pickle
# import pymongo
# import requests
#
# # Define your MongoDB client and collection here
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["database"]
# collection = db["cifar10_images"]
#
# # Define the CIFAR-10 dataset URLs
# cifar10_urls = [
#     "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
# ]
#
#
# def unpickle(file):
#     with open(os.path.join("datasets/data/cifar-10-batches-py", file), 'rb') as fo:
#         cifar_data = pickle.load(fo, encoding='bytes')
#     return cifar_data
#
#
# def download_cifar10_data():
#     for url in cifar10_urls:
#         # Download the CIFAR-10 dataset
#         response = requests.get(url)
#         if response.status_code == 200:
#             # Save the downloaded file with an absolute path
#             file_name = os.path.join(os.getcwd(), url.split("/")[-1])
#             with open(file_name, "wb") as f:
#                 f.write(response.content)
#             print(f"Downloaded {file_name}")
#         else:
#             print(f"Failed to download {url}")
#
#
#
# def push_cifar10_data_to_mongodb():
#     for batch_num in range(1, 6):  # CIFAR-10 has 5 batches
#         file_name = f"data_batch_{batch_num}"
#         cifar_data = unpickle(file_name)
#
#         for i, (data, label, filename) in enumerate(
#                 zip(cifar_data[b"data"], cifar_data[b"labels"], cifar_data[b"filenames"])
#         ):
#             data_dict = {
#                 "batch_label": cifar_data[b"batch_label"].decode("utf-8"),
#                 "labels": label,
#                 "data": data.tolist(),
#                 "filenames": filename.decode("utf-8"),
#             }
#             # Check if data already exists
#             existing_data = collection.find_one({"filenames": filename.decode("utf-8")})
#             if existing_data:
#                 print(f"Data for image {filename.decode('utf-8')} already exists in MongoDB.")
#             else:
#                 # Push data_dict to MongoDB
#                 collection.insert_one(data_dict)
#                 print(f"Pushed data for image {filename.decode('utf-8')} in batch {batch_num} to MongoDB")
#
#
# def push_cifar10_data_to_mongodb_test():
#     for batch_num in range(1, 6):  # CIFAR-10 has 5 batches
#         file_name = f"test_batch"
#         cifar_data = unpickle(file_name)
#
#         for i, (data, label, filename) in enumerate(
#                 zip(cifar_data[b"data"], cifar_data[b"labels"], cifar_data[b"filenames"])
#         ):
#             data_dict = {
#                 "batch_label": cifar_data[b"batch_label"].decode("utf-8"),
#                 "labels": label,
#                 "data": data.tolist(),
#                 "filenames": filename.decode("utf-8"),
#             }
#             # Check if data already exists
#             existing_data = collection.find_one({"filenames": filename.decode("utf-8")})
#             if existing_data:
#                 print(f"Data for image {filename.decode('utf-8')} already exists in MongoDB.")
#             else:
#                 # Push data_dict to MongoDB
#                 collection.insert_one(data_dict)
#                 print(f"Pushed data for image {filename.decode('utf-8')} in batch {batch_num} to MongoDB")
#
#
#
# if __name__ == "__main__":
#     # download_cifar10_data()
#     # push_cifar10_data_to_mongodb()
#     push_cifar10_data_to_mongodb_test()
#     client.close()
#


# =================================================================
# from six.moves import cPickle as pickle
# from  PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
# f = open('/Users/apagnoux/PycharmProjects/pythonProject2/datasets/data/cifar-10-batches-py/data_batch_1', 'rb')
#
# tupled_data= pickle.load(f, encoding='bytes')
#
# f.close()
#
# img = tupled_data[b'data']
#
# single_img = np.array(img[5])
#
# single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
#
# plt.imshow(single_img_reshaped)
# plt.show()


# =============================================================================
import pickle
import pymongo
import requests

# Connect to the MongoDB database
client = pymongo.MongoClient("localhost", 27017)
db = client["cifar10"]

# Download the CIFAR10 train and test batches
train_batches = []
test_batches = []

for i in range(1, 6):
    train_batch_url = f"https://www.cs.toronto.edu/~kriz/cifar-10-python/cifar-10-python.tar.gz/cifar-10-batches-py/data_batch_{i}"
    test_batch_url = f"https://www.cs.toronto.edu/~kriz/cifar-10-python/cifar-10-python.tar.gz/cifar-10-batches-py/test_batch"

    # Download the train batch
    train_batch_response = requests.get(train_batch_url)

    # Check the integrity of the pickle file
    try:
        pickle.loads(train_batch_response.content)
    except pickle.UnpicklingError:
        raise Exception(f"The pickle file at '{train_batch_url}' is corrupted.")

    # Load the train batch
    train_batch = pickle.loads(train_batch_response.content)
    train_batches.append(train_batch)

    # Download the test batch
    test_batch_response = requests.get(test_batch_url)

    # Check the integrity of the pickle file
    try:
        pickle.loads(test_batch_response.content)
    except pickle.UnpicklingError:
        raise Exception(f"The pickle file at '{test_batch_url}' is corrupted.")

    # Load the test batch
    test_batch = pickle.loads(test_batch_response.content)
    test_batches.append(test_batch)

# Push the train and test batches to the database
train_collection = db["train"]
test_collection = db["test"]

for train_batch in train_batches:
    train_collection.insert_one(train_batch)

for test_batch in test_batches:
    test_collection.insert_one(test_batch)

# Close the connection to the database
client.close()



