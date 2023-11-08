import base64
import datetime
import json
import pickle
import shutil

from bson import ObjectId
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Path as FastAPIPath
from typing import Dict, Union
from pathlib import Path
import os
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from starlette.responses import JSONResponse
from starlette.requests import Request
from fastapi import FastAPI
from typing import List
from gridfs import GridFS
from fastapi.responses import FileResponse

app = FastAPI()
UPLOAD_DIR = "data/images"  # New directory path for images
os.makedirs(UPLOAD_DIR, exist_ok=True)
METADATA_DIR = "data/metadata"  # Directory for metadata
BATCH_DIR = "datasets/data/cifar-10-batches-py"
# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["continuous-learning"]
collection = db["data"]
collection2 = db["id_metadata_collection"]
collection3 = db["id_metadata_collection_TEST"]
fs = GridFS(db)

train_collection = db["train"]
test_collection = db["test"]

class Metadata(BaseModel):
    batch_label: str
    labels: List[int]
    filenames: List[str]


class UpdateData(BaseModel):
    file_name: str
    bool_ood: bool
    scores_conf: float
    # pred_scores: float
    # pred_labels: int
    # unknown_idx: int

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict



@app.post("/uploadfiles/")
async def create_upload_files(dataset_name: str = Form(...), files: list[UploadFile] = Form(...)):
    base_path = Path(UPLOAD_DIR)
    dataset_path = base_path / dataset_name

    dataset_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists; if not, create it

    saved_files = []

    # Store images
    for file in files:
        try:
            file_path = dataset_path / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        except Exception as e:
            print(f"Could not save file {file.filename}. Reason: {e}")
            continue

    # Instead of "metadata.json", we'll have "metadata_<dataset_name>.json"
    metadata_file = Path(METADATA_DIR) / "metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    if metadata_file.exists():
        with metadata_file.open("r") as mf:
            metadata = json.load(mf)
    else:
        metadata = []

    # Update metadata in MongoDB
    for image_file in saved_files:
        # Create the metadata document for each image
        metadata_document = {
            "file_name": image_file,
            "dataset": dataset_name,
            "file_path": str(dataset_path / image_file),  # Convert Path object to string
            "upload_timestamp": datetime.datetime.now().isoformat()  # Store the timestamp in ISO format
        }

        # Insert the metadata document into the MongoDB collection
        collection.insert_one(metadata_document)

    # Base64 encode the actual image data for storage in the metadata file
    for image_file in saved_files:
        image_path = dataset_path / image_file
        with image_path.open("rb") as img:
            encoded_content = base64.b64encode(img.read()).decode('utf-8')
        metadata.append({
            "name": image_file,
            "dataset": dataset_name,
            "data": encoded_content
        })

    # Save the updated metadata
    with metadata_file.open("w") as mf:
        json.dump(metadata, mf)

    return {"filenames": saved_files}



@app.get("/get_datasets/")
async def list_datasets():
    base_path = Path(UPLOAD_DIR)

    # List all directories under UPLOAD_DIR
    datasets = [d.name for d in base_path.iterdir() if d.is_dir()]

    return {"datasets": datasets}

@app.post("/store_metadata/")
async def store_metadata():
    data_path = Path(BATCH_DIR)

    # Load train batches from local storage
    train_batches = [unpickle(data_path / f"data_batch_{i}") for i in range(1, 6)]

    # Load test batch from local storage
    test_batches = [unpickle(data_path / "test_batch")]

    # Store metadata for each image in MongoDB
    for train_batch in train_batches:
        for label, filename in zip(train_batch[b'labels'], train_batch[b'filenames']):
            train_metadata = {
                "batch_label": train_batch[b'batch_label'].decode("utf-8"),
                "label": label,
                "filename": filename.decode("utf-8")
            }
            train_collection.insert_one(train_metadata)

    for test_batch in test_batches:
        for label, filename in zip(test_batch[b'labels'], test_batch[b'filenames']):
            test_metadata = {
                "batch_label": test_batch[b'batch_label'].decode("utf-8"),
                "label": label,
                "filename": filename.decode("utf-8")
            }
            test_collection.insert_one(test_metadata)

    return {"status": "Metadata stored successfully for each image"}


class FeatureData(BaseModel):
    data_name: str
    dataset_split: str
    feat_log: list
    label: int
    repr_flag: bool = None  # If this is optional


@app.post("/push_feature_data/")
async def push_feature_data(data: FeatureData):
    # Determine the correct collection based on the dataset_split
    if data.dataset_split == "train":
        collection = train_collection
    elif data.dataset_split == "val":  # Assuming 'val' means 'test' in your context
        collection = test_collection
    else:
        raise HTTPException(status_code=400, detail=f"Unknown dataset split: {data.dataset_split}")

    # Insert the data into the appropriate MongoDB collection
    collection.insert_one(data.dict())
    # collection.update_one(query, update, upsert=True)

    return {"message": f"Data for {data.data_name} inserted successfully into {data.dataset_split} collection!"}


class OODUpdate(BaseModel):
    file_name: str  # to identify which document to update
    ood_feat_log: List[float]  # since it seems like a list of floats from your function
    ood_label: int

class OODUpdate2(BaseModel):
    file_name: str  # to identify which document to update
    bool_ood: bool # since it seems like a list of floats from your function
    scores_conf: float

@app.post("/update_ood_data/")
async def update_ood_data(data: OODUpdate):
    query = {"file_name": data.file_name}
    update = {
        "$set": {
            "ood_feat_log": data.ood_feat_log,
            "ood_label": data.ood_label
        }
    }

    result = collection.update_one(query, update)

    if result.modified_count == 1:
        return {"status": "success", "message": f"Document with file_name {data.file_name} updated."}
    else:
        raise HTTPException(status_code=400, detail=f"Could not update document with file_name {data.file_name}.")



@app.post("/update_ood_data2/")
async def update_ood_data2(data: OODUpdate2):
    query = {"file_name": data.file_name}
    update = {
        "$set": {
            "bool_ood": data.bool_ood,
            "scores_conf": data.scores_conf
        }
    }

    result = collection.update_one(query, update)

    if result.modified_count == 1:
        return {"status": "success", "message": f"Document with file_name {data.file_name} updated."}
    else:
        raise HTTPException(status_code=400, detail=f"Could not update document with file_name {data.file_name}.")



# class ScoresUpdate(BaseModel):
#     ood_feat_log: List[float] # to identify which document to update
#     scores_conf: float
#     bool_ood: bool
#
#
# @app.post("/update_scores/")
# async def update_scores(data: ScoresUpdate):
#     # Define the query and the update to be made
#     query = {"ood_feat_log": data.ood_feat_log}
#     update = {
#         "$set": {
#             "scores_conf": data.scores_conf,
#             "bool_ood": data.bool_ood
#         }
#     }
#
#     # Update the document in the MongoDB collection
#     result = collection.update_one(query, update)
#
#     # Return success message if the document was updated, else throw an error
#     if result.modified_count == 1:
#         return {"status": "success", "message": f"Document with file_name {data.ood_feat_log} updated."}
#     else:
#         raise HTTPException(status_code=400, detail=f"Could not update document with file_name {data.ood_feat_log}.")


class UpdateModel(BaseModel):
    name: str
    bool_ood: bool
    scores_conf: float

@app.post("/update_data/")
async def update_data(update: UpdateModel):
    try:
        filter = {"name": update.name}
        newvalues = {"$set": {"bool_ood": update.bool_ood, "scores_conf": update.scores_conf}}
        collection.update_one(filter, newvalues)
        return {"status": "success"}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_database/")
async def update_database(names: List[str], bool_ood: List[bool], scores_conf: List[Union[float, int]]):
    try:
        for name, b_ood, s_conf in zip(names, bool_ood, scores_conf):
            update_query = {"$set": {"bool_ood": b_ood, "scores_conf": s_conf}}
            collection.update_one({"name": name}, update_query)
        return {"status": "success"}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))
# =================================================================

@app.get("/list_datasets/")
async def list_datasets():
    # base_path = Path(UPLOAD_DIR)
    #
    # # Ensure the base directory exists
    # if not base_path.exists():
    #     raise HTTPException(status_code=404, detail="Upload directory not found!")
    #
    # # List all subdirectories inside the main directory
    # datasets = [subdir.name for subdir in base_path.iterdir() if subdir.is_dir()]

    datasets = collection.distinct("dataset")

    if not datasets:
        raise HTTPException(status_code=404, detail="No datasets found!")

    return {"datasets": datasets}


@app.get("/get_image/{image_name}/")
async def get_image(image_name: str):
    # Load metadata.json
    metadata_file = Path(METADATA_DIR) / "metadata.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found!")

    with metadata_file.open("r") as mf:
        metadata = json.load(mf)

    # Search for the image in metadata
    image_data = next((item for item in metadata if item["name"] == image_name), None)

    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found in metadata!")

    # Construct the image path and check its existence
    image_path = Path(UPLOAD_DIR) / image_data["dataset"] / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found!")

    return FileResponse(image_path)

@app.get("/list_files/{dataset_name}/")
async def list_files(dataset_name: str = FastAPIPath(...)):
    documents = collection.find({"dataset": dataset_name})

    if not documents:
        raise HTTPException(status_code=404, detail="No files found for the specified dataset!")

    # Extract file names from the documents
    files = [doc["file_name"] for doc in documents]

    return {"files": files}


@app.post("/update_results/")
async def update_results(data: List[UpdateData]):
    for item in data:
        # Find the record by file_name and update
        collection.update_one(
            {"file_name": item.file_name},
            {
                "$set": {
                    # "unknown_idx" : item.unknown_idx,
                    "bool_ood": item.bool_ood,
                    "scores_conf": item.scores_conf,
                    # "pred_scores": item.pred_scores,
                    # "pred_labels": item.pred_labels,
                }
            },
        )
    return {"status": "Updated successfully"}



@app.get("/get_ood_images/")
async def get_ood_images():
    ood_images = list(collection.find({"bool_ood": True}, {"_id": 0, "file_path": 1, "file_name": 1, "dataset": 1}))
    return ood_images




class UpdateGroundTruth(BaseModel):
    file_names: List[str]
    class_ground_truth: str
    dataset : str

@app.post("/update_ground_truth/")
async def update_ground_truth(data: UpdateGroundTruth):
    for file_name in data.file_names:
        collection.update_one(
            {"file_name": file_name},
            {"$set": {"class_ground_truth": data.class_ground_truth, "dataset" : data.dataset}}
        )
    return {"status": "Ground truth updated successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")


# =================================================================


# @app.get("/fetch_id_data/{id_name}/{split}/")
# async def fetch_id_data(id_name: str, split: str) -> Dict:
#     # Assuming id_name corresponds to data_name in MongoDB
#     # split corresponds to dataset_split (either 'train' or 'val')
#     fetched_data = list(
#         train_collection.find({"data_name": id_name, "dataset_split": split})) if split == "train" else list(
#         test_collection.find({"data_name": id_name, "dataset_split": split}))
#
#     return {"data": fetched_data}
#
#
# @app.get("/fetch_ood_data/{ood_name}/")
# async def fetch_ood_data(ood_name: str) -> Dict:
#     # Assuming ood_name corresponds to the 'dataset' in the MongoDB collection
#     fetched_data = list(collection.find({"dataset": ood_name}))
#
#     return {"data": fetched_data}



# Fetch in-distribution data
@app.get("/fetch_id_data/")
async def fetch_id_data(data_name: str, dataset_split: str):
    """
    Fetch in-distribution data based on data_name and dataset_split
    :param data_name: Name of the dataset e.g., CIFAR-10
    :param dataset_split: 'train' or 'test' (assuming 'val' means 'test')
    :return: feat_log and label from the database for in-distribution data
    """
    # Choose the right collection based on dataset_split
    if dataset_split == "train":
        collection = train_collection
    elif dataset_split == "val":  # Assuming 'val' means 'test' in your context
        collection = test_collection
    else:
        raise HTTPException(status_code=400, detail=f"Unknown dataset split: {dataset_split}")

    # Query to fetch data
    data = collection.find_one({"data_name": data_name}, {"feat_log": 1, "label": 1, "_id": 0})

    if data:
        return data
    else:
        raise HTTPException(status_code=404, detail=f"No data found for {data_name} in {dataset_split} collection.")


@app.get("/fetch_ood_data/")
async def fetch_ood_data(file_name: str):
    """
    Fetch out-of-distribution data based on file_name
    :param file_name: Name of the file to identify which document to fetch
    :return: ood_feat_log and ood_label from the database for out-of-distribution data
    """
    data = collection.find_one({"file_name": file_name}, {"ood_feat_log": 1, "ood_label": 1, "_id": 0})

    if data:
        return data
    else:
        raise HTTPException(status_code=404, detail=f"No data found for {file_name}.")



@app.get("/get_train_data/")
async def get_train_data():
    try:
        # Fetch all documents from the train_collection
        cursor = train_collection.find({})
        data = list(cursor)  # Convert the cursor to a list
        # Removing ObjectId from the response
        for item in data:
            item["_id"] = str(item["_id"])
        return {"data": data}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_test_data/")
async def get_test_data():
    try:
        # Fetch all documents from the train_collection
        cursor = test_collection.find({})
        data = list(cursor)  # Convert the cursor to a list
        # Removing ObjectId from the response
        for item in data:
            item["_id"] = str(item["_id"])
        return {"data": data}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/get_ood_data/")
async def get_ood_data():
    try:
        # Fetch all documents from the train_collection
        cursor = collection.find({})
        data = list(cursor)  # Convert the cursor to a list
        # Removing ObjectId from the response
        for item in data:
            item["_id"] = str(item["_id"])
        return {"data": data}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======@

# class UpdateTrainTestData(BaseModel):
#     dataset_split: str  # can be "train", "test", or "data"
#     file_name: str  # to identify which document to update
#     feat_log: List[float]
#     label: int
#
# @app.post("/update_train_test_data/")
# async def update_train_test_data(data: UpdateTrainTestData):
#     # Determine the correct collection based on the dataset_split
#     if data.dataset_split == "train":
#         coll = train_collection
#     elif data.dataset_split == "test":
#         coll = test_collection
#     elif data.dataset_split == "data":
#         coll = collection
#     else:
#         raise HTTPException(status_code=400, detail=f"Unknown dataset split: {data.dataset_split}")
#
#     # Define the query and the update
#     query = {"file_name": data.file_name}
#     update = {
#         "$set": {
#             "feat_log": data.feat_log,
#             "label": data.label
#         }
#     }
#
#     result = coll.update_one(query, update)
#
#     if result.modified_count == 1:
#         return {"status": "success", "message": f"Document with file_name {data.file_name} updated in {data.dataset_split} collection."}
#     else:
#         raise HTTPException(status_code=400, detail=f"Could not update document with file_name {data.file_name} in {data.dataset_split} collection.")

class TrainUpdate(BaseModel):
    data_name: str
    feat_log: List[float]
    label: int

class TestUpdate(BaseModel):
    data_name: str
    feat_log: List[float]
    label: int

class DataUpdate(BaseModel):
    file_name: str
    ood_feat_log: List[float]
    ood_label: int


@app.post("/update_train_data/")
async def update_train_data(data: TrainUpdate):
    query = {"data_name": data.data_name}
    update = {
        "$set": {
            "feat_log": data.feat_log,
            "label": data.label
        }
    }
    result = train_collection.update_one(query, update)
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Data not updated, maybe item not found!")
    return {"message": f"Data for {data.data_name} updated successfully in train collection!"}

# For test collection


@app.post("/update_test_data/")
async def update_test_data(data: TestUpdate):
    query = {"data_name": data.data_name}
    update = {
        "$set": {
            "feat_log": data.feat_log,
            "label": data.label
        }
    }
    result = test_collection.update_one(query, update, upsert=True)
    if result.matched_count == 0:
        raise HTTPException(status_code=400, detail=f"Data with name {data.data_name} not found!")
    elif result.modified_count == 0:
        raise HTTPException(status_code=400, detail=f"Data for {data.data_name} not updated. Maybe it's the same data?")
    return {"message": f"Data for {data.data_name} updated successfully in test collection!"}
# For data collection

@app.post("/update_data_collection/")
async def update_data_collection(data: DataUpdate):
    query = {"file_name": data.file_name}
    update = {
        "$set": {
            "ood_feat_log": data.ood_feat_log,
            "ood_label": data.ood_label
        }
    }
    result = collection.update_one(query, update)
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Data not updated, maybe item not found!")
    return {"message": f"Data for {data.file_name} updated successfully in data collection!"}


from pymongo import UpdateOne
@app.post("/update_feature_data/")
async def update_feature_data(data_list: List[FeatureData]):
    # Assume train_collection and test_collection are defined elsewhere
    bulk_ops_train = []
    bulk_ops_val = []
    for data in data_list:
        filter_query = {"data_name": data.data_name, "dataset_split": data.dataset_split}
        update_query = {"$set": data.dict()}
        operation = UpdateOne(filter_query, update_query, upsert=True)
        if data.dataset_split == "train":
            bulk_ops_train.append(operation)
        elif data.dataset_split == "val":
            bulk_ops_val.append(operation)

    if bulk_ops_train:
        train_collection.bulk_write(bulk_ops_train)
    if bulk_ops_val:
        test_collection.bulk_write(bulk_ops_val)

    return {"messages": ["Bulk update executed successfully."]}









@app.get("/get_image_paths/{dataset_name}")
async def get_image_paths(dataset_name: str):
    image_metadata = list(collection.find({"dataset": dataset_name}, {"_id": 0, "file_path": 1}))
    image_paths = [meta["file_path"] for meta in image_metadata]
    return {"image_paths": image_paths}


from fastapi import FastAPI, HTTPException, Query
from typing import Optional
@app.get("/ood_count/")
async def ood_count(dataset: Optional[str] = Query(None, title="Dataset Name")):
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset name is required!")

    # Count the OOD samples for the given dataset
    ood_count = collection.count_documents({"dataset": dataset, "bool_ood": True})
    return {"ood_count": ood_count}


# Visualization, get the embeddings to display
@app.get("/get_id_data/")
async def get_id_data():
    id_data_cursor = train_collection.find()
    id_data = list(id_data_cursor)
    # Assuming the features and labels are stored with keys 'feat_log' and 'label'
    id_feat = [data['feat_log'] for data in id_data]
    id_label = [data['label'] for data in id_data]
    return {"id_feat": id_feat, "id_label": id_label}

@app.get("/get_ood_data/{dataset_name}")
async def get_ood_data(dataset_name: str):
    ood_data_cursor = collection.find({"dataset": dataset_name})  # Change to the correct key
    ood_data = list(ood_data_cursor)
    ood_feat = [data['ood_feat_log'] for data in ood_data if 'ood_feat_log' in data]  # Ensure the key exists
    ood_label = [data['ood_label'] for data in ood_data if 'ood_label' in data]  # Ensure the key exists
    return {"ood_feat": ood_feat, "ood_label": ood_label}




@app.get("/get_image_data/{dataset_name}")
async def get_image_data(dataset_name: str):
    # Query the MongoDB for images belonging to dataset_name
    image_data_cursor = collection.find({"dataset": dataset_name})

    # Convert cursor to list of dictionaries
    image_data_list = list(image_data_cursor)

    if not image_data_list:
        raise HTTPException(status_code=404, detail=f"No image data found for dataset: {dataset_name}")

    # Simplify the structure if needed and remove unnecessary MongoDB fields like '_id'
    image_data_response = [
        {
            "image_path": item["file_path"],
            "target": item.get("ood_label", 0)  # Provide default target if not present
        }
        for item in image_data_list
    ]

    return {"image_data": image_data_response}
