import base64
import datetime
import json
import pickle
import shutil

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Path as FastAPIPath
from typing import Dict, Union
from pathlib import Path
import os
from pydantic import BaseModel
from pymongo import MongoClient
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

    return {"message": f"Data for {data.data_name} inserted successfully into {data.dataset_split} collection!"}


class OODUpdate(BaseModel):
    file_name: str  # to identify which document to update
    ood_feat_log: List[float]  # since it seems like a list of floats from your function
    ood_label: int


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


# =================================================================

@app.get("/list_datasets/")
async def list_datasets():
    base_path = Path(UPLOAD_DIR)

    # Ensure the base directory exists
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Upload directory not found!")

    # List all subdirectories inside the main directory
    datasets = [subdir.name for subdir in base_path.iterdir() if subdir.is_dir()]

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
    dataset_path = Path(UPLOAD_DIR) / dataset_name

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found!")

    # List all files inside the selected dataset
    files = [f.name for f in dataset_path.iterdir() if f.is_file()]

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
    ood_images = list(collection.find({"bool_ood": True}, {"_id": 0, "file_path": 1, "file_name": 1}))
    return ood_images




class UpdateGroundTruth(BaseModel):
    file_names: List[str]
    class_ground_truth: str

@app.post("/update_ground_truth/")
async def update_ground_truth(data: UpdateGroundTruth):
    for file_name in data.file_names:
        collection.update_one(
            {"file_name": file_name},
            {"$set": {"class_ground_truth": data.class_ground_truth}}
        )
    return {"status": "Ground truth updated successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
