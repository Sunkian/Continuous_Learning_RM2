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
db = client["nine-continuous-learning"]
collection = db["data"]
collection2 = db["id_metadata_collection"]

fs = GridFS(db)

train_collection = db["train"]
val_collection = db["val"]



def check_existence(data_name):
    return train_collection.find_one({'data_name': data_name}) is not None

def insert_to_db(data):
    if data['dataset_split'] == 'train':
        train_collection.insert_one(data)
    elif data['dataset_split'] == 'val':
        val_collection.insert_one(data)

def update_val_db(data):
    existing_data = val_collection.find_one({'data_name': data['data_name']})
    if existing_data:
        val_collection.update_one(
            {'data_name': data['data_name']},
            {'$set': {'updated_label': data['ground_truth_label']}}
        )
    else:
        val_collection.insert_one(data)