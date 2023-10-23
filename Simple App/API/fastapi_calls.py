from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

app = FastAPI()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["mycollection"]

class TextItem(BaseModel):
    text: str

@app.post("/add_text/")
async def add_text(text_item: TextItem):
    new_text = {"text": text_item.text}
    result = collection.insert_one(new_text)
    return {"message": "Text added successfully", "id": str(result.inserted_id)}

@app.get("/get_text/")
async def get_text():
    texts = [item["text"] for item in collection.find()]
    return {"texts": texts}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
