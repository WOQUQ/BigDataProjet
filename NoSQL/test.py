import pymongo
import pandas as pd
import json

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["mydatabase"]
name="CompustatName"
collection=db[name]
CompustatName = pd.read_csv("train.csv")
data = json.loads(CompustatName.to_json(orient="records"))
collection.insert(data)