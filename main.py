from fastapi import FastAPI, File, UploadFile
import os
from fastapi.middleware.cors import CORSMiddleware
from predict import evaluate


current_dir = os.path.dirname(__file__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

@app.get("/")
async def hello_world():
    return {"hello": "world !"}

@app.post("/evaluate")
async def predict(file: UploadFile = File(...)):
    evaluate()