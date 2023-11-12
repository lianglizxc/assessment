# main.py
from fastapi import UploadFile, FastAPI
from models.efficientNet import Inference

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile):
    try:
        contents = file.file.read()
        file_name = 'test_image.jpg'
        with open('test_image.jpg', 'wb') as f:
            f.write(contents)
        inference = Inference("model_config.json")
        inference.load_latest_checkpoint("efficientNet_ckpt")
        pred = inference.prediction(file_name)
        if isinstance(pred, list):
            pred = pred[0]
    except Exception as e:
        return {"prediction": "There was an error when the file",
                "error": str(e)}
    finally:
        file.file.close()
    return {"prediction": pred}