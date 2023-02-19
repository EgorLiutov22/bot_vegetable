import io
import uvicorn
from fastapi import FastAPI, File, UploadFile
from nnet import Network
from PIL import Image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_file(file: UploadFile = File(...)):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    net = Network(img)
    v_name = net.prediction()
    return {"vegetable": v_name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

