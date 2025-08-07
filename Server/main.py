from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from inference import generate_caption

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Receive an image file and return a generated caption."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        caption = generate_caption(image)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}

# Run the server: uvicorn main:app --host 0.0.0.0 --port 8000
