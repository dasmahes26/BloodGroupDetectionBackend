from fastapi import FastAPI, File, UploadFile 
from fastapi.middleware.cors import CORSMiddleware 
from PIL import Image 
import io
import keras
import numpy as np

'''
image_h = 224
image_w = 224
image_url = ""
'''

app = FastAPI()
# Allow Next.js to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("API working....")

model = keras.layers.TFSMLayer(
    "model_saved",
    call_endpoint="serving_default"
)
print("Loaded....")
class_names = ["A+","A-","AB+","AB-","B+","B-","O+","O-"]





@app.get('/')
def hello():
    print("Hello")
    return "Blood group predcition"

@app.get('/home')
def hello():
    return "Blood group"

'''@app.get('/predict')
def mypredict():
    print ("api hit")
    return {"message":"It is under test."}'''

@app.post('/predict')
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"message": "Invalid Image File"}
    
    # Resize
    img = img.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Normalize (0–255 → 0–1)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    result = model(img_array)
    # extract probabilities safely
    probs = list(result.values())[0].numpy()

    # get index of highest probability
    pred_index = np.argmax(probs)

    # map to class name
    pred_class = class_names[pred_index]

    print("Predicted class:", pred_class)

    return {
    "message": f"Blood Group is {pred_class}"
    }