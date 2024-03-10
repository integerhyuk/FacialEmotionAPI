import torch
import torchvision.transforms as transforms
from PIL import Image
from models import ResNet18
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

# Load your model
checkpoint = torch.load('best_checkpoint.tar', map_location=torch.device('cpu'))
model = ResNet18()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,), std=(255,))
])

imagenet_classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def video_to_frames(video_bytes):
    """Extract frames from video bytes by writing to a temporary file."""
    # Create a temporary file and write the video bytes to it
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_name = tmp_file.name

    # Now use OpenCV to open the temporary file
    cap = cv2.VideoCapture(tmp_file_name)
    frames = []
    success, frame = cap.read()
    while success:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        frames.append(rgb_frame)
        success, frame = cap.read()

    # Release the capture and close the file
    cap.release()

    # Remove the temporary file
    os.remove(tmp_file_name)

    return frames

@app.post("/recognize/")
async def recognize_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse(content={"error": "This API supports only video files."}, status_code=400)

    video_bytes = await file.read()
    frames = video_to_frames(video_bytes)

    aggregated_predictions = {emotion: 0 for emotion in imagenet_classes.values()}

    for frame in frames:
        image = Image.fromarray(frame)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)  # Get the top prediction
        emotion = imagenet_classes[top_catid.item()]
        aggregated_predictions[emotion] += 1

    return {"predictions": aggregated_predictions}
