from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import json
from models import ResNet18

app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('best_checkpoint.tar', map_location=device)
model = ResNet18().to(device)
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
    1: "Happy",
    2: "Sad",
    3: "Neutral"
}

async def video_to_frames(video_bytes, timestamps: List[List[float]]):
    """Extract frames from video bytes within given timestamps."""
    frames = []
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file.flush()
        cap = cv2.VideoCapture(tmp_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for start, end in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)  # Start time in milliseconds
            while cap.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000:  # End time in milliseconds
                success, frame = cap.read()
                if not success:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    return frames


@app.post("/recognize/")
async def recognize_video(file: UploadFile = File(...), chunks_json: str = Form(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse(content={"error": "This API supports only video files."}, status_code=400)

    try:
        chunks_data = json.loads(chunks_json)
        timestamps = [chunk["timestamp"] for chunk in chunks_data["chunks"]]
    except (json.JSONDecodeError, KeyError):
        return JSONResponse(content={"error": "Invalid or missing JSON data."}, status_code=400)

    video_bytes = await file.read()
    frames = await video_to_frames(video_bytes, timestamps)

    # Initialize a dictionary to hold the sum of probabilities for each emotion
    sum_probabilities = {emotion: 0.0 for emotion in imagenet_classes.values()}
    frame_count = 0

    for frame in frames:
        image = Image.fromarray(frame)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Add probabilities to the corresponding emotion
        for idx, emotion in imagenet_classes.items():
            sum_probabilities[emotion] += probabilities[idx].item()

        frame_count += 1

    # If there were no frames processed, avoid division by zero
    if frame_count == 0:
        avg_probabilities = {emotion: 0.0 for emotion in imagenet_classes.values()}
    else:
        # Calculate the average probability for each emotion
        avg_probabilities = {emotion: total_prob / frame_count for emotion, total_prob in sum_probabilities.items()}

    return {"predictions": avg_probabilities}
