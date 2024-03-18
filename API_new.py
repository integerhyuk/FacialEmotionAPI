import torch
import torchvision.transforms as transforms
from PIL import Image
from models import ResNet18
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import tempfile
import os
from collections import defaultdict

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
    1: "Happy",
    2: "Sad",
    3: "Neutral"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def video_to_frames_with_timestamps(video_bytes, sampling_rate=1):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_name = tmp_file.name

    cap = cv2.VideoCapture(tmp_file_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_and_timestamps = []
    frame_id = 0
    success, frame = cap.read()
    while success:
        if frame_id % sampling_rate == 0:
            timestamp = frame_id / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_and_timestamps.append((timestamp, rgb_frame))
        success, frame = cap.read()
        frame_id += 1

    cap.release()
    os.remove(tmp_file_name)
    return frames_and_timestamps


def process_frames_and_calculate_averages(frames_and_timestamps, num_groups=4):

    total_frames = len(frames_and_timestamps)
    frames_per_group = total_frames // num_groups
    grouped_averages = []

    # Calculate the duration of the video in seconds
    total_duration = frames_and_timestamps[-1][0] if frames_and_timestamps else 0
    group_duration = total_duration / num_groups

    for i in range(num_groups):
        start_time = i * group_duration
        end_time = (i + 1) * group_duration if (i < num_groups - 1) else total_duration
        group_frames = frames_and_timestamps[i * frames_per_group:(i + 1) * frames_per_group]
        emotion_probabilities = defaultdict(list)

        for timestamp, frame in group_frames:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_frame = frame[y:y + h, x:x + w]
                image = Image.fromarray(face_frame)
                input_tensor = transform(image)
                input_batch = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    output = model(input_batch)

                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                for idx, emotion in enumerate(imagenet_classes.values()):
                    emotion_probabilities[emotion].append(probabilities[idx].item())

        averages = {emotion: sum(probs) / len(probs) if probs else 0 for emotion, probs in
                    emotion_probabilities.items()}
        grouped_averages.append({
            "time_range": f"{start_time:.2f}s - {end_time:.2f}s",
            "averages": averages
        })

    return grouped_averages

def detect_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


@app.post("/recognize/")
async def recognize_video(file: UploadFile = File(...), sampling_rate: int = 1):

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="This API supports only video files.")

    video_bytes = await file.read()
    frames_and_timestamps = video_to_frames_with_timestamps(video_bytes, sampling_rate)
    grouped_averages_with_time = process_frames_and_calculate_averages(frames_and_timestamps)

    return {"grouped_averages_with_time": grouped_averages_with_time}