from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import shutil
import tempfile
import os
from moviepy.editor import VideoFileClip
import json
import torch
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from PIL import Image
import cv2
from typing import List
from models import ResNet18  # Ensure this correctly points to your model definition

app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('best_checkpoint.tar', map_location=device)
model = ResNet18().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transformation for the model input
transform = Compose([
    Grayscale(),
    Resize((48, 48)),
    ToTensor(),
    Normalize(mean=(0,), std=(255,))
])

# Emotion classes
imagenet_classes = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: "Neutral"
}

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def chop_video(video_path, start, end):
    """Chop the video based on start and end timestamps."""
    segment_path = tempfile.mktemp(suffix=".mp4")  # Temporary path for the segment
    clip = VideoFileClip(video_path).subclip(start, end)
    clip.write_videofile(segment_path, codec="libx264", audio_codec="aac")
    clip.close()
    return segment_path


def process_video_segment(segment_path):
    """Process each video segment for emotion recognition."""
    cap = cv2.VideoCapture(segment_path)
    emotion_probabilities = {emotion: [] for emotion in imagenet_classes.values()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_frame = cv2.resize(frame[y:y + h, x:x + w], (48, 48))
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            face_tensor = transform(Image.fromarray(face_frame))
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                for i, emotion in enumerate(imagenet_classes.values()):
                    emotion_probabilities[emotion].append(probabilities[i].item())

    cap.release()
    # Average the probabilities for each emotion
    averages = {emotion: sum(probs) / len(probs) if probs else 0 for emotion, probs in emotion_probabilities.items()}
    return averages


@app.post("/recognize/")
async def recognize_video(file: UploadFile = File(...), timestamps: str = Form(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="This API supports only video files.")

    timestamps = json.loads(timestamps)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        video_path = temp_video.name

    results = []
    for ts in timestamps:
        segment_path = chop_video(video_path, ts["start"], ts["end"])
        emotion_result = process_video_segment(segment_path)
        results.append({
            "time_range": f"{ts['start']}s to {ts['end']}s",
            "emotion_analysis": emotion_result
        })
        os.remove(segment_path)  # Cleanup

#    os.remove(video_path)  # Cleanup the original video file

    return results
