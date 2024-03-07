import torch
import torchvision.transforms as transforms
from PIL import Image
from models import ResNet18
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io

app = FastAPI()

# Load your model
checkpoint = torch.load('best_checkpoint.tar')
model = ResNet18()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(255,))
        #transforms.TenCrop(40),
        #transforms.Lambda(lambda crops: torch.stack(
            #[transforms.ToTensor()(crop) for crop in crops])),
        #transforms.Lambda(lambda tensors: torch.stack(
            #[transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
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

@app.post("/recognize/")
async def recognize_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "This API supports only image files."}, status_code=400)

    # Load the image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Transform the image to be compatible with the model
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 categories predicted by the model
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    categories = [imagenet_classes[catid.item()] for catid in top5_catid]  # Assume `imagenet_classes` is a list or dict of ImageNet class names

    # Return the top 5 predictions as JSON
    result = {"predictions": dict(zip(categories, top5_prob.tolist()))}
    return result


# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=5000)
