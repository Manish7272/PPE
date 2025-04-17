import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os

# Load your model once
model = torch.load(r"runs\detect\train\weights\best.pt", map_location=torch.device('cpu'))
model.eval()

# Preprocessing for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = predicted.item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][label].item()
    return {"label": str(label), "confidence": round(confidence, 3)}

def process_video(video_path):
    # Dummy video processing – just copying for now
    output_path = video_path.replace(".mp4", "_processed.mp4")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Here you can apply frame-by-frame processing using your model
        # frame = your_model.process_frame(frame)
        out.write(frame)
    cap.release()
    out.release()

    # Dummy result
    return output_path, {"video_analysis": "Processed successfully"}
