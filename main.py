from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import models, transforms

app = FastAPI()

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("modelo_demo.pt", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

clases = ["Healthy", "Diseased"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1).item()

    return {"prediction": clases[pred]}
