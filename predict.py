import sys
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("modelo_demo.pt", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    probs = F.softmax(output, dim=1)
    pred = torch.argmax(probs, 1).item()
    confidence = probs[0][pred].item()

clases = ["Healthy", "Diseased"]
print(f"Resultado: {clases[pred]} ({confidence*100:.2f}% de confianza)")
