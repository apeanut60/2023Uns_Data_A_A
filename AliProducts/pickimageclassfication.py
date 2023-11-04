import torch
from torchvision import transforms
from PIL import Image

model = torch.load('checkpoints/resnest/24_ResNet101.pt')

'''preprocess = transforms.Compose([
   transforms.Resize(224),
   transforms.ToTensor(),
   transforms.Normalize(mean, std)
])'''

image = Image.open('/mnt/e/dataset/pokemom/images/snover.png')
input = image
#input = input.unsqueeze(0) 

with torch.no_grad():
   output = model(input)

pred_class = torch.argmax(output, dim=1).item()
print(f'Predicted class is: {pred_class}')