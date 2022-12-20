from PIL import Image
import torchvision.transforms as transforms

f = './data/0.jpg'

transform = transforms.Compose([
    transforms.Resize([128, 128,]),  # [h,w]
    transforms.ToTensor(),
])

img = Image.open(f)
print(f)
img = transform(img).numpy()

print(img.min())