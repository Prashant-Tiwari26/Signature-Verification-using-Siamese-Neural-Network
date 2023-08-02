from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from PIL.ImageOps import invert

transform = Compose([
    invert(),
    ToTensor(),
    Resize((155,220), interpolation=InterpolationMode.BICUBIC),
    Normalize(mean=0.5, std=0.5)
])