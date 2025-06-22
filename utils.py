import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

# Configuración base del generador
latent_dim = 100
num_classes = 10
image_size = 28 * 28

# ---------------------------
# Modelo Generador
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenar ruido + embeddings de las etiquetas
        x = torch.cat((noise, self.label_embedding(labels)), dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# ---------------------------
# Cargar el modelo desde archivo
# ---------------------------
def load_model(path='model/generator.pth'):
    model = Generator()
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------------------
# Generar imágenes desde el modelo
# ---------------------------
def generate_images(model, digit, num_images=5):
    z = torch.randn(num_images, latent_dim)  # ruido aleatorio
    labels = torch.tensor([digit] * num_images)  # etiqueta repetida
    with torch.no_grad():
        images = model(z, labels)
    return images  # Tensor: [B, 1, 28, 28]
