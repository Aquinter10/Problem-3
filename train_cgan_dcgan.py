import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import os

# ---------------------------
# Hiperparámetros
# ---------------------------
latent_dim = 100
num_classes = 10
img_size = 28
channels = 1
batch_size = 128
lr = 0.0002
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Prepara carpeta para guardar imágenes
# ---------------------------
os.makedirs("samples", exist_ok=True)

# ---------------------------
# Dataset MNIST
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Para que coincida con Tanh
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# Generador con convoluciones
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 10)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(110, 128, 3, 1, 0),  # input: z (100) + label (10) → 110
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_embed(labels)
        x = torch.cat((z, label_input), dim=1).unsqueeze(2).unsqueeze(3)
        return self.model(x)

# ---------------------------
# Discriminador con convoluciones
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 10)

        self.model = nn.Sequential(
            nn.Conv2d(1 + 1, 64, 4, 2, 1),  # imagen + etiqueta
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_embed(labels).unsqueeze(2).unsqueeze(3)  # [B, 10, 1, 1]
        label_map = label_map.expand(-1, 1, img_size, img_size)  # [B, 1, 28, 28]
        x = torch.cat((img, label_map), dim=1)
        return self.model(x)

# ---------------------------
# Inicializa modelos y optimizadores
# ---------------------------
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ---------------------------
# Entrenamiento
# ---------------------------
for epoch in range(1, epochs + 1):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        # Entrena Generador
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = G(z, gen_labels)

        optimizer_G.zero_grad()
        g_loss = criterion(D(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Entrena Discriminador
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = criterion(D(imgs, labels), valid)
        fake_loss = criterion(D(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # ---------------------
    # Imprime y guarda imágenes
    # ---------------------
    print(f"Epoch [{epoch}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    if epoch % 10 == 0:
        z = torch.randn(10, latent_dim).to(device)
        fixed_labels = torch.arange(0, 10).to(device)
        with torch.no_grad():
            sample_imgs = G(z, fixed_labels)
        grid = make_grid(sample_imgs, nrow=10, normalize=True)
        save_image(grid, f"samples/epoch_{epoch}.png")

# ---------------------------
# Guarda el modelo
# ---------------------------
torch.save(G.state_dict(), "generator_dcgan.pth")
print("✔ Modelo guardado como 'generator_dcgan.pth'")
