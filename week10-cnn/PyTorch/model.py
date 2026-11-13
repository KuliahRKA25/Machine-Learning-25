"""Model terbaik dari 2_DL_5054251001.ipynb (CNN + BatchNorm + Dropout) beserta fungsi prediksinya."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).parent
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class CNNBatchNormDropout(nn.Module):
    # Harus sama persis dengan definisi di notebook agar state_dict bisa dimuat.
    def __init__(self, p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path=HERE / "model.pth"):
    model = CNNBatchNormDropout()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()  # eval(): BatchNorm dan Dropout dalam mode inferensi


def preprocess(image: Image.Image) -> torch.Tensor:
    """Samakan gambar dengan format training: grayscale 28x28, nilai 0-1, objek terang di latar gelap."""
    # shortcut: gambar langsung di-resize ke 28x28 tanpa menjaga rasio atau memusatkan objek,
    # ganti dengan crop + padding jika dipakai untuk foto yang tidak persegi.
    gray = np.asarray(image.convert("L").resize((28, 28)), dtype=np.float32) / 255.0
    # Fashion-MNIST berlatar hitam, sedangkan foto produk biasanya berlatar terang: balik warnanya.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 0.5:
        gray = 1.0 - gray
    return torch.from_numpy(gray).view(1, 1, 28, 28)


def predict(model, image: Image.Image) -> dict[str, float]:
    with torch.no_grad():
        proba = torch.softmax(model(preprocess(image)), dim=1)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, proba)}


if __name__ == "__main__":
    # Self-check: examples/<label>.png diambil dari test set dan harus diprediksi sebagai kelas <label>.
    model = load_model()
    files = sorted((HERE / "examples").glob("*.png"))
    assert len(files) == 10, files
    for f in files:
        proba = predict(model, Image.open(f))
        top = max(proba, key=proba.get)
        assert top == CLASS_NAMES[int(f.stem)], (f.name, top)
        print(f"{f.name}: {top} ({proba[top]:.3f})")
    # Gambar yang sama dengan latar terang harus tetap dikenali (cek pembalikan warna).
    inverted = Image.eval(Image.open(files[1]).convert("L"), lambda v: 255 - v)
    assert max(p := predict(model, inverted), key=p.get) == CLASS_NAMES[1]
    print("ok")
