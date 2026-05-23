"""
Script de entrenamiento — diseñado para correr en Google Colab con GPU T4.
Uso: python -m src.image_classifier.train
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.image_classifier.model import build_model
from src.image_classifier.preprocess import preprocess_for_training, preprocess_for_inference
from src.utils.config import DATA_DIR, IMAGE_CLASSIFIER_MODEL_PATH, DEVICE, SEED
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def train(
    data_dir=DATA_DIR / "processed" / "fashion_products",
    epochs_head: int = 10,
    epochs_full: int = 10,
    lr_head: float = 1e-3,
    lr_full: float = 1e-4,
    batch_size: int = 32,
) -> None:
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)

    train_ds = ImageFolder(data_dir / "train", transform=preprocess_for_training)
    val_ds = ImageFolder(data_dir / "val", transform=preprocess_for_inference)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    num_classes = len(train_ds.classes)
    model = build_model(num_classes=num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Fase 1 — entrenar solo la cabeza
    logger.info("Fase 1: entrenando la cabeza clasificadora (%d épocas)", epochs_head)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_head)
    _run_epochs(model, train_loader, val_loader, criterion, optimizer, epochs_head, device)

    # Fase 2 — fine-tuning completo
    logger.info("Fase 2: fine-tuning completo (%d épocas)", epochs_full)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_full)
    _run_epochs(model, train_loader, val_loader, criterion, optimizer, epochs_full, device)

    IMAGE_CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), IMAGE_CLASSIFIER_MODEL_PATH)
    logger.info("Modelo guardado en %s", IMAGE_CLASSIFIER_MODEL_PATH)


def _run_epochs(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Época {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = _evaluate(model, val_loader, device)
        logger.info("Época %d | Loss: %.4f | Val Acc: %.2f%%", epoch, total_loss / len(train_loader), val_acc * 100)


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    train()
