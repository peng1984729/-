import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import config
from model import Mymodel
from preprocess import get_dataloaders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train_loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train_acc")
    axes[1].plot(epochs, history["val_acc"], marker="s", label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass

def train():
    train_loader, val_loader, test_loader, classes, num_classes = get_dataloaders()
    model = Mymodel(num_classes=num_classes).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()                       
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.STEP_SIZE,
                                                gamma=config.GAMMA) 

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()             
            outputs = model(images)            
            loss = criterion(outputs, labels)  
            loss.backward()                   
            optimizer.step()                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(config.MODEL_DIR, "best.pth"))

    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "final.pth"))
    with open(os.path.join(config.MODEL_DIR, "classes.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(classes))
    plot_history(history, os.path.join(config.MODEL_DIR, "training_curve.png"))
    return history


if __name__ == "__main__":
    train()
