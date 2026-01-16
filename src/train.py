import json

def main():
    print("Training script placeholder. Everything is set up correctly!")


if __name__ == "__main__":
    main()


from torch.utils.data import DataLoader
from src.split_dataset import get_datasets

def get_loaders(batch_size=16):
    train_ds, val_ds = get_datasets()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader


import torch
import torch.nn as nn
import torch.optim as optim

from src.model import create_model
from src.train import get_loaders   # adjust if needed


def train(num_epochs=5, lr=1e-4):
    
    train_loader, val_loader = get_loaders()
    

    num_classes = len(train_loader.dataset.dataset.class_to_idx)
    # save class mapping so inference/streamlit uses the same labels
    class_to_idx = train_loader.dataset.dataset.class_to_idx
    with open("class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)



    model = create_model(num_classes)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()

