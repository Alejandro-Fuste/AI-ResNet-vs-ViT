import argparse, torch, torch.nn as nn, torch.optim as optim
from torchvision.models import resnet18, vit_b_16
from .datasets import build_loaders

def get_model(model_name, num_classes, weights=None):
    if model_name == "resnet18":
        m = resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif model_name == "vit_b_16":
        m = vit_b_16(weights=weights)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("model_name must be resnet18 or vit_b_16")
    return m

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
    return running / len(loader.dataset), correct / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet18","vit_b_16"], required=True)
    ap.add_argument("--data",  default="data")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

  
    loaders = build_loaders(args.data, model_name=args.model, batch_size=args.batch_size)
    num_classes = len(loaders["class_to_idx"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        best_acc = max(best_acc, val_acc)

    print(f"Best val acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()