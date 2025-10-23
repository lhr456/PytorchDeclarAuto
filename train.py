import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TensorTransformSafe

def train_demo(epochs=5, batch_size=128, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📦 使用设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 模型
    model = TensorTransformSafe(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ====== 训练 ======
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"🧩 Epoch {epoch+1} 平均损失: {total_loss / len(trainloader):.4f}")

        # ====== 验证 ======
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"✅ 测试准确率: {acc:.2f}%")

if __name__ == "__main__":
    train_demo(epochs=3)
