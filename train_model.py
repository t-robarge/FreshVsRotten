# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Suppose we have a custom dataset
from data import FruitDataset, load_data, resnet_transform
from model import MultiHeadResNet50  # Our modular model class

# Example transforms
import torchvision.transforms as T
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

# 1) Create dataset & dataloader
TRAIN_PATH = 'dataset/Train'

# process our data
df = load_data(TRAIN_PATH)
train_dataset = FruitDataset(df, transform=resnet_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2) Initialize model
num_fruits = 9
model = MultiHeadResNet50(num_fruits=num_fruits, freeze_layers=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3) Define Losses & Optimizer
criterion_fruit  = nn.CrossEntropyLoss()
criterion_rotten = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# 4) Training Loop
EPOCHS = 10
alpha = .7
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, fruit_labels, rotten_labels in train_loader:
        images = images.to(device)
        fruit_labels = fruit_labels.to(device)
        rotten_labels = rotten_labels.to(device)

        fruit_out, rotten_out = model(images)

        loss_fruit  = criterion_fruit(fruit_out, fruit_labels)
        loss_rotten = criterion_rotten(rotten_out, rotten_labels)
        loss = alpha*loss_fruit + (1-alpha)*loss_rotten

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device

    correct_fruit, correct_rotten, total = 0, 0, 0
    with torch.no_grad():
        for images, fruit_labels, rotten_labels in loader:
            images = images.to(device)
            fruit_labels = fruit_labels.to(device)
            rotten_labels = rotten_labels.to(device)

            fruit_out, rotten_out = model(images)

            # fruit accuracy
            fruit_preds = torch.argmax(fruit_out, dim=1)
            correct_fruit += (fruit_preds == fruit_labels).sum().item()

            # rotten/fresh accuracy
            rotten_preds = torch.argmax(rotten_out, dim=1)
            correct_rotten += (rotten_preds == rotten_labels).sum().item()

            total += images.size(0)

    return correct_fruit / total, correct_rotten / total

fruit_acc, rotten_acc = evaluate(model, train_loader)
print(f"Fruit Type Accuracy:  {fruit_acc*100:.2f}%")
print(f"Fresh/Rotten Accuracy: {rotten_acc*100:.2f}%")
