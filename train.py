import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
from dataset import extract_unique_labels, ThaiEngOCRDataset
from model import ThaiEngOCRModel
from args import get_args

# Parse arguments
args = get_args()
batch_size = args.batch_size
epochs = args.epochs
train_file = args.train_file

# Extract unique labels from the training data
unique_labels = extract_unique_labels(train_file)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

# Dataset and DataLoader for training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ThaiEngOCRDataset(train_file, label_to_index, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model, optimizer, and loss function
model = ThaiEngOCRModel(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Training function
def train_model(train_loader, model, criterion, optimizer, num_epochs=5):
    device = torch.device("cuda") #instead of cpu
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0

        for inputs, labels in tqdm.tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Training Loss after epoch {epoch + 1}: {train_loss}")

    return model

# Train the model
trained_model = train_model(train_loader, model, criterion, optimizer, num_epochs=epochs)

# Save the trained model
save_path = "./ThaiEng_model.pth"
torch.save(trained_model.state_dict(), save_path)
print(f"Model saved to {save_path}")
