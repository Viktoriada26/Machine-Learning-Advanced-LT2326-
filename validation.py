import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import tqdm
from dataset import extract_unique_labels, ThaiEngOCRDataset
from model import ThaiEngOCRModel
from args import get_args

args = get_args()
batch_size = args.batch_size
val_file = args.val_file

unique_labels = extract_unique_labels(args.train_file)  
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_dataset = ThaiEngOCRDataset(val_file, label_to_index, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ThaiEngOCRModel(num_classes=num_classes)
model.load_state_dict(torch.load("./ThaiEng_model.pth", weights_only=True))
model.eval()



def evaluate_model(loader, model):
    device = torch.device("cuda") #instead of cpu
    model.to(device)

    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predlab = outputs.argmax(dim=1).cpu().numpy()
            predicted_labels.extend(predlab)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    return accuracy, precision, recall, f1

val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(val_loader, model)

print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Precision: {val_precision}")
print(f"Validation Recall: {val_recall}")
print(f"Validation F1-Score: {val_f1}")
