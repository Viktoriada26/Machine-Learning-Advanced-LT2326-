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
test_file = args.test_file

unique_labels = extract_unique_labels(args.train_file)  
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = ThaiEngOCRDataset(test_file, label_to_index, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ThaiEngOCRModel(num_classes=num_classes)
model.load_state_dict(torch.load("./ThaiEng_model.pth", weights_only=True))
model.eval()

def evaluate_model(loader, model):
    device = torch.device("cuda")
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

test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(test_loader, model)

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1-Score: {test_f1}")
