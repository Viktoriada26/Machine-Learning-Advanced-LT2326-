import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def extract_unique_labels(data_file):
    unique_labels = []
    with open(data_file, 'r') as f:
        for line in f:
            img_path = line.strip()
            label = img_path.split('/')[-4]  # Extract label from path structure

            if label not in unique_labels:
                unique_labels.append(label)

    return sorted(unique_labels)  # Ensure consistent indexing

class ThaiEngOCRDataset(Dataset):
    def __init__(self, data_file, label_to_index, transform=None):
        self.file_paths = []
        self.labels = []
        self.transform = transform
        self.label_to_index = label_to_index
        valid_extension = ('.bmp')

        with open(data_file, 'r') as f:
            for line in f:
                img_path = line.strip()  
                
                if img_path.lower().endswith(valid_extension):
                    self.file_paths.append(img_path)
                    label = self.extract_label_from_path(img_path)
                    self.labels.append(self.label_to_index[label])

    def extract_label_from_path(self, img_path):
        path_parts = img_path.split('/')
        return path_parts[-4]  

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path)  # Load grayscale image

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label
