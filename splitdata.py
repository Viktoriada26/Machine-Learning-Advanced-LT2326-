import os
import argparse
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser(description="Split data into training, validation, and test datasets.")
    
    parser.add_argument('train_dataset', type=str, help="Dataset for training ( 'Thai_normal_alldpi', 'Thai_normal_200dpi', 'Thai_normal_400dpi', 'Thai_bold_alldpi', 'All_Thai_styles', 'Thai_English_normal', 'Thai_bold_400dpi', 'Thai_bold_italic', 'All_Thai_English_styles')")
    parser.add_argument('val_dataset', type=str, help="Dataset for validation ( 'Thai_normal_alldpi', 'Thai_normal_200dpi', 'Thai_normal_400dpi', 'Thai_bold_alldpi', 'All_Thai_styles', 'Thai_English_normal', 'Thai_bold_400dpi', 'Thai_bold_italic', 'All_Thai_English_styles')")
    parser.add_argument('test_dataset', type=str, help="Dataset for testing ( 'Thai_normal_alldpi', 'Thai_normal_200dpi', 'Thai_normal_400dpi', 'Thai_bold_alldpi', 'All_Thai_styles', 'Thai_English_normal', 'Thai_bold_400dpi', 'Thai_bold_italic', 'All_Thai_English_styles')")
    
    parser.add_argument('directory', type=str, help="The path of the dataset (e.g., /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet).")
    
    return parser.parse_args()

# match images based on language, style, and dpi
def match_images(path, language=None, style=None, dpi=None, ignored_language=None):

    match_files = []
    
    if not os.path.exists(os.path.join(path, language)):
        print(f"Directory {os.path.join(path, language)} does not exist.")
        return match_files

    for root, dirs, files in os.walk(os.path.join(path, language)):
        if (language is None or language in root) and (dpi is None or str(dpi) in root):
            if ignored_language is None or f'/{ignored_language}/' not in root:
                if style is None:
                    for file in files:
                        match_files.append(os.path.join(root, file))
                elif style == "bold" and "bold_italic" not in root and "bold" in root:
                    for file in files:
                        match_files.append(os.path.join(root, file))
                elif style == "bold_italic" and "bold_italic" in root:
                    for file in files:
                        match_files.append(os.path.join(root, file))
                elif style == "normal" and "normal" in root:
                    for file in files:
                        match_files.append(os.path.join(root, file))
    
    return match_files

# split data into training, validation, and test sets
def split_train_val_test(data, test_size=0.1, validation_size=0.1):
    testvalid_size = test_size + validation_size
    training_data, testvalid_data = train_test_split(data, test_size=testvalid_size, random_state=42)
    validation_data, test_data = train_test_split(testvalid_data, test_size=0.5, random_state=42)
    
    return training_data, validation_data, test_data


def split_data(path):
    data = {}

    data['Thai_normal_alldpi'] = (
        match_images(path, language="Thai", style="normal", dpi=200, ignored_language="English") + 
        match_images(path, language="Thai", style="normal", dpi=400, ignored_language="English")
    )

    data['Thai_normal_200dpi'] = match_images(path, language="Thai", style="normal", dpi=200, ignored_language="English")
    
    data['Thai_normal_400dpi'] = match_images(path, language="Thai", style="normal", dpi=400, ignored_language="English")

    data['Thai_bold_alldpi'] = (
        match_images(path, language="Thai", style="bold", dpi=400, ignored_language="English") + 
        match_images(path, language="Thai", style="bold", dpi=200, ignored_language="English")
    )

    data['All_Thai_styles'] = match_images(path, language="Thai", ignored_language="English")
    
    data['Thai_English_normal'] = (
        match_images(path, language="Thai", style="normal") + 
        match_images(path, language="English", style="normal")
    ) 

    data['Thai_bold_400dpi'] = match_images(path, language="Thai", style="bold", dpi=400, ignored_language="English")
    
    data['Thai_bold_italic'] = match_images(path, language="Thai", style="bold_italic", dpi=400, ignored_language="English")
    
    data['All_Thai_English_styles'] = (
        match_images(path, language="Thai") + 
        match_images(path, language="English")
    )

    train_val_test_data = {}
    for key, files in data.items():
        training_data, validation_data, test_data = split_train_val_test(files)
        train_val_test_data[key] = {
            'train': training_data,
            'val': validation_data,
            'test': test_data
        }
    
    return train_val_test_data


def save_to_file(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(item + "\n")


if __name__ == "__main__":
    args = get_args()

    train_val_test_data = split_data(args.directory)

    if args.train_dataset in train_val_test_data:
        save_to_file(os.path.join(os.getcwd(), "training_data.txt"), train_val_test_data[args.train_dataset]['train'])
    else:
        print(f"No data found for training dataset: {args.train_dataset}")

    if args.val_dataset in train_val_test_data:
        save_to_file(os.path.join(os.getcwd(), "validation_data.txt"), train_val_test_data[args.val_dataset]['val'])
    else:
        print(f"No data found for validation dataset: {args.val_dataset}")

    if args.test_dataset in train_val_test_data:
        save_to_file(os.path.join(os.getcwd(), "test_data.txt"), train_val_test_data[args.test_dataset]['test'])
    else:
        print(f"No data found for test dataset: {args.test_dataset}")

    print("Data saved successfully.")
