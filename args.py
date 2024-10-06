import argparse

def get_args():

    parser = argparse.ArgumentParser(description="Train, validate, and test a model")
    parser.add_argument('--train_file', type=str, default='training_data.txt', help='Path to the training dataset file.')
    parser.add_argument('--val_file', type=str, default='validation_data.txt', help='Path to the validation dataset file.')
    parser.add_argument('--test_file', type=str, default='test_data.txt', help='Path to the test dataset file.')
    
    # Model and training parameters
    parser.add_argument('--model_path', type=str, default='./ThaiEng_model.pth', help='Path to save/load the model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')

    return parser.parse_args()


