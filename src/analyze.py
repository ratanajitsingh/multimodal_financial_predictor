import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import FinancialDataset
from model import MultimodalPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

#config
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
#hard pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_epoch_10.pth")

def load_test_data():
    full_data = FinancialDataset(
        news_path = os.path.join(PROJECT_ROOT, 'data/Combined_News_DJIA.csv'),
        prices_path = os.path.join(PROJECT_ROOT, 'data/upload_DJIA_table.csv')
    )

    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size

    generator = torch.Generator().manual_seed(42)
    _, test_data = random_split(full_data, [train_size, test_size], generator=generator)

    return DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=False)

def get_prediction(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            prices = batch['price_features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, mask, prices)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            confidence_tensor = probs.gather(1, preds.unsqueeze(1)).squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(confidence_tensor.cpu().numpy())


    return all_labels, all_preds, all_probs

def plot_confusion_matrix(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize = (8,6))
    sns.heatmap(cm, annot = True, fmt='d',cmap='Blues', xticklabels=['Down','Up'], yticklabels=['Down','Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix: Understanding where the model fails')
    save_path = os.path.join(PROJECT_ROOT, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.show()

def plot_confidence_dist(probs):
    plt.figure(figsize = (8,6))
    plt.hist(probs, bins = 20, color='purple', alpha=0.7)
    plt.xlabel('Model Confidence (0.5 - 1.0)')
    plt.ylabel('Number of samples')
    plt.title('Confidence Distribution: How sure is the model')
    plt.axvline(x=0.5, color = 'red', linestyle = '--', label = 'Guessing')
    plt.legend()

    save_path = os.path.join(PROJECT_ROOT, 'confidence_distribution.png')
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    test_loader = load_test_data()

    model = MultimodalPredictor(num_numerical_features=5).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Model not found, quitting")
        exit()

    y_true, y_pred, y_probs = get_prediction(model, test_loader)

    print("Classification report")
    print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))

    plot_confusion_matrix(y_true, y_pred)
    plot_confidence_dist(y_probs)
