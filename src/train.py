import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from dataset import FinancialDataset
from model import MultimodalPredictor

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Split_RATIO = 0.8

def train():
    #load data
    full_data = FinancialDataset(
        news_path='data/Combined_News_DJIA.csv',
        prices_path='data/upload_DJIA_table.csv'
    )

    train_size = int(Split_RATIO * len(full_data))
    test_size = len(full_data) - train_size
    train_data, test_data = random_split(full_data, [train_size, test_size])

    print(f"Data loaded {len(full_data)} samples")
    print(f"Train size {len(train_data)} samples, test size {len(test_data)} samples")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    #initialize model
    model = MultimodalPredictor(num_numerical_features=5, hidden_size=64).to(DEVICE)

    #optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            prices = batch['price_features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, mask, prices)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss = loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_acc = evaluate(model, test_loader)

        print(f"Train Loss: {avg_train_loss:.3f}, Train Accuracy: {train_acc:.3f}, Test Accuracy: {val_acc:.3f}")

        save_path = f"models/model_epoch_{epoch+1}.pth"
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), save_path)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            prices =  batch['price_features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, mask, prices)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

if __name__ == '__main__':
    train()