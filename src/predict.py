import torch
import numpy as np
from transformers import DistilBertTokenizer
from model import MultimodalPredictor
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER_NAME = "distilbert-base-uncased"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# finding models is an issue for some reason it is very irritating - simply 'models/model_epoch_10.pth' should work idk why it doesnt
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_FILENAME = 'model_epoch_10.pth'
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_FILENAME)

MEAN_VALUES = np.array([170.0, 172.0, 169.0, 171.0, 1000000.0], dtype=np.float32)
STD_VALUES = np.array([20.0, 20.0, 20.0, 20.0, 500000.0], dtype=np.float32)

def predict_custom(headline, open_price, high, low,close,volume):

    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return
    print(f"Analyzing {headline}")

    model = MultimodalPredictor(num_numerical_features = 5).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("File not found")
        return

    #training mode off
    model.eval()

    #Processing text
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    encoding = tokenizer(
        headline,
        return_tensors='pt',
        padding = 'max_length',
        truncation = True,
        max_length = 128,
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    mask = encoding['attention_mask'].to(DEVICE)

    #normalizing and processing numbers
    raw_nums = np.array([open_price, high, low, close, volume], dtype=np.float32)
    norm_nums = (raw_nums - MEAN_VALUES) / STD_VALUES
    price_tensor = torch.tensor(norm_nums).unsqueeze(0).to(DEVICE)

    #predicting
    with torch.no_grad():
        logits = model(input_ids, mask, price_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        condfidence, prediction = torch.max(probabilities, 1)

    result = "UP" if prediction.item() == 1 else "DOWN"

    print(f"Prediction: {result}")
    print(f"confidence: {condfidence.item() * 100:.2f}%")

if __name__ == "__main__":
    # Scenario 1: Good News
    predict_custom(
        headline="Tech giant reports record breaking profits, market rallies",
        open_price=150.0, high=155.0, low=149.0, close=154.0, volume=5000000
    )

    # Scenario 2: Bad News
    predict_custom(
        headline="Inflation hits 40 year high, fed plans to raise rates aggressively",
        open_price=150.0, high=151.0, low=140.0, close=142.0, volume=8000000
    )
