import yfinance as yf
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer
from model import MultimodalPredictor
import os
import datetime

#CONFIG

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER_NAME = "distilbert-base-uncased"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_epoch_10.pth")

WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 'AMD', 'NFLX', 'SPY']

#fetches the latest price and news for a stock, also the last 50 days to calculate dynamic normalization stats
def get_live_data(ticker_symbol):

    stock = yf.Ticker(ticker_symbol)

    hist = stock.history(period='3mo')

    if len(hist) < 50:
        print(f"not enough data for {ticker_symbol}")
        return None

    current = hist.iloc[-1]

    #dynamic normalization stats
    rolling_window = hist.iloc[-51:-1]

    mean_stats = rolling_window[['Open', 'High', 'Low', 'Close', 'Volume']].mean()
    std_stats = rolling_window[['Open', 'High', 'Low', 'Close', 'Volume']].std()

    #news getter
    news_items = stock.news
    if news_items:
        item = news_items[0]
        headline = item.get('title', item.get('headline', ''))

        if not headline and 'content' in item:
            headline = item['content'].get('title', '')

        if not headline:
            headline = f"News found for {ticker_symbol}, but cant find title"

        news_link = item.get('link', '')
    else:
        headline = "nothing today"
        news_link = ""

    return {
        'ticker': ticker_symbol,
        'headline': headline,
        'open': current['Open'],
        'high': current['High'],
        'low': current['Low'],
        'close': current['Close'],
        'volume': current['Volume'],
        'means': mean_stats.values,
        'stds': std_stats.values,
        'date': current.name,
        'link': news_link,
    }

def run_prediction(data):
    if data is None: return

    model = MultimodalPredictor(num_numerical_features=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("no model")
        return

    model.eval()

    #text processing
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    encoding = tokenizer(
        data['headline'],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128,
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    mask = encoding['attention_mask'].to(DEVICE)


    #Numerical processing
    #creating a Z-score to help with normalization

    raw_nums = np.array([data['open'], data['high'], data['low'], data['close'], data['volume']],dtype=np.float32)
    means = data['means'].astype(np.float32)
    stds = data['stds'].astype(np.float32)

    stds[stds==0] = 1.0

    norms = (raw_nums - means)/ stds
    price_tensor = torch.tensor(norms).unsqueeze(0).to(DEVICE)

    #predict
    with torch.no_grad():
        logits = model(input_ids,mask, price_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probs, 1)

    result = "UP" if prediction.item()==1 else "DOWN"

    print(f"Stock: {ticker}")
    print(f"Price: {data['close']:.2f}")
    print(f"Headline: {data['headline']}")
    print(f"Confidence: {confidence.item() * 100:.1f}%")
    print(f"Prediction :  {result}")

if __name__ == "__main__":
    print(f"Tracking: {', '.join(WATCHLIST)}")

    for ticker in WATCHLIST:
        data = get_live_data(ticker)
        run_prediction(data)