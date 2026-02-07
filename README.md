# Multimodal Financial Predictor (BERT + Numerical Fusion)

A Deep Learning system that predicts stock market trends (Up/Down) by fusing **Financial News Sentiment** with **Historical Price Data**.
The model uses a **Multimodal Architecture** that combines a pre-trained **DistilBERT** for text processing and a **Neural Network** for numerical analysis, achieving a baseline accuracy of ~54% on the DJIA dataset (beating random choice in a highly stochastic environment)

## Features 
* **Multimodal Fusion:** Concatenates text embeddings with normalized price vectors.
* *8Dynamic Normalization:** Automatically adjusts to different stock price ranges using rolling window statistics.
* **Live Automation:** A script that fetches real-time data for top stocks and generates instant predictions
* **Robust Pipeline:** Includes custom PyTorch Datasets, rigorous data cleaning, and modular training loops

## Tech Stack 
* **Core:** Python 3.9+, PyTorch 
* **NLP:** Hugging Face Transformers (DistilBERT)
* **Data:** Pandes, NumPy, yfinance (Yahoo Finance API)
* **Training:** AdamW Optimizer, CrossEntropyLoss

## Project Structure
```angular2html
financial_predictor/
├── data/                   # Dataset storage (DJIA CSVs)
├── models/                 # Saved model checkpoints (.pth)
├── src/
│   ├── dataset.py          # Custom PyTorch Dataset (Text+Price alignment)
│   ├── model.py            # Multimodal Neural Network Architecture
│   ├── train.py            # Training loop with validation
│   ├── predict.py          # Inference script for manual inputs
│   └── automate.py         # Live stock screening & prediction
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
# Quick Start 
## 1. Installation 
clone the repo and install dependencies:
```angular2html
git clone [https://github.com/YOUR_USERNAME/multimodal_financial_predictor.git](https://github.com/YOUR_USERNAME/multimodal_financial_predictor.git)
cd multimodal_financial_predictor
pip install -r requirements.txt
```
## 2. Training the Model
To train the model from scratch on the Combined News DJIA dataset:
```
python src/train.py
```
This will train 10 epochs and save weights to the models/ folder

## 3. Live Predictions (Automation)
To predict the movement of top tech stocks right now: 
```angular2html
python src/automate.py
```
# Model Architecture
The system uses a **Late Fusion** approach: 
* **Text Branch:** News headlines are tokenized and passed through DistilBERT.
* **Numerical Branch:** OHLCV(Open, High, Low, Close, Volume) data is normalized using Z-score and then processed through a **Linear Layer**.
* **Fusion:** The two feature vectors are concatenated and passed through a final classifier layer to output logits for UP or DOWN

# Results 
* **Training Loss:** Converged to ~0.69
* **Test Accuracy:** ~54% (Baseline)
* **Observation:** THe model successfully identifies extreme market sentiment but, like all financial models, faces challenges with neutral/noise news.

# Future Improvements 
* Implement LSTM/GRU layers for better time-series handling of price history 
* Add Technical Indicators (RSI, MACD) as additional input features
* Build a Backtesting Engine to simulate trading profits over time