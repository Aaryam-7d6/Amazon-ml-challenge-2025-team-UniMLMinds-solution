"""
Optimized DistilBERT Price Prediction Baseline (25k subset)
GPU-Accelerated for RTX 3050
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 128
    BATCH_SIZE = 64           # larger batch to maximize GPU throughput
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_LOG_PRICE = True
    EPOCHS = 2                # fewer epochs for quick iteration
    MODEL_PATH = 'trained_model_25k.pth'
    TRAIN_SIZE = 25000
    TEST_SIZE = 25000
    MIXED_PRECISION = True    # Enable AMP (Automatic Mixed Precision)


# DATASET

class PriceDataset(Dataset):
    def __init__(self, texts, prices=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.prices = prices
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.prices is not None:
            item['price'] = torch.tensor(self.prices[idx], dtype=torch.float)
        return item


# MODEL ARCHITECTURE

class DistilBERTPricePredictor(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        price = self.regressor(cls_output)
        return price.squeeze()


# TRAINING FUNCTION

def train_model(train_df):
    print("🚀 Training Optimized DistilBERT Model...")
    print(f"Device: {Config.DEVICE}")
    
    # Subset 25k samples
    train_df = train_df.sample(n=Config.TRAIN_SIZE, random_state=42).reset_index(drop=True)
    texts = train_df['catalog_content'].values
    prices = train_df['price'].values

    if Config.USE_LOG_PRICE:
        prices = np.log1p(prices)
        print("✅ Applied log1p transformation to prices")

    tokenizer = DistilBertTokenizerFast.from_pretrained(Config.MODEL_NAME)
    model = DistilBERTPricePredictor(Config.MODEL_NAME).to(Config.DEVICE)

    train_dataset = PriceDataset(texts, prices, tokenizer, Config.MAX_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_fn = nn.HuberLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.MIXED_PRECISION)

    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(Config.DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(Config.DEVICE, non_blocking=True)
            target_prices = batch['price'].to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=Config.MIXED_PRECISION):
                predictions = model(input_ids, attention_mask)
                loss = loss_fn(predictions, target_prices)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"✅ Model saved to {Config.MODEL_PATH}")
    return model, tokenizer


# PREDICTION

def batch_predictor(test_df, model, tokenizer):
    test_df = test_df.sample(n=Config.TEST_SIZE, random_state=42).reset_index(drop=True)
    texts = test_df['catalog_content'].values

    test_dataset = PriceDataset(texts, None, tokenizer, Config.MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(Config.DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(Config.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=Config.MIXED_PRECISION):
                preds = model(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy())

    predictions = np.array(predictions)
    if Config.USE_LOG_PRICE:
        predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0.01)
    predictions = np.round(predictions, 2)
    return predictions


# MAIN

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'

    print("="*60)
    print("DistilBERT Price Prediction (25k Subset - Optimized GPU)")
    print("="*60)

    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"📊 Train: {len(train_df)} | Test: {len(test_df)}")

    model, tokenizer = train_model(train_df)
    preds = batch_predictor(test_df, model, tokenizer)
    test_df = test_df.iloc[:len(preds)].copy()
    test_df['price'] = preds

    output_path = os.path.join(DATASET_FOLDER, 'test_out_25k.csv')
    test_df[['sample_id', 'price']].to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")

