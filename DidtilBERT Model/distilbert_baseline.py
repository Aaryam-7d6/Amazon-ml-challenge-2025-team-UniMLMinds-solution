"""
DistilBERT Price Prediction Baseline - Amazon ML Challenge
Formatted for competition submission structure
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 128
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_LOG_PRICE = True
    MODEL_PATH = 'trained_model.pth'  # Path to save/load trained model


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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
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
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        price = self.regressor(cls_output)
        return price.squeeze()


# TRAINING FUNCTION

def train_model(train_df):
    """
    Train the DistilBERT model on training data
    """
    print("🚀 Starting Model Training...")
    print(f"Device: {Config.DEVICE}")
    print(f"Training samples: {len(train_df)}")
    
    # Prepare data
    texts = train_df['catalog_content'].values
    prices = train_df['price'].values
    
    if Config.USE_LOG_PRICE:
        prices = np.log1p(prices)
        print("✅ Applied log1p transformation to prices")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = DistilBERTPricePredictor(Config.MODEL_NAME).to(Config.DEVICE)
    
    # Create dataset and dataloader
    train_dataset = PriceDataset(texts, prices, tokenizer, Config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.HuberLoss(delta=1.0)
    
    # Training loop
    epochs = 3  # Reduced for faster training
    model.train()
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            target_prices = batch['price'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, target_prices)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': Config.MODEL_NAME,
            'use_log_price': Config.USE_LOG_PRICE
        }
    }, Config.MODEL_PATH)
    print(f"\n✅ Model saved to {Config.MODEL_PATH}")
    
    return model, tokenizer


# PREDICTION FUNCTION (COMPETITION FORMAT)

def predictor(sample_id, catalog_content, image_link, model=None, tokenizer=None):
    '''
    Predict product price using DistilBERT model
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image (not used in baseline)
    
    Returns:
    - price: Predicted price as a float
    '''
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided")
    
    # Tokenize input
    encoding = tokenizer(
        str(catalog_content),
        max_length=Config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(Config.DEVICE)
    attention_mask = encoding['attention_mask'].to(Config.DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(input_ids, attention_mask)
    
    # Inverse transform
    price = prediction.cpu().item()
    if Config.USE_LOG_PRICE:
        price = np.expm1(price)  # exp(x) - 1
    
    # Ensure positive price
    price = max(price, 0.01)
    
    return round(price, 2)


# BATCH PREDICTION (OPTIMIZED FOR SPEED)

def batch_predictor(test_df, model, tokenizer, batch_size=32):
    """
    Predict prices for entire test set in batches (faster than row-by-row)
    """
    print("\n🔮 Generating Predictions...")
    
    texts = test_df['catalog_content'].values
    test_dataset = PriceDataset(texts, None, tokenizer, Config.MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            
            preds = model(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Inverse transform
    if Config.USE_LOG_PRICE:
        predictions = np.expm1(predictions)
    
    # Ensure positive prices
    predictions = np.maximum(predictions, 0.01)
    predictions = np.round(predictions, 2)
    
    return predictions


# MAIN EXECUTION

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    print("="*60)
    print("DistilBERT Price Prediction - Amazon ML Challenge")
    print("="*60)
    
    # Check if model already trained
    if os.path.exists(Config.MODEL_PATH):
        print(f"\n✅ Found existing model at {Config.MODEL_PATH}")
        print("Loading trained model...")
        
        # Load model
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
        model = DistilBERTPricePredictor(Config.MODEL_NAME).to(Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ Model loaded successfully!")
        
    else:
        print(f"\n⚠️ No trained model found. Training new model...")
        
        # Load training data
        train_path = os.path.join(DATASET_FOLDER, 'train.csv')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        train_df = pd.read_csv(train_path)
        print(f"Loaded {len(train_df)} training samples")
        
        # Train model
        model, tokenizer = train_model(train_df)
    
    # Load test data
    print(f"\n📂 Loading test data...")
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')
    test = pd.read_csv(test_path)
    print(f"Loaded {len(test)} test samples")
    
    # METHOD 1: Batch Prediction (FASTER - RECOMMENDED)
    print("\n🚀 Using batch prediction for speed...")
    test['price'] = batch_predictor(test, model, tokenizer, batch_size=Config.BATCH_SIZE)
    
    # METHOD 2: Row-by-row prediction (slower, but matches competition format)
    # Uncomment below if you want to use row-by-row approach
    """
    print("\n🚀 Using row-by-row prediction...")
    test['price'] = test.apply(
        lambda row: predictor(
            row['sample_id'], 
            row['catalog_content'], 
            row['image_link'],
            model=model,
            tokenizer=tokenizer
        ), 
        axis=1
    )
    """
    
    # Select only required columns
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE")
    print("="*60)
    print(f"📁 Predictions saved to: {output_filename}")
    print(f"📊 Total predictions: {len(output_df)}")
    print(f"💰 Price range: ₹{output_df['price'].min():.2f} - ₹{output_df['price'].max():.2f}")
    print(f"📈 Average price: ₹{output_df['price'].mean():.2f}")
    print(f"\n🔍 Sample predictions:")
    print(output_df.head(10))
    print("="*60)
