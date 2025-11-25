
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# CONFIG
class CFG:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 128
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 2
    LR = 3e-5
    USE_LOG_PRICE = True
    TRAIN_SIZE = 25000
    TEST_SIZE = 25000 # This is for the final prediction on the test set
    VAL_SIZE = 5000   # Define validation set size
    MODEL_PATH = 'trained_model_25k.pth'
    MIXED_PRECISION = True

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
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.prices is not None:
            item['price'] = torch.tensor(self.prices[idx], dtype=torch.float)
        return item

# MODEL
class DistilBERTRegressor(nn.Module):
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
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.regressor(cls).squeeze()

# SMAPE METRIC
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)
    return np.mean(diff) * 100

# TRAIN FUNCTION
def train_model(train_df):
    print(f"Training on {CFG.TRAIN_SIZE} samples | Device: {CFG.DEVICE}")

    train_df = train_df.sample(n=CFG.TRAIN_SIZE, random_state=42).reset_index(drop=True)
    texts, prices = train_df['catalog_content'].values, train_df['price'].values

    if CFG.USE_LOG_PRICE:
        prices = np.log1p(prices)

    tokenizer = DistilBertTokenizerFast.from_pretrained(CFG.MODEL_NAME)
    model = DistilBERTRegressor(CFG.MODEL_NAME).to(CFG.DEVICE)
    dataset = PriceDataset(texts, prices, tokenizer, CFG.MAX_LEN)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR)
    loss_fn = nn.HuberLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.MIXED_PRECISION)

    model.train()
    for epoch in range(CFG.EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}"):
            ids, mask, y = (
                batch['input_ids'].to(CFG.DEVICE),
                batch['attention_mask'].to(CFG.DEVICE),
                batch['price'].to(CFG.DEVICE)
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=CFG.MIXED_PRECISION):
                preds = model(ids, mask)
                loss = loss_fn(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), CFG.MODEL_PATH)
    print(f"Model saved → {CFG.MODEL_PATH}")
    return model, tokenizer

# PREDICT FUNCTION
def predict(df, model, tokenizer): # Removed num_samples parameter
    dataset = PriceDataset(df['catalog_content'].values, None, tokenizer, CFG.MAX_LEN)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            ids, mask = batch['input_ids'].to(CFG.DEVICE), batch['attention_mask'].to(CFG.DEVICE)
            with torch.cuda.amp.autocast(enabled=CFG.MIXED_PRECISION):
                out = model(ids, mask)
            preds.extend(out.cpu().numpy())

    preds = np.array(preds)
    if CFG.USE_LOG_PRICE:
        preds = np.expm1(preds)
    return np.clip(preds, 0.01, None)

# EVALUATION
def evaluate(y_true, y_pred):
    print("\n📊 Model Evaluation Metrics:")
    print(f"SMAPE : {smape(y_true, y_pred):.2f}%")
    print(f"MAE   : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE  : {mean_squared_error(y_true, y_pred)}")
    print(f"R²    : {r2_score(y_true, y_pred):.4f}")

# MAIN
if __name__ == "__main__":
    DATASET_PATH = '/content/'
    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))

    model, tokenizer = train_model(train_df)

    # Validation Split
    val_df = train_df.sample(CFG.VAL_SIZE, random_state=123) # Use CFG.VAL_SIZE
    val_true = val_df['price'].values
    if CFG.USE_LOG_PRICE:
        val_true = np.expm1(np.log1p(val_true))
    val_pred = predict(val_df, model, tokenizer) # Removed num_samples
    evaluate(val_true, val_pred)

    # Final Test Predictions
    test_df_sampled = test_df.sample(n=CFG.TEST_SIZE, random_state=42).reset_index(drop=True) # Sample test_df
    preds = predict(test_df_sampled, model, tokenizer) # Predict on sampled test_df
    test_df_sampled['price'] = np.round(preds, 2) # Assign predictions to sampled test_df
    test_df_sampled[['sample_id', 'price']].to_csv(os.path.join(DATASET_PATH, 'test_out_25k.csv'), index=False) # Save sampled test_df
    print("\n✅ test_out_25k.csv saved successfully.")


    import matplotlib.pyplot as plt
    plt.scatter(val_true, val_pred, alpha=0.4)
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("True vs Predicted Prices")
    plt.show()

    #from google.colab import files
    #files.download('/content/best_model.pth')
