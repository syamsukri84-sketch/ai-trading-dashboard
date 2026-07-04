import numpy as np
import pandas as pd
import logging
import os
import joblib
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

if torch is not None:
    class PyTorchLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 25)
            self.fc2 = nn.Linear(25, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :] # Ambil output dari time step terakhir
            out = self.dropout(torch.relu(self.fc1(out)))
            out = self.fc2(out)
            return out

class LSTMPriceProjector:
    """
    Model berbasis Deep Learning (LSTM) untuk memproyeksikan pergerakan harga saham.
    Memanfaatkan historical 'lookback' (misal 20 hari ke belakang) untuk memprediksi N hari ke depan.
    """
    
    def __init__(self, projection_horizon: int = 3, lookback: int = 20):
        self.projection_horizon = projection_horizon
        self.lookback = lookback
        self.scaler = None
        self.model = None

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        data = df.copy()
        
        # Ambil semua kolom numerik/fitur
        feature_cols = [c for c in data.columns if c.startswith('feat_') or c in ['open', 'high', 'low', 'close', 'volume']]
        X_raw = data[feature_cols].values
        
        # Target = Return persentase N hari ke depan
        target = (data['close'].shift(-self.projection_horizon) / data['close']) - 1.0
        y_raw = target.values
        
        # Normalisasi data (Sangat penting untuk Neural Networks)
        if is_training:
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            if self.scaler is None:
                raise ValueError("Model belum dilatih dan scaler belum di-fit.")
            X_scaled = self.scaler.transform(X_raw)
            
        X, y = [], []
        for i in range(self.lookback, len(X_scaled)):
            X.append(X_scaled[i-self.lookback:i])
            y.append(y_raw[i-1]) # Sesuaikan target dengan akhir sequence
            
        X = np.array(X)
        y = np.array(y)
        
        if is_training:
            # Hapus baris yang nilai targetnya NaN (karena shift ke masa depan)
            valid_idx = ~np.isnan(y)
            return X[valid_idx], y[valid_idx]
        
        return X

    def train(self, features_df: pd.DataFrame, epochs: int = 10, batch_size: int = 32, save_path: Optional[str] = None):
        """Melatih model LSTM pada data sekuensial."""
        if torch is None:
            logger.error("Library PyTorch belum terinstal. Jalankan: pip install torch scikit-learn")
            return
            
        logger.info(f"Melatih model proyeksi LSTM ({epochs} epochs) dengan lookback {self.lookback} hari...")
        X_train, y_train = self._prepare_data(features_df, is_training=True)
        
        # Konversi ke PyTorch Tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = PyTorchLSTM(input_size=X_train.shape[2])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        logger.info("Pelatihan model LSTM selesai.")

    def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Memprediksi return menggunakan data N hari terakhir (lookback window)."""
        if self.model is None or torch is None:
            return {"projected_return_pct": 0.0, "projected_price": 0.0}
            
        X_test = self._prepare_data(features_df, is_training=False)
        latest_sequence = X_test[-1].reshape(1, self.lookback, -1)
        seq_tensor = torch.tensor(latest_sequence, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            projected_return = float(self.model(seq_tensor).item())
            
        current_price = float(features_df['close'].iloc[-1])
        
        return {
            "projected_return_pct": projected_return * 100,
            "projected_price": current_price * (1 + projected_return)
        }