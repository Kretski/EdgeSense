# src/edgesense_core.py
"""
EdgeSenseAnomalyFull – пълна версия за symbolic distillation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import os
import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyAnomalyNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)  # без sigmoid – за symbolic по-лесно

class EdgeSenseAnomalyFull:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.nn_model = TinyAnomalyNN(input_dim).to(DEVICE)
        self.optimizer = optim.AdamW(self.nn_model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        self.lin_model = SGDRegressor(max_iter=1000, learning_rate='adaptive', eta0=0.01)
        self.tree_model = DecisionTreeRegressor(max_depth=8)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Обучение на NN + класически модели"""
        logger.info(f"Training NN on {X.shape[0]} samples, {epochs} epochs...")
        
        X_torch = torch.from_numpy(X).float().to(DEVICE)
        y_torch = torch.from_numpy(y).float().unsqueeze(1).to(DEVICE)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.nn_model(X_torch)
            loss = self.criterion(pred, y_torch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Fit класически модели
        self.lin_model.fit(X, y)
        self.tree_model.fit(X, y)
        
        logger.info("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble предсказание"""
        X_t = torch.from_numpy(X).float().to(DEVICE)
        with torch.no_grad():
            p_nn = self.nn_model(X_t).cpu().numpy().ravel()
        
        p_lin = self.lin_model.predict(X)
        p_tree = self.tree_model.predict(X)
        
        # Прост ensemble (средно)
        return (p_nn + p_lin + p_tree) / 3

    def export_onnx(self, input_dim: int, save_path: str = "anomaly_full.onnx"):
        """Експорт на NN в ONNX"""
        dummy_input = torch.randn(1, input_dim, device=DEVICE)
        
        torch.onnx.export(
            self.nn_model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=18,  # по-нова версия, за да избегнем ReLU проблеми
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        
        logger.info(f"ONNX exported → {save_path}")