import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation='relu', dropout=0.0):
        """
        input_size   : int
        output_size  : int
        hidden_layers: list  (e.g. [128, 64, 32])
        activation   : str   ('relu', 'tanh', 'gelu', 'leaky_relu', 'sigmoid')
        """
        super(FFNN, self).__init__()

        # Activation mapping
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid
        }

        if activation.lower() not in activations:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from {list(activations.keys())}"
            )

        act_fn = activations[activation.lower()]

        layers = []
        prev_size = input_size

        for neurons in hidden_layers:
            layers.append(nn.Linear(prev_size, neurons))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout)) 
            prev_size = neurons

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # (batch, seq_len, features) → (batch, seq_len*features)
        x = x.view(x.size(0), -1)
        return self.model(x)



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='linear', dropout=0.0):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Output activation (linear by default for regression)
        activations = {
            'linear': nn.Identity(),   
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
        }

        act = activation.lower()
        if act not in activations:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activations.keys())}")
        self.activation = activations[act]

    def forward(self, x):
        out, _ = self.gru(x)        
        out = out[:, -1, :]       
        out = self.fc(out)         
        out = self.activation(out) 
        return out


class PositionalEncoding(nn.Module):
    """Add positional encoding to input embeddings"""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    """Vanilla Transformer for time series forecasting"""
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, 
                 output_length, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_length = output_length
        
        # Input projection: map input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=500)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: from d_model to output_length
        self.output_projection = nn.Linear(d_model, output_length)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_length, d_model)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Use the last output for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Project to output length (30 days)
        output = self.output_projection(x)  # (batch_size, output_length)
        
        return output

