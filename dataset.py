from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

class GasDemandDataset:
    def __init__(self, df,features,sales_scaler=None, crude_scaler=None,
                 seq_length=60, forecast_length=30):

        self.dataset_df = df.copy()
        self.features = features
        self.seq_length = seq_length
        self.forecast_length = forecast_length
        self.sales_scaler = sales_scaler
        self.crude_scaler = crude_scaler

        if self.sales_scaler:
            self.dataset_df['Sales'] = self.sales_scaler.transform(self.dataset_df['Sales'].values.reshape(-1,1)).flatten()
        if 'Crude' in self.features and self.crude_scaler:
            self.dataset_df['Crude'] = self.crude_scaler.transform(self.dataset_df['Crude'].values.reshape(-1,1)).flatten()

        self.dates = self.dataset_df['Date'].values
        self.sales = self.dataset_df['Sales'].values
        self.noised_sales = self.dataset_df['Noised_Sales'].values
        self.features_values = self.dataset_df[self.features].values
    
        # Create sequences
        self.X, self.Y, self.Y_noised, self.sample_dates = self._create_sequences()
    
    def _create_sequences(self):
        X, Y, Y_noised, dates = [], [], [], []

        for i in range(self.seq_length, self.features_values.shape[0] - self.forecast_length):

            dates.append(self.dates[i])
            X.append(self.features_values[i - self.seq_length : i])
            Y.append(self.sales[i  : i + self.forecast_length])
            Y_noised.append(self.noised_sales[i  : i + self.forecast_length])
                    
        return np.array(X), np.array(Y), np.array(Y_noised), np.array(dates)
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        X_tensor = torch.FloatTensor(self.X)
        y_tensor = torch.FloatTensor(self.Y).float()
        
        dataset = TensorDataset(
            X_tensor, 
            y_tensor, 
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

