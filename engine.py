import numpy as np
import torch
import torch.optim as optim
import copy
import torch.nn as nn

def asymmetric_mse(under_parameter, over_parameter):
    def loss_fn(y_pred, y_true):
        """
        Custom asymmetric loss:
        - Penalizes under-prediction more/less using under_parameter
        - Penalizes over-prediction more/less using over_parameter
        """

        epsilon = 1e-8
        y_true = torch.clamp(y_true, min=epsilon)

        error = (y_pred - y_true) / y_true
        mse = (y_pred - y_true) ** 2

        # Masks
        under_mask = error < 0
        over_mask  = error > 0

        loss = torch.zeros_like(mse)

        # Under-prediction penalty
        loss = torch.where(
            under_mask,
            under_parameter * mse,
            loss
        )

        # Over-prediction penalty
        loss = torch.where(
            over_mask,
            over_parameter * mse,
            loss
        )

        return torch.mean(loss)

    return loss_fn

def asymmetric_mape(under_parameter, over_parameter):

    def loss_fn(y_pred, y_true):
        eps=1e-8
        # Avoid division by zero
        y_true_safe = torch.clamp(y_true, min=eps)

        # Percentage error
        perc_error = (y_pred - y_true_safe) / y_true_safe

        # Absolute percentage error
        ape = torch.abs(perc_error)

        # Masks
        under_mask = y_pred < y_true_safe
        over_mask  = y_pred > y_true_safe

        loss = torch.zeros_like(ape)

        # Apply asymmetric weights
        loss = torch.where(
            under_mask,
            under_parameter * ape,
            loss
        )

        loss = torch.where(
            over_mask,
            over_parameter * ape,
            loss
        )

        # Mean over batch + horizon
        return torch.mean(loss)

    return loss_fn

def train_model(model, train_loader, val_loader,
                under_parameter, over_parameter,
                epochs=50, lr=0.001, device='cpu',
                horizon_start=0, horizon_end=30,
                patience=10, min_delta=1e-4,loss_fn = 'asymmetric_mse'):
    """
    Train model with early stopping.
    Returns train losses, val losses, and the best model.
    """
    if loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'asymmetric_mse':
        criterion = asymmetric_mse(
            under_parameter=under_parameter,
            over_parameter=over_parameter
        )
    elif loss_fn == 'asymmetric_mape':
        criterion = asymmetric_mape(
            under_parameter=under_parameter,
            over_parameter=over_parameter
        )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    model.to(device)

    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            y_target = y_batch[:, horizon_start:horizon_end]
            loss = criterion(outputs, y_target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                y_target = y_batch[:, horizon_start:horizon_end]
                loss = criterion(outputs, y_target)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # -------- Early stopping logic --------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1}. "
                f"Best Val Loss: {best_val_loss:.6f}"
            )
            break

    # -------- Restore best model --------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, optimizer, criterion




def generate_predictions(model, test_dataset, device='cpu'):
    """Generate predictions for test set and calculate metrics"""
    model.eval()
    
    all_predictions = []
    all_actuals = []
    all_noised = []
    all_dates = []
    with torch.no_grad():
        for i in range(len(test_dataset.X)):
            X = torch.FloatTensor(test_dataset.X[i]).unsqueeze(0).to(device)
            y_pred = model(X).cpu().numpy()[0]
            y_actual = test_dataset.Y[i]
            y_noised = test_dataset.Y_noised[i].flatten()
            date_actual = test_dataset.sample_dates[i]
            # Inverse transform
            if test_dataset.sales_scaler is not None:
                # y_pred_original = test_dataset.scaler.inverse_transform(
                #     np.pad(y_pred.reshape(-1, 1), ((0, 0), (0, 0)), mode='constant')
                # ).flatten()
                y_pred_original = test_dataset.sales_scaler.inverse_transform(
                    y_pred.reshape(-1, 1)
                ).flatten()
                y_actual_original = test_dataset.sales_scaler.inverse_transform(
                    y_actual.reshape(-1, 1)
                ).flatten()
            else:
                y_pred_original = y_pred
                y_actual_original = y_actual.flatten()

            all_predictions.append(y_pred_original)
            all_actuals.append(y_actual_original)
            all_noised.append(y_noised)
            all_dates.append(date_actual)
    
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_dates = np.array(all_dates)
    all_noised = np.array(all_noised)
    
    return all_predictions, all_actuals, all_noised

def generate_warm_up_predictions(
    model,
    test_dataset,
    val_dataset,
    optimizer,
    criterion,
    warmup_days=7,          # actually number of samples
    warmup_epochs=1,
    warmup_every_n_days=1,
    val_pretrain_epochs=10,  # <-- NEW
    device='cpu'
):
    """
    Generate predictions with:
    1) One-time training on validation data
    2) Rolling warm-start retraining during test
    """

    model.to(device)

    all_predictions = []
    all_actuals = []
    all_noised = []
    all_dates = []

    # --------------------------------------------------
    # 0. One-time training on validation dataset
    # --------------------------------------------------
    model.train()
    for _ in range(val_pretrain_epochs):
        for x, y in zip(val_dataset.X, val_dataset.Y):
            x = torch.FloatTensor(x).unsqueeze(0).to(device)
            y = torch.FloatTensor(y).unsqueeze(0).to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    # --------------------------------------------------
    # Combine validation + test for warm-up indexing
    # --------------------------------------------------
    full_X = np.concatenate([val_dataset.X, test_dataset.X], axis=0)
    full_Y = np.concatenate([val_dataset.Y, test_dataset.Y], axis=0)

    val_len = len(val_dataset.X)

    # --------------------------------------------------
    # 1. Rolling prediction on test set
    # --------------------------------------------------
    for i in range(len(test_dataset.X)):

        # -------- Periodic warm-up --------
        do_warmup = (
            warmup_every_n_days is not None
            and i % warmup_every_n_days == 0
        )

        if do_warmup:
            model.train()

            current_idx = val_len + i
            start_idx = max(0, current_idx - warmup_days)

            warmup_X = full_X[start_idx:current_idx]
            warmup_Y = full_Y[start_idx:current_idx]

            for _ in range(warmup_epochs):
                for x, y in zip(warmup_X, warmup_Y):
                    x = torch.FloatTensor(x).unsqueeze(0).to(device)
                    y = torch.FloatTensor(y).unsqueeze(0).to(device)

                    optimizer.zero_grad()
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()

        # -------- Prediction --------
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(test_dataset.X[i]).unsqueeze(0).to(device)
            y_pred = model(X).cpu().numpy()[0]

        y_actual = test_dataset.Y[i]
        y_noised = test_dataset.Y_noised[i].flatten()
        date_actual = test_dataset.sample_dates[i]

        # -------- Inverse transform --------
        if test_dataset.sales_scaler is not None:
            y_pred_original = test_dataset.sales_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()

            y_actual_original = test_dataset.sales_scaler.inverse_transform(
                y_actual.reshape(-1, 1)
            ).flatten()
        else:
            y_pred_original = y_pred
            y_actual_original = y_actual.flatten()

        all_predictions.append(y_pred_original)
        all_actuals.append(y_actual_original)
        all_noised.append(y_noised)
        all_dates.append(date_actual)

    return (
        np.array(all_predictions),
        np.array(all_actuals),
        np.array(all_noised)
    )
