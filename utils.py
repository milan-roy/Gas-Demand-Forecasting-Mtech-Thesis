import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataset import GasDemandDataset
from tabulate import tabulate

def prepare_merged_df(denoised_sales_df,
                  noised_sales_df, 
                  crude_df):
    
    
    denoised_sales_df = denoised_sales_df.copy()
    noised_sales_df = noised_sales_df.copy()
    crude_df = crude_df.copy()

    # ---- RENAME COLUMNS ----
    denoised_sales_df.columns = ['Date', 'Sales']
    noised_sales_df.columns = ['Date', 'Noised_Sales']
    crude_df.columns = ['Date', 'Crude']

    # ---- PARSE & SORT DATES ----
    for df in [denoised_sales_df, noised_sales_df, crude_df]:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ---- MERGE SALES DFS ----
    merged_df = pd.merge(
        denoised_sales_df,
        noised_sales_df,
        on='Date',
        how='inner'
    )

    # ---- FFILL() MISSING DATES IN CRUDE_DF ----
    full_dates = pd.date_range(
        start=crude_df['Date'].min(),
        end=crude_df['Date'].max(),
        freq='D'
    )

    crude_df = (crude_df.set_index('Date').reindex(full_dates))
    crude_df['Crude'] = crude_df['Crude'].ffill()
    crude_df = crude_df.reset_index().rename(columns={'index': 'Date'})

    
    merged_df = pd.merge(
        merged_df,
        crude_df,
        on='Date',
        how='left'
    )

    return merged_df

def generate_features(merged_df,
                      use_crude=False,
                      use_month=False,
                      use_dayofweek=False,
                      use_month_sin_cos=False,
                      use_dayofweek_sin_cos=False,
                      ):
    
    features = ['Sales']
    merged_df = merged_df.copy()
    if use_crude:
        features.append(f'Crude')  
    if use_month:   
        features.append('Month')
    if use_month_sin_cos:
        features.extend(['Month_sin', 'Month_cos'])
    if use_dayofweek:
        features.append('dow')
    if use_dayofweek_sin_cos:
        features.extend(['dow_sin', 'dow_cos'])
    
    # ---- Month and Day Features ----
    if use_month:
        merged_df['Month'] = merged_df['Date'].dt.month
    if use_dayofweek:
        merged_df['dow'] = merged_df['Date'].dt.dayofweek  # 0=Mon, 6=Sun
    
    # ---- Cyclic Encoding ----
    if use_month_sin_cos:
        months = merged_df['Month'] = merged_df['Date'].dt.month
        merged_df['Month_sin'] = np.sin(2 * np.pi * months/ 12)
        merged_df['Month_cos'] = np.cos(2 * np.pi * months / 12)
    if use_dayofweek_sin_cos:
        dow = merged_df['Date'].dt.dayofweek
        merged_df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        merged_df['dow_cos'] = np.cos(2 * np.pi * dow / 7)


    merged_df = merged_df.dropna().reset_index(drop=True)
    return merged_df, features


def load_and_prepare_data(denoised_sales_df,
                            noised_sales_df, 
                            crude_df,
                            test_split_date,
                            val_split_date=None,
                            seq_length=60,
                            forecast_length=30,
                            batch_size=32,
                            sales_scaler=None,
                            crude_scaler=None,
                            use_crude = False,
                            use_month = False,
                            use_month_sin_cos = False,
                            use_dayofweek = False,
                            use_dayofweek_sin_cos = False,
                            shift_crude_days=0):

    denoised_sales_df = denoised_sales_df.copy()
    noised_sales_df = noised_sales_df.copy()
    crude_df = crude_df.copy()

    merged_df = prepare_merged_df(denoised_sales_df,
                                  noised_sales_df, 
                                  crude_df)
    
    merged_df, features = generate_features(merged_df,
                                            use_crude,
                                            use_month,
                                            use_dayofweek,
                                            use_month_sin_cos,  
                                            use_dayofweek_sin_cos,
                                            )
    # ---- SHIFT CRUDE OIL PRICES IF NEEDED ----
    if use_crude and shift_crude_days != 0:
        merged_df['Crude'] = merged_df['Crude'].shift(shift_crude_days)
        merged_df = merged_df.dropna().reset_index(drop=True)

    test_split = pd.to_datetime(test_split_date)
    
    if val_split_date:
        val_split = pd.to_datetime(val_split_date)
        train_df = merged_df[merged_df['Date'] < val_split].copy()
        val_df   = merged_df[(merged_df['Date'] >= val_split) & (merged_df['Date'] < test_split)].copy()
        test_df  = merged_df[merged_df['Date'] >= test_split].copy()

        train_to_val_overlap = train_df.tail(seq_length)
        val_df = pd.concat([train_to_val_overlap, val_df]).reset_index(drop=True)

        val_to_test_overlap = val_df.tail(seq_length)
        test_df = pd.concat([val_to_test_overlap, test_df]).reset_index(drop=True)
        
    else:
        train_df = merged_df[merged_df['Date'] < test_split].copy()
        test_df  = merged_df[merged_df['Date'] >= test_split].copy()

        train_to_test_overlap = train_df.tail(seq_length)
        test_df = pd.concat([train_to_test_overlap, test_df]).reset_index(drop=True)

        val_df = test_df.copy()

    print(f"Total records : {len(merged_df)}")
    print(f"Train records : {len(train_df)}")
    if val_split_date:
        print(f"Val records   : {len(val_df)-seq_length}")
    print(f"Test records  : {len(test_df)-seq_length}")
    print()
    if val_split_date:
        print(f"Val starts  at: {val_df.iloc[seq_length]['Date'].date()}")
    print(f"Test starts at: {test_df.iloc[seq_length]['Date'].date()}")


    print("Using features:", features)

    # ---- SCALER ----
    if sales_scaler:
        sales_scaler.fit(train_df['Sales'].values.reshape(-1, 1))
    if crude_scaler and use_crude:
        crude_scaler.fit(train_df['Crude'].values.reshape(-1, 1))

    # ---- DATASETS ----
    train_dataset = GasDemandDataset(
        df=train_df,
        features=features,
        sales_scaler=sales_scaler,
        crude_scaler=crude_scaler,
        seq_length=seq_length,
        forecast_length=forecast_length 
    )
    
    val_dataset = GasDemandDataset(
        df=val_df,
        features=features,
        sales_scaler=sales_scaler,
        crude_scaler=crude_scaler,
        seq_length=seq_length,
        forecast_length=forecast_length
    )

    test_dataset = GasDemandDataset(
        df=test_df,
        features=features,
        sales_scaler=sales_scaler,
        crude_scaler=crude_scaler,
        seq_length=seq_length,
        forecast_length=forecast_length 
    )

    return train_dataset, val_dataset, test_dataset


def get_MAPES(Y_pred, Y_true):

    Y_true = np.array(Y_true.copy(), dtype=float)
    Y_pred = np.array(Y_pred.copy(), dtype=float)

    # Avoid division by zero
    epsilon=1e-8
    Y_true = np.where(Y_true == 0, epsilon, Y_true)

    # Absolute percentage error
    ape = np.abs((Y_true - Y_pred)*100 / Y_true)

    # Mean over horizon (axis=1)
    mapes = np.mean(ape, axis=1)

    return mapes
    
def get_error_df(
    Y_pred_uncorrected,
    Y_pred_corrected,
    Y_true_noised,
    sample_dates
):

    Y_true_noised = Y_true_noised.copy()
    Y_pred_corrected = Y_pred_corrected.copy()
    Y_pred_uncorrected = Y_pred_uncorrected.copy()
    sample_dates = sample_dates.copy()
    
    # Convert inputs safely (no unnecessary copies)
    Y_true_noised = np.asarray(Y_true_noised, dtype=float)
    Y_pred_corr = np.asarray(Y_pred_corrected, dtype=float)
    Y_pred_uncorr = np.asarray(Y_pred_uncorrected, dtype=float)
    dates = pd.to_datetime(sample_dates)

    # --- MAPE ---
    uncorrected_mapes = get_MAPES(Y_true=Y_true_noised, Y_pred=Y_pred_uncorr)
    corrected_mapes   = get_MAPES(Y_true=Y_true_noised, Y_pred=Y_pred_corr)

    # --- Percentage errors ---
    prediction_errors = (Y_pred_corr - Y_true_noised) * 100.0 / Y_true_noised

    # --- Penalities ---
    under_mask = prediction_errors <= -5
    over_mask  = prediction_errors >= 10

    penality_predictions = np.sum(under_mask | over_mask, axis=1)
    penality_under = np.sum(under_mask, axis=1)
    penality_over  = np.sum(over_mask, axis=1)

     # Use masked arrays to avoid Python loops
    under_errors = np.where(prediction_errors < 0, prediction_errors, np.nan)
    over_errors  = np.where(prediction_errors > 0, prediction_errors, np.nan)

    under_mean = np.nanmean(under_errors, axis=1)
    over_mean  = np.nanmean(over_errors, axis=1)

    # --- Build DataFrame ---
    error_df = pd.DataFrame(
        {
            "CorrectedMape": corrected_mapes,
            "UncorrectedMape": uncorrected_mapes,
            "NumberPenalities": penality_predictions,
            "NumberUnderPrediction": penality_under,
            "NumberOverPrediction": penality_over,
            "ErrorOverPrediction": over_mean,
            "ErrorUnderPrediction": under_mean,
        },
        index=dates,
    )

    # --- Date filtering ---
    error_df = error_df.loc["2024-01-01":"2024-11-30"]
    return error_df

def plot_error_df(error_df):
    error_df = error_df.copy()

    # ---- PLOTTING ----   
    # Plot Unorrected and Corrected MAPE
    fig = go.Figure(data=[go.Histogram(x=error_df['UncorrectedMape'], nbinsx=50)])
    fig.update_layout(xaxis_title='Daily Uncorrected MAPE MAPE', yaxis_title='Count', title=f'Histogram of Uncorrected Average MAPE. Average  = ' + str(error_df['UncorrectedMape'].mean()))
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=error_df['CorrectedMape'], nbinsx=50)])
    fig.update_layout(xaxis_title='Daily Corrected MAPE MAPE', yaxis_title='Count', title=f'Histogram of Corrected Average MAPE. Average  = ' + str(error_df['CorrectedMape'].mean()))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Line(x = error_df.index,
                        y = error_df['CorrectedMape'],
                        mode = 'lines+markers',
                        name = 'Corrected Average MAPE'))
    fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Corrected MAPE', title = 'Corrected MAPE vs Date.  Average = '+str(error_df['CorrectedMape'].mean()))
    fig.show()  

    # Plot Number of Penalities
    fig = go.Figure()
    fig.add_trace(go.Line(x = error_df.index,
                        y = error_df['NumberPenalities'],
                        mode = 'lines+markers',
                        name = 'Number of Penalities'))
    fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Number of Penalities', title = 'Number of Penalities vs Date.  Average = '+str(error_df['NumberPenalities'].mean()))
    fig.show()      

    # Plot Number of Under Predictions
    fig = go.Figure()
    fig.add_trace(go.Line(x = error_df.index,
                        y = error_df['NumberUnderPrediction'],
                        mode = 'lines+markers',
                        name = 'Number of Under Predictions'))
    fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Number of Under Predictions', title = 'Number of Under Predictions vs Date.  Average = '+str(error_df['NumberUnderPrediction'].mean()))
    fig.show()  

    # Plot Number of Over Predictions
    fig = go.Figure()
    fig.add_trace(go.Line(x = error_df.index,             
                        y = error_df['NumberOverPrediction'],
                        mode = 'lines+markers',
                        name = 'Number of Over Predictions'))   
    fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Number of Over Predictions', title = 'Number of Over Predictions vs Date.  Average = '+str(error_df['NumberOverPrediction'].mean()))
    fig.show()

    # Plot Error Over Predictions
    fig = go.Figure()
    fig.add_trace(go.Line(x = error_df.index,             
                        y = error_df['ErrorOverPrediction'],
                        mode = 'lines+markers',
                        name = 'Error Over Predictions'))   
    fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Error Over Predictions', title = 'Error Over Predictions vs Date.  Average = '+str(error_df['ErrorOverPrediction'].mean(skipna=True))+'%')
    fig.show()

    # Plot Error Under Predictions
    fig = go.Figure()
    fig.add_trace(go.Line(x = error_df.index,             
                        y = error_df['ErrorUnderPrediction'],
                        mode = 'lines+markers',     
                        name = 'Error Under Predictions'))   
    fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Error Under Predictions', title = 'Error Under Predictions vs Date.  Average = '+str(error_df['ErrorUnderPrediction'].mean(skipna=True))+'%')
    fig.show()

    # --- Print distribution of errors ---
    total_days = len(error_df)
    over_mask = error_df['ErrorOverPrediction'] >= 10
    under_mask = error_df['ErrorUnderPrediction'] <= -5

    over_count = over_mask.sum()
    under_count = under_mask.sum()
    both_count = (over_mask & under_mask).sum()
    either_count = (over_mask | under_mask).sum()

    print(f"Over-prediction >= 10%: {over_count} days ({over_count/total_days*100:.2f}%)")
    print(f"Under-prediction <= -5%: {under_count} days ({under_count/total_days*100:.2f}%)")
    print(f"Both conditions: {both_count} days ({both_count/total_days*100:.2f}%)")
    print(f"Either condition: {either_count} days ({either_count/total_days*100:.2f}%)")
    

def plot_one_day(error_df, Y_pred_corrected, Y_pred_uncorrected, Y_true_noised, test_dataset,date):
    
    df = error_df.copy()
    start_date = test_dataset.sample_dates[0]
    Y_pred_corrected = Y_pred_corrected.copy()
    Y_pred_uncorrected = Y_pred_uncorrected.copy()
    Y_true_noised = Y_true_noised.copy()

    i = (pd.to_datetime(date) - pd.to_datetime(start_date)).days
    x = pd.date_range(start=pd.to_datetime(date), periods=30)
    fig = go.Figure()
    uncorrected_mape = df["UncorrectedMape"][date]
    corrected_mape = df["CorrectedMape"][date]

    fig.add_trace(go.Scatter(x=x,
                            y=Y_pred_corrected[i],
                            mode='lines+markers',
                            name='Prediction',
                            legendgroup='lines',
                            legend='legend'))
    fig.add_trace(go.Scatter(x=x,
                            y=Y_pred_uncorrected[i],
                            mode='lines+markers',
                            name='Uncorrected Prediction',
                            legendgroup='lines',
                            legend='legend'))
    fig.add_trace(go.Scatter(x=x,
                            y=Y_true_noised[i],
                            mode='lines+markers',
                            name='True values',
                            legendgroup='lines',
                            legend='legend'))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                            mode='markers',
                            marker=dict(opacity=0),
                            showlegend=True,
                            legendgroup='mape',
                            legend='legend2',
                            name=f'Uncorrected MAPE: {uncorrected_mape:.3f}<br>Corrected MAPE:   {corrected_mape:.3f}'))
    fig.update_layout(
        title=f'Prediction at {date}',
        legend=dict(
            title='Legend',
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
        ),
        legend2=dict(
            title='MAPE',
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
        )
    )
    fig.show()


    prediction_errors = (Y_pred_corrected - Y_true_noised)*100/Y_true_noised
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                            y=prediction_errors[i], 
                            mode='lines+markers', 
                            name=f'Prediction_error'))

    # Add horizontal line at y = -5 (underprediction threshold)
    fig.add_shape(
        type='line',
        x0=min(x),
        x1=max(x),
        y0=-5,
        y1=-5,
        line=dict(color='red', dash='dash'),
        name='Underprediction Limit'
    )

    # Add horizontal line at y = +10 (overprediction threshold)
    fig.add_shape(
        type='line',
        x0=min(x),
        x1=max(x),
        y0=10,
        y1=10,
        line=dict(color='green', dash='dash'),
        name='Overprediction Limit'
    )
    fig.add_annotation(x=max(x), y=-5, text='-5% limit', showarrow=False, yshift=10, font=dict(color='red'))
    fig.add_annotation(x=max(x), y=10, text='+10% limit', showarrow=False, yshift=10, font=dict(color='green'))

    fig.update_layout(
        title=f"Prediction Error with Tolerance Bands. Over Prediction Error:{df['ErrorOverPrediction'][date]:.3f}, Under Prediction Error:{df['ErrorUnderPrediction'][date]:.3f}",
        yaxis_title='Percentage Error (%)',
        xaxis_title='Date or Index'
    )
    fig.show()

def plot_monthly_error_distribution(error_df):
    monthly_error_df = error_df.copy().resample('ME').mean()
    monthly_error_df['YearMonth'] = monthly_error_df.index.to_period('M').strftime('%B')
    monthly_error_df['CorrectedMape'] = monthly_error_df['CorrectedMape'].round(2)
    monthly_error_df['ErrorOverPrediction'] = monthly_error_df['ErrorOverPrediction'].round(2)
    monthly_error_df['ErrorUnderPrediction'] = monthly_error_df['ErrorUnderPrediction'].round(2)

    # --- Bar Chart 1: Corrected MAPE ---
    fig_mape = go.Figure()

    fig_mape.add_trace(
        go.Bar(
            x=monthly_error_df["YearMonth"],
            y=monthly_error_df["CorrectedMape"],
            name="Corrected MAPE",
            text=monthly_error_df["CorrectedMape"],
            textposition="outside"
        )
    )

    fig_mape.update_layout(
        title="Monthly Corrected MAPE",
        xaxis_title="Month",
        yaxis_title="Corrected MAPE",
        bargap=0.2
    )

    fig_mape.show()


    # --- Bar Chart 2: Over & Under Prediction Errors ---
    fig_errors = go.Figure()

    fig_errors.add_trace(
        go.Bar(
            x=monthly_error_df["YearMonth"],
            y=monthly_error_df["ErrorOverPrediction"],
            name="Over-Prediction Error",
            text=monthly_error_df["ErrorOverPrediction"],
            textposition="outside"
        )
    )

    fig_errors.add_trace(
        go.Bar(
            x=monthly_error_df["YearMonth"],
            y=monthly_error_df["ErrorUnderPrediction"],
            name="Under-Prediction Error",
            text=monthly_error_df["ErrorUnderPrediction"],
            textposition="outside"
        )
    )

    fig_errors.update_layout(
        title="Monthly Over & Under Prediction Errors",
        xaxis_title="Month",
        yaxis_title="Prediction Error",
        barmode="group",
        bargap=0.2
    )

    fig_errors.show()

def print_error_distribution(
    values,
    labels=["<3%", "3-5%", "5-7%", "7-10%", "10-15%", "15-20%", ">20%"],
    bins=[0, 3, 5, 7, 10, 15, 20, np.inf],
    title=None,
    abs_values=False
):
    if abs_values:
        values = np.abs(values)
        bins=[0, 1, 3, 5, 7, 10, 15, np.inf]
        labels=["<1%", "1-3%", "3-5%", "5-7%", "7-10%", "10-15%", ">15%"]

    counts, _ = np.histogram(values, bins=bins)
    total = len(values)
    percentages = (counts / total) * 100

    cumulative_counts = np.cumsum(counts)
    cumulative_percentages = np.cumsum(percentages)

    table_data = []
    for label, count, percent, cum_count, cum_percent in zip(
        labels, counts, percentages, cumulative_counts, cumulative_percentages
    ):
        table_data.append([
            label,
            count,
            f"{percent:.2f}%",
            cum_count,
            f"{cum_percent:.2f}%"
        ])

    if title:
        print(f"\n{title}")

    print(tabulate(
        table_data,
        headers=["Range", "Count", "Percentage", "Cumulative Count", "Cumulative %"],
        tablefmt="grid"
    ))

def tabulate_error_distribution(error_df):
    print_error_distribution(
        values=error_df.copy()["CorrectedMape"].values,
        title="Daily Corrected MAPE Distribution"
    )

    print_error_distribution(
        values=error_df.copy()["ErrorOverPrediction"].values,
        title="Daily Over-Prediction Error Distribution"
    )

    print_error_distribution(
        values=error_df.copy()["ErrorUnderPrediction"].values,
        title="Daily Under-Prediction Error Distribution",
        abs_values=True
    )
    print_error_distribution(
        values=error_df.copy().resample("W").mean()["CorrectedMape"].values,
        title="Weekly Corrected MAPE Distribution"
    )
    print_error_distribution(
        values=error_df.copy().resample("W").mean()["ErrorOverPrediction"].values,
        title="Weekly Over-Prediction Error Distribution"
    )
    print_error_distribution(
        values=error_df.copy().resample("W").mean()["ErrorUnderPrediction"].values,
        title="Weekly Under-Prediction Error Distribution",
        abs_values=True
    )