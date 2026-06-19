import numpy as np
import pandas as pd


np.random.seed(42)

# Load data
sales_df = pd.read_excel("Data/DemandData.xlsx")
holidays_df = pd.read_csv("Data/holidays.csv")

sales_df['Date'] = pd.to_datetime(sales_df['Date'])
sales_df.set_index('Date', inplace=True)

# Filling missing dates in sales data
full_dates = pd.date_range(start=sales_df.index.min(), end=sales_df.index.max(), freq='D')
sales_df = sales_df.reindex(full_dates)
sales_df['Sales'] = sales_df['Sales'].ffill()
sales_df.index.name = 'Date'
sales_df.reset_index(inplace=True)
sales_df.to_excel("Data/DataMissingValuesFilled.xlsx",index=False)

sales_df.set_index('Date', inplace=True)
# Holiday denoising: random + smooth edges
for _, row in holidays_df.iterrows():
    for date in row.iloc[1:-2]:  
        if pd.notna(date):
            date = pd.to_datetime(date)
            start = date - pd.Timedelta(days=row['DaysBefore'])
            end = date + pd.Timedelta(days=row['DaysAfter'])

            # Make sure window is within available date range
            if start < sales_df.index.min() or end > sales_df.index.max():
                continue

            # Get boundary values (just outside holiday window)
            prev_val = sales_df.loc[:start].iloc[-2]['Sales']
            next_val = sales_df.loc[end:].iloc[1]['Sales']

            # Skip if boundaries are missing
            if pd.isna(prev_val) or pd.isna(next_val):
                continue

            # Length of holiday window
            idx = sales_df.loc[start:end].index
            n = len(idx)

            # Random values between edges
            low = min(prev_val, next_val)
            high = max(prev_val, next_val)
            rand_vals = np.random.uniform(low, high, size=n)

            # Smooth transition from prev_val -> next_val
            weights = np.linspace(0, 1, n)
            smooth_vals = prev_val * (1 - weights) + next_val * weights

            # Hybrid fill (50% random + 50% smooth)
            hybrid_vals = 0.5 * rand_vals + 0.5 * smooth_vals

            # Replace holiday window
            sales_df.loc[start:end, 'Sales'] = hybrid_vals

# Reset index if needed later
sales_df.reset_index(inplace=True)

sales_df.to_excel("Data/FixedData_Random_Smoothed.xlsx",index=False)