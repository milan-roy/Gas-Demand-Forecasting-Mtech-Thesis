import pandas as pd
import os
import numpy as np

def average_decrease(sales_df, date_list):
    """
    Calculates the average percentage decrease in sales on a specific holiday across multiple years.

    For each date in `date_list`, this function computes the percentage decrease in sales 
    compared to the previous day (i.e., `(prev_day_sales - today_sales) / prev_day_sales * 100`),
    and then returns the average of those decreases.

    Parameters
    ----------
    sales_df : pd.DataFrame
        A DataFrame indexed by date with a 'Sales' column containing daily sales values.

    date_list : list of pd.Timestamp
        A list of holiday dates (one for each year) for a specific holiday.

    Returns
    -------
    float or None
        The average percentage decrease in sales across the given dates.
        Returns None if no valid comparisons could be made.
    """

    decreases = []

    for date in date_list[:-1]:  # Exclude the most recent year (e.g., 2024), which is being predicted
        prev_date = date - pd.Timedelta(days=1)

        # Ensure both current and previous day exist in the index
        if date in sales_df.index and prev_date in sales_df.index:
            today_sales = sales_df.loc[date, 'Sales']
            prev_day_sales = sales_df.loc[prev_date, 'Sales']

            if prev_day_sales > 0:  # Prevent division by zero
                percentage_decrease = ((prev_day_sales - today_sales) / prev_day_sales) * 100
                decreases.append(percentage_decrease)

    # Return the average decrease if available
    return sum(decreases) / len(decreases) if decreases else None

def correct_holiday(dates_test, holiday_dates, Y_pred, avg_decrease):
    """
    Applies average percentage decrease to predicted sales for the **last holiday** date only.

    Parameters
    ----------
    - dates_test: list of starting dates for each prediction row (len = 337)
    - holiday_dates: list of holiday dates (only the last one is used)
    - Y_pred: np.ndarray of shape (337, 30)
    - avg_decrease: average percent decrease (float)
    
    Returns
    ----------
    - Modified Y_pred with correction applied to the last holiday date
    """
    dates_test = pd.to_datetime(dates_test)
    target_date = pd.to_datetime(holiday_dates[-1])  # Only last holiday

    for i, start_date in enumerate(dates_test):
        delta_days = (target_date - start_date).days
        j = delta_days  # because prediction starts from day +1

        if j==0:
            Y_pred[i][j] = Y_pred[i-1][j] * (1 - avg_decrease / 100.0)
        elif 0 < j < Y_pred.shape[1]:  # j > 0 to access j-1 safely
            try:
                Y_pred[i][j] = Y_pred[i][j - 1] * (1 - avg_decrease / 100.0)
            except:
                print(start_date)

    return Y_pred

def holiday_correction(Y_pred, dates_test, sales_df, holiday_df):
    """
    Applies holiday-aware corrections to model predictions based on historical
    sales drops during holidays.

    This function uses past sales data to estimate the average decrease in sales
    around specific holiday periods (including days before and after). It adjusts
    the model's predicted values (Y_pred) to reintroduce the effects of these 
    holidays which were removed during training via denoising.

    Parameters
    ----------
    Y_pred : np.ndarray
        A 2D NumPy array of shape (n_days, forecast_horizon), containing the
        model's raw sales predictions from a denoised dataset.
    
    dates_test : pd.DatetimeIndex or list-like
        A sequence of dates corresponding to each row in `Y_pred`. These represent
        the "current" date from which the forecast was made.

    Returns
    -------
    np.ndarray
        A NumPy array of the same shape as `Y_pred`, containing sales predictions
        corrected for holiday effects.
    """
    Y_pred = Y_pred.copy()
    dates_test = dates_test.copy()
    sales_df = sales_df.copy()
    holiday_df = holiday_df.copy()
    sales_df = sales_df.set_index('Date')
    
    # Read holiday dates and convert year columns to datetime
    years = ['2020', '2021', '2022', '2023', '2024']
    holiday_df[years] = holiday_df[years].apply(pd.to_datetime)

    all_holiday_dates = []

    # Generate holiday windows (DaysBefore, Holiday, DaysAfter)
    for _, row in holiday_df.iterrows():
        for offset in range(-row['DaysBefore'], row['DaysAfter'] + 1):
            holiday_dates = [row[year] + pd.Timedelta(days=offset) for year in years]
            all_holiday_dates.append(holiday_dates)

    # Apply correction to each set of holiday dates
    for holiday_dates in all_holiday_dates:
        avg_holiday_decrease = average_decrease(sales_df, holiday_dates)
        Y_pred = correct_holiday(dates_test, holiday_dates, Y_pred, avg_holiday_decrease)

    return Y_pred
