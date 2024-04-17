import matplotlib.pyplot as plt


def plot_time_series(time_series):
    """
    Plots the given time series.

    Parameters:
        time_series (np.ndarray): The time series to plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Non-Stationary Time Series with Varying Variance')
    plt.title('Non-Stationary Time Series Generation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
