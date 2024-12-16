import numpy as np
import pandas as pd
from scipy.integrate import odeint

from src.config import SIR_DATA_PATH


def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def main():
    # Initial conditions: S0 (susceptible), I0 (infected), R0 (recovered)
    S0 = 999
    I0 = 1
    R0 = 0

    # Total population, N
    N = S0 + I0 + R0

    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta = 0.3
    gamma = 0.1

    # Initial conditions vector
    y0 = S0, I0, R0

    # Time grid (in days)
    t = np.linspace(0, 160, 160)

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    data = pd.DataFrame({'Time': t, 'Susceptible': S, 'Infected': I, 'Recovered': R})

    # Save to a CSV file
    data.to_csv(SIR_DATA_PATH, index=False)


if __name__ == '__main__':
    main()
