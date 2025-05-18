```markdown
# BTC/USDT Daily Signal Generator & Out-of-Sample Performance 

This repository contains the Python script (`btc_daily_signal_oos_analysis.py`) designed to generate daily trading signals for BTC/USDT. It also demonstrates the Out-of-Sample (OOS) performance of the underlying strategies that were developed and optimized through a series of prior backtesting and optimization scripts.

**The primary goal is to showcase a specific, refined approach to signal generation for BTC/USDT on a daily timeframe, combining two distinct strategies: an "Overbought Long" (OB_Long) and a "MACD Bearish Divergence" strategy.**

**🚨 IMPORTANT DISCLAIMER 🚨**
*   **This code and the signals generated are for EDUCATIONAL AND SHOWCASE PURPOSES ONLY.**
*   **This is NOT FINANCIAL ADVICE. DO NOT use for live trading without extensive personal verification, understanding the risks, and adapting it to your own risk tolerance.**
*   The strategies and parameters were derived from historical data and **past performance is not indicative of future results.**
*   The cryptocurrency market is highly volatile. Significant losses are possible.

## Project Evolution & Strategy Selection

This script (`btc_daily_signal_oos_analysis.py`) is the result of an iterative process involving:
1.  Development of a general backtesting framework.
2.  Optimization of parameters for various strategies, including RSI/Stochastic "Overbought Long" conditions and MACD divergence signals, specifically for BTC/USDT.
3.  Rigorous Out-of-Sample (OOS) validation of the most promising optimized configurations to assess their robustness on unseen data.

The two strategies combined in the final signal generator were selected based on their performance and characteristics observed during this optimization and validation process.

## Core Functionality of `btc_daily_signal_oos_analysis.py`

*   **Fetches Historical Data:** Downloads the latest daily OHLCV data for BTC/USDT from Binance.
*   **Calculates Indicators:** Computes necessary technical indicators (RSI, Stochastic, MACD, ATR) based on predefined, optimized parameters.
*   **Generates Combined Signal:**
    *   Checks for an "Overbought Long" entry signal (RSI(14) > 70, Stoch_K(14,3,3) > 80).
    *   If no Long signal, it checks for a "MACD Bearish Divergence" (using Distance 16) SHORT signal.
    *   Outputs a final recommendation: "GO LONG", "GO SHORT", or "STAY FLAT" for the next trading day.
*   **Provides Indicative SL/TP:** Calculates Stop-Loss and Take-Profit levels based on ATR multipliers specific to the triggered strategy.
*   **Displays Historical OOS Trades:** Runs a backtest on the Out-of-Sample period (from 2023-01-01 onwards) for *both* strategies individually to show their historical performance on unseen data up to the latest available data point.

## Out-of-Sample (OOS) Performance Summary

The following results were obtained by running `btc_daily_signal_oos_analysis.py` on **BTC/USDT Daily data**. The Out-of-Sample (OOS) period for this report starts from **2023-01-01 and extends up to 2025-05-18** (the date this report was generated, reflecting the latest data available at that time).

*Initial Capital for backtest simulation: $10,000. Commission: 0.1% per trade side.*

---

### Strategy 1: OB_Long_ATR_2.5SL_4.0TP
*Description: Enters LONG when RSI(14) > 70 and Stoch_K(14,3,3) > 80. Exits on ATR-based SL (2.5x ATR from entry) or TP (4.0x ATR from entry).*

| Metric             | In-Sample (IS) | Out-of-Sample (OOS) |
|--------------------|--------------------|-----------------------|
| CAGR (%)           | 63.66              | 52.79                 |
| Sharpe Ratio       | 1.35               | 1.77                  |
| Max Drawdown (%)   | -32.65             | -11.10                |
| Num Trades         | 33.00              | 14.00                 |
| Win Rate (%)       | 69.70              | 78.57                 |
| Profit Factor      | 3.28               | 4.80                  |

**Detailed OOS Trades for OB_Long_ATR_2.5SL_4.0TP (Period: 2023-01-01 to 2025-05-18):**
```
    entry_date  entry_price   exit_date   exit_price exit_reason  net_pnl_pct  duration
0   2023-01-12     17943.26  2023-01-13   19513.07          TP       0.0855         1
1   2023-01-14     19930.01  2023-01-20   22016.14          TP       0.1027         6
2   2023-01-21     22666.00  2023-03-09   20875.81          SL      -0.0810        47
3   2023-03-18     27395.13  2023-10-23   32470.84          TP       0.1833       219
4   2023-10-24     33069.99  2023-11-09   37837.91          TP       0.1422        16
5   2023-12-04     39972.26  2024-01-02   44874.21          TP       0.1206        29
6   2024-02-10     47132.78  2024-02-26   53061.57          TP       0.1238        16
7   2024-02-27     54476.48  2024-02-28   61350.94          TP       0.1242         1
8   2024-02-29     62432.11  2024-03-11   71244.66          TP       0.1392        11
9   2024-03-12     72078.10  2024-03-19   64218.60          SL      -0.1110         7
10  2024-10-21     69032.00  2024-11-08   76939.57          TP       0.1125        18
11  2024-11-09     76509.78  2024-11-11   86061.69          TP       0.1228         2
12  2024-11-12     88648.00  2024-12-05  101391.60          TP       0.1418        23
13  2024-12-18    106133.74  2024-12-19   95818.93          SL      -0.0992         1
```

---

### Strategy 2: MACD_Bearish_D16_SL1.5_TP1.5
*Description: Enters SHORT on a MACD(12,26,9) bearish divergence (price makes higher high, MACD makes lower high) with a peak distance of 16 candles. Exits on ATR-based SL (1.5x ATR from entry) or TP (1.5x ATR from entry).*

| Metric             | In-Sample (IS) | Out-of-Sample (OOS) |
|--------------------|--------------------|-----------------------|
| CAGR (%)           | 11.58              | 5.83                  |
| Sharpe Ratio       | 0.88               | 0.68                  |
| Max Drawdown (%)   | -11.08             | -5.83                 |
| Num Trades         | 13.00              | 4.00                  |
| Win Rate (%)       | 76.92              | 75.00                 |
| Profit Factor      | 4.04               | 3.45                  |

**Detailed OOS Trades for MACD_Bearish_D16_SL1.5_TP1.5 (Period: 2023-01-01 to 2025-05-18):**
```
   entry_date  entry_price   exit_date  exit_price exit_reason  net_pnl_pct  duration
0  2023-02-17     23517.72  2023-02-19  24921.57          SL      -0.0583         2
1  2023-06-24     30688.51  2023-07-24  29120.14          TP       0.0519        30
2  2024-01-12     46339.16  2024-01-13  42855.95          TP       0.0793         1
3  2025-01-21    102260.00  2025-02-03   95359.59          TP       0.0704        13
```

---

## Prerequisites

*   Python 3.8+
*   Libraries: `ccxt`, `pandas`, `numpy`, `pandas-ta`, `scipy` (for `find_peaks`), `matplotlib` (if re-enabling any plotting).
    (A `requirements.txt` file should ideally list these).

## Setup & Usage

1.  **Clone/Download:** Obtain the `btc_daily_signal_oos_analysis.py` script.
2.  **Install Dependencies (if not already installed):**
    ```bash
    pip install ccxt pandas numpy pandas-ta scipy matplotlib
    ```
3.  **Run the script:**
    ```bash
    python btc_daily_signal_oos_analysis.py
    ```
    The script will:
    *   Fetch the latest BTC/USDT daily data.
    *   Calculate necessary indicators.
    *   Run Out-of-Sample backtests for the predefined strategies and print their performance summaries and recent OOS trades (as shown above).
    *   Check for a combined signal for the *next trading day* and print the final recommendation.

## License

This project is released under the **MIT License**.
*(Ensure you have a LICENSE file with the MIT License text, or your chosen license)*
```

