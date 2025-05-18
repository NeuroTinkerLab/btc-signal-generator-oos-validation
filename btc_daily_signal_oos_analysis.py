# -*- coding: utf-8 -*-
"""
Generatore Segnali Giornalieri BTC + Storico Trade OOS (v12.0)
Combina strategie OB_Long e MACD_Bearish_D16.
Mostra trade OOS passati e genera segnale per domani.
"""
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import time
from scipy.signal import find_peaks
import warnings
from datetime import datetime, timedelta # Importa datetime e timedelta

# Ignora specifici FutureWarning e SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None # default='warn'

# --- CONFIGURAZIONE GLOBALE ---
EXCHANGE_ID = 'binance'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE_STR = '2017-01-01T00:00:00Z' # Scarica storico completo per backtest
FETCH_LIMIT = 1500
INITIAL_CAPITAL = 10000
COMMISSION_PCT = 0.001 # 0.1%

# --- DATE PER DIVISIONE IN-SAMPLE / OUT-OF-SAMPLE ---
OOS_START_DATE = '2023-01-01'

# --- CONFIGURAZIONI DA VALIDARE/USARE ---
CONFIGS = [
    {
        'name': 'OB_Long_ATR_2.5SL_4.0TP',
        'signal_type': 'overbought_long',
        'indicator_params': { 'rsi_period': 14, 'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth_k': 3, 'macd_fast': 0, 'atr_period': 14 },
        'strategy_params': { 'rsi_overbought': 70, 'stoch_overbought': 80, 'sl_atr_multiplier': 2.5, 'tp_atr_multiplier': 4.0 },
        'sl_atr_multiplier': 2.5, 'tp_atr_multiplier': 4.0,
    },
    {
        'name': 'MACD_Bearish_D16_SL1.5_TP1.5',
        'signal_type': 'indicator_bear_div',
        'indicator_key': 'MACD', 'distance': 16, 'prominence': None,
        'indicator_params': { 'rsi_period': 0, 'stoch_k': 0, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'atr_period': 14 },
        'strategy_params': { 'sl_atr_multiplier': 1.5, 'tp_atr_multiplier': 1.5, 'macd_confirm_bull_above_zero': False, 'macd_confirm_bear_below_zero': False },
        'sl_atr_multiplier': 1.5, 'tp_atr_multiplier': 1.5,
    },
]

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- FUNZIONI ---
# (Riutilizziamo fetch_ohlcv_data, calculate_indicators, _find_divergences,
#  generate_signals, run_backtest, calculate_performance_metrics da v10.1)

def fetch_ohlcv_data(exchange_id, symbol, timeframe, since, limit):
    # ... (Codice identico da v10.1) ...
    """Scarica dati OHLCV storici da un exchange usando ccxt."""
    logging.info(f"Tentativo download dati per {symbol}...")
    df = pd.DataFrame() # Inizializza df vuoto
    used_symbol = symbol # Tieni traccia del simbolo effettivamente usato
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'rateLimit': 1200, 'enableRateLimit': True})
        try:
             exchange.load_markets(reload=False)
        except ccxt.BaseError:
             exchange.load_markets()

        original_symbol = symbol
        if symbol not in exchange.markets:
            alt_symbol = None
            if symbol == 'MATIC/USDT' and 'POLYGON/USDT' in exchange.markets: alt_symbol = 'POLYGON/USDT'

            if alt_symbol:
                logging.warning(f"Simbolo {symbol} non trovato, provo con {alt_symbol}")
                used_symbol = alt_symbol
            else:
                 logging.error(f"Simbolo {symbol} (e alternative) non trovato su {exchange_id}. Salto.")
                 return pd.DataFrame(), original_symbol

        symbol = used_symbol

        ohlcv_data = []
        current_since = exchange.parse8601(since)
        now = exchange.milliseconds()

        max_retries = 3
        retries = 0
        while current_since < now and retries < max_retries:
            try:
                chunk = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
                if not chunk:
                    logging.info(f"Nessun altro dato ricevuto per {symbol} da {exchange.iso8601(current_since)}. Download completato.")
                    break

                ohlcv_data.extend(chunk)
                last_timestamp = chunk[-1][0]
                current_since = last_timestamp + exchange.parse_timeframe(timeframe) * 1000
                time.sleep(exchange.rateLimit / 1000)
                retries = 0

                if len(ohlcv_data) > limit and ohlcv_data[-limit][0] == last_timestamp:
                    logging.warning(f"Potenziale loop infinito rilevato per {symbol}. Interruzione.")
                    break

            except ccxt.RateLimitExceeded as e:
                logging.warning(f"Rate limit superato per {symbol}: {e}. Attesa...")
                time.sleep(5 + retries * 5); retries += 1
            except ccxt.NetworkError as e:
                logging.warning(f"Errore di rete per {symbol}: {e}. Attesa e riprova...")
                time.sleep(10 + retries * 10); retries += 1
            except ccxt.ExchangeError as e:
                logging.error(f"Errore Exchange per {symbol}: {e}. Interruzione download.")
                return pd.DataFrame(), used_symbol
            except Exception as e:
                logging.error(f"Errore imprevisto fetch per {symbol}: {e}. Interruzione download.")
                return pd.DataFrame(), used_symbol

        if retries >= max_retries:
            logging.error(f"Max tentativi raggiunto per {symbol}. Download fallito.")
            return pd.DataFrame(), used_symbol

        if not ohlcv_data:
             logging.warning(f"Nessun dato OHLCV scaricato per {symbol}.")
             return pd.DataFrame(), used_symbol

        logging.info(f"Download per {symbol} completato. Totale candele: {len(ohlcv_data)}")
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        return df, used_symbol

    except AttributeError:
        logging.error(f"Exchange '{exchange_id}' non supportato.")
        return pd.DataFrame(), symbol
    except ValueError as ve:
        logging.error(f"Errore valore (es. data) per {symbol}: {ve}")
        return pd.DataFrame(), symbol
    except Exception as e:
        logging.error(f"Errore grave fetch data per {symbol}: {e}", exc_info=True)
        return pd.DataFrame(), symbol


def calculate_indicators(df, indicator_params_list, strategy_params_list): # Accetta liste
    # ... (Codice CORRETTO da v10.1) ...
    """Calcola tutti gli indicatori potenzialmente necessari (RSI, Stoch, MACD, ATR)."""
    logging.debug(f"Calcolo indicatori base...")
    df_out = df.copy()

    # Trova i parametri massimi/necessari da tutte le configurazioni
    rsi_p = max([p.get('rsi_period', 0) for p in indicator_params_list] + [0])
    stoch_k = max([p.get('stoch_k', 0) for p in indicator_params_list] + [0])
    stoch_d = max([p.get('stoch_d', 0) for p in indicator_params_list] + [3])
    stoch_sk = max([p.get('stoch_smooth_k', 0) for p in indicator_params_list] + [3])
    macd_f = max([p.get('macd_fast', 0) for p in indicator_params_list] + [0])
    macd_s = max([p.get('macd_slow', 0) for p in indicator_params_list] + [26])
    macd_sig = max([p.get('macd_signal', 0) for p in indicator_params_list] + [9])
    atr_p = max([p.get('atr_period', 0) for p in indicator_params_list] + [14])

    try:
        # RSI
        if rsi_p > 0:
            df_out.loc[:, 'RSI'] = ta.rsi(df_out['Close'], length=rsi_p)

        # Stocastico
        if stoch_k > 0:
            k_col, d_col = f'STOCHk_{stoch_k}_{stoch_d}_{stoch_sk}', f'STOCHd_{stoch_k}_{stoch_d}_{stoch_sk}'
            stoch_df = ta.stoch(df_out['High'], df_out['Low'], df_out['Close'], k=stoch_k, d=stoch_d, smooth_k=stoch_sk, append=False)
            if k_col in stoch_df.columns: df_out.loc[:, 'Stoch_K'] = stoch_df[k_col]
            if d_col in stoch_df.columns: df_out.loc[:, 'Stoch_D'] = stoch_df[d_col]

        # MACD
        if macd_f > 0:
            macd_df = ta.macd(df_out['Close'], fast=macd_f, slow=macd_s, signal=macd_sig, append=False)
            macd_col = f'MACD_{macd_f}_{macd_s}_{macd_sig}'
            if macd_col in macd_df.columns: df_out.loc[:, 'MACD'] = macd_df[macd_col]
            hist_col = f'MACDh_{macd_f}_{macd_s}_{macd_sig}'
            signal_col = f'MACDs_{macd_f}_{macd_s}_{macd_sig}'
            if hist_col in macd_df.columns: df_out.loc[:, 'MACD_Hist'] = macd_df[hist_col]
            if signal_col in macd_df.columns: df_out.loc[:, 'MACD_Signal'] = macd_df[signal_col]

        # ATR
        if atr_p > 0:
            df_out.loc[:, 'ATR'] = ta.atr(df_out['High'], df_out['Low'], df_out['Close'], length=atr_p)
        else:
             logging.error("ATR Period non valido o mancante, ma è richiesto.")
             return pd.DataFrame()

        logging.debug("Indicatori base calcolati.")
        required_cols = ['Open', 'High', 'Low', 'Close', 'ATR']
        if rsi_p > 0: required_cols.append('RSI')
        if stoch_k > 0: required_cols.append('Stoch_K')
        if macd_f > 0: required_cols.append('MACD')

        cols_present = [col for col in required_cols if col in df_out.columns]
        if not all(item in cols_present for item in required_cols):
             missing = [c for c in required_cols if c not in cols_present]
             logging.error(f"Colonne richieste mancanti dopo calcolo: {missing}.")
             return pd.DataFrame()

        df_cleaned = df_out.dropna(subset=cols_present)

        min_len_needed = max(rsi_p, stoch_k, macd_s, atr_p) + 5
        if len(df_cleaned) < min_len_needed:
            logging.warning(f"Dati insufficienti dopo dropna ({len(df_cleaned)} < {min_len_needed}).")
            return pd.DataFrame()
        logging.debug(f"DataFrame ridotto a {len(df_cleaned)} righe dopo dropna.")
        return df_cleaned

    except Exception as e:
        logging.error(f"Errore durante il calcolo degli indicatori: {e}", exc_info=True)
        return pd.DataFrame()


def _find_divergences(price_series, indicator_series, peak_distance, peak_prominence, is_bullish=True):
    # ... (Codice identico da v6.1) ...
    """Helper per trovare divergenze (logica migliorata)."""
    divergence_points = []
    if price_series is None or indicator_series is None: return pd.DatetimeIndex([])
    price_idx = price_series.index
    ind_idx = indicator_series.index
    common_idx = price_idx.intersection(ind_idx)

    if common_idx.empty or len(common_idx) < peak_distance * 2:
        return pd.DatetimeIndex([])

    price = price_series.loc[common_idx]
    indicator = indicator_series.loc[common_idx]

    if price.empty or indicator.empty:
        return pd.DatetimeIndex([])

    try:
        prominence_value = peak_prominence if peak_prominence is not None else None
        prominence_indicator = prominence_value

        if is_bullish:
            price_peaks_indices, _ = find_peaks(-price, distance=peak_distance, prominence=None)
            indicator_peaks_indices, _ = find_peaks(-indicator, distance=peak_distance, prominence=prominence_indicator)
        else: # Bearish
            price_peaks_indices, _ = find_peaks(price, distance=peak_distance, prominence=None)
            indicator_peaks_indices, _ = find_peaks(indicator, distance=peak_distance, prominence=prominence_indicator)
    except Exception as e:
        logging.error(f"Errore in find_peaks: {e}")
        return pd.DatetimeIndex([])

    if len(price_peaks_indices) < 2 or len(indicator_peaks_indices) < 2:
        return pd.DatetimeIndex([])

    price_peak_dates = price.index[price_peaks_indices]
    indicator_peak_dates = indicator.index[indicator_peaks_indices]

    for i in range(1, len(price_peak_dates)):
        date1_price, date2_price = price_peak_dates[i-1], price_peak_dates[i]
        if date1_price not in price.index or date2_price not in price.index: continue
        price1, price2 = price.loc[date1_price], price.loc[date2_price]

        potential_ind_peaks1 = indicator_peak_dates[indicator_peak_dates <= date1_price]
        potential_ind_peaks2 = indicator_peak_dates[(indicator_peak_dates > date1_price) & (indicator_peak_dates <= date2_price + pd.Timedelta(days=peak_distance//2))]

        if not potential_ind_peaks1.empty and not potential_ind_peaks2.empty:
            date1_indicator = potential_ind_peaks1[-1]
            time_diffs = np.abs(potential_ind_peaks2.to_series() - date2_price)
            if not time_diffs.empty:
                 idx_closest = (time_diffs.astype(np.int64)).idxmin()
                 date2_indicator = idx_closest
            else: continue

            if date1_indicator not in indicator.index or date2_indicator not in indicator.index: continue
            if date1_indicator < date2_indicator:
                 indicator1, indicator2 = indicator.loc[date1_indicator], indicator.loc[date2_indicator]

                 if is_bullish:
                     if price2 < price1 and indicator2 > indicator1:
                         divergence_points.append(date2_price)
                 else: # Bearish
                     if price2 > price1 and indicator2 < indicator1:
                         divergence_points.append(date2_price)

    return pd.DatetimeIndex(list(set(divergence_points))).sort_values()


def generate_signals(df, config):
    # ... (Codice CORRETTO da v10.1) ...
    """Genera segnali per OB_Long o Divergenza MACD."""
    logging.debug(f"Generazione segnali per: {config.get('name', 'Unnamed')}")
    df_sig = df.copy()
    df_sig['Signal'] = 0
    signal_type = config.get('signal_type', '')
    strat_params = config.get('strategy_params', {})

    if signal_type == 'overbought_long':
        rsi_ob = strat_params.get('rsi_overbought', 70)
        stoch_ob = strat_params.get('stoch_overbought', 80)
        if 'RSI' not in df_sig.columns or 'Stoch_K' not in df_sig.columns:
            logging.warning("OB_Long: Colonne RSI/Stoch_K mancanti.")
            return df_sig
        is_overbought = (df_sig['RSI'] > rsi_ob) & (df_sig['Stoch_K'] > stoch_ob)
        df_sig.loc[is_overbought.shift(1).fillna(False), 'Signal'] = 1

    elif signal_type == 'indicator_bull_div' or signal_type == 'indicator_bear_div':
        indicator_key = config.get('indicator_key')
        if indicator_key not in df_sig.columns:
            logging.warning(f"Div {indicator_key}: Indicatore mancante.")
            return df_sig

        indicator_series = df_sig[indicator_key]
        peak_distance = config.get('distance')
        peak_prominence = config.get('prominence')
        is_bullish = 'bull' in signal_type.lower()

        if is_bullish:
            div_points = _find_divergences(df_sig['Low'], indicator_series, peak_distance, peak_prominence, is_bullish=True)
        else:
            div_points = _find_divergences(df_sig['High'], indicator_series, peak_distance, peak_prominence, is_bullish=False)

        valid_div_points = df_sig.index.intersection(div_points)

        if not valid_div_points.empty:
            condition_at_div = pd.Series(True, index=valid_div_points) # Default
            confirm_above_zero = strat_params.get('macd_confirm_bull_above_zero', False)
            confirm_below_zero = strat_params.get('macd_confirm_bear_below_zero', False)
            if is_bullish and confirm_above_zero and 'MACD' in df_sig.columns: condition_at_div &= (df_sig.loc[valid_div_points, 'MACD'] > 0)
            if not is_bullish and confirm_below_zero and 'MACD' in df_sig.columns: condition_at_div &= (df_sig.loc[valid_div_points, 'MACD'] < 0)

            valid_div_points = valid_div_points[condition_at_div]
            entry_days = df_sig.index.intersection([idx + pd.Timedelta(days=1) for idx in valid_div_points])
            if not entry_days.empty:
                 signal_value = 1 if is_bullish else -1
                 df_sig.loc[entry_days, 'Signal'] = signal_value
                 logging.debug(f"[{config.get('name')}] Trovate {len(entry_days)} divergenze {'rialziste' if is_bullish else 'ribassiste'} valide.") # Log debug

    else:
        logging.warning(f"Tipo segnale non riconosciuto in generate_signals: {signal_type}")

    num_signals = (df_sig['Signal'] != 0).sum()
    if num_signals > 0: logging.debug(f"Generati {num_signals} segnali per {config.get('name')}")
    return df_sig


def run_backtest(df, config):
    # ... (Codice identico da v10.1/v8.0/v6.1) ...
    """Esegue il backtest per una data configurazione (LONG/SHORT)."""
    start_time_backtest = time.time()
    df_backtest = df.copy()
    signal_col = 'Signal'

    if df_backtest.empty or signal_col not in df_backtest.columns:
         logging.warning(f"DataFrame vuoto o senza segnali per '{config.get('name', 'Unnamed')}'.")
         df_out = pd.DataFrame(index=df.index if not df.empty else None)
         df_out['Equity_Curve'] = INITIAL_CAPITAL; df_out['Strategy_Return'] = 0.0
         df_out['Asset_Return'] = df['Close'].pct_change().fillna(0) if 'Close' in df.columns else 0.0
         return df_out, []


    if df_backtest[signal_col].eq(0).all():
        df_backtest['Equity_Curve'] = INITIAL_CAPITAL; df_backtest['Strategy_Return'] = 0.0
        if 'Asset_Return' not in df_backtest.columns: df_backtest['Asset_Return'] = df_backtest['Close'].pct_change().fillna(0)
        return df_backtest, []

    position = 0; entry_price = 0.0; entry_date = None; stop_loss_price = 0.0; take_profit_price = 0.0
    trades = []
    sl_atr_multiplier = config.get('sl_atr_multiplier')
    tp_atr_multiplier = config.get('tp_atr_multiplier')

    for i in range(len(df_backtest)):
        current_index = df_backtest.index[i]; current_row = df_backtest.iloc[i]; current_signal = current_row[signal_col]
        exit_today = False; exit_price = np.nan

        if position != 0:
            if (position == 1 and current_row['Low'] <= stop_loss_price) or \
               (position == -1 and current_row['High'] >= stop_loss_price):
                exit_price = stop_loss_price; exit_today = True; exit_reason = "SL"
            elif (position == 1 and current_row['High'] >= take_profit_price) or \
                 (position == -1 and current_row['Low'] <= take_profit_price):
                exit_price = take_profit_price; exit_today = True; exit_reason = "TP"

            if exit_today:
                if position == 1: pnl_pct = (exit_price / entry_price) - 1 if entry_price != 0 else 0.0
                else: pnl_pct = (entry_price / exit_price) - 1 if exit_price != 0 else -1.0
                commission = COMMISSION_PCT * 2; net_pnl_pct = pnl_pct - commission
                trades.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': current_index, 'exit_price': exit_price, 'type': 'LONG' if position == 1 else 'SHORT', 'pnl_pct': pnl_pct, 'net_pnl_pct': net_pnl_pct, 'exit_reason': exit_reason, 'config_name': config.get('name')})
                position = 0; entry_price = 0.0

        if position == 0 and not exit_today and current_signal != 0:
            position = int(current_signal); entry_price = current_row['Open']
            if entry_price <= 0: position = 0; continue
            entry_date = current_index
            current_atr = current_row.get('ATR', np.nan) if 'ATR' in current_row else np.nan

            if pd.notna(current_atr) and sl_atr_multiplier is not None and tp_atr_multiplier is not None:
                 if position == 1:
                     stop_loss_price = entry_price - (sl_atr_multiplier * current_atr)
                     take_profit_price = entry_price + (tp_atr_multiplier * current_atr)
                 else:
                     stop_loss_price = entry_price + (sl_atr_multiplier * current_atr)
                     take_profit_price = entry_price - (tp_atr_multiplier * current_atr)
                 # Validazione
                 if (position == 1 and stop_loss_price >= entry_price) or (position == -1 and stop_loss_price <= entry_price): stop_loss_price = -np.inf if position == 1 else np.inf
                 if (position == 1 and take_profit_price <= entry_price) or (position == -1 and take_profit_price >= entry_price): take_profit_price = np.inf if position == 1 else -np.inf
            else:
                 stop_loss_price = -np.inf if position == 1 else np.inf
                 take_profit_price = np.inf if position == 1 else -np.inf

    # Calcola Equity Curve
    if trades:
        trades_df = pd.DataFrame(trades).set_index('exit_date')
        trades_df = trades_df[~trades_df.index.duplicated(keep='first')]
        daily_returns = trades_df['net_pnl_pct'].reindex(df_backtest.index, fill_value=0.0)
        df_backtest['Trade_Return'] = daily_returns
    else: df_backtest['Trade_Return'] = 0.0

    df_backtest['Equity_Factor'] = 1.0 + df_backtest['Trade_Return']
    df_backtest['Equity_Curve'] = INITIAL_CAPITAL * df_backtest['Equity_Factor'].cumprod()
    df_backtest['Equity_Curve'] = df_backtest['Equity_Curve'].clip(lower=1e-9)

    df_backtest['Strategy_Return'] = df_backtest['Equity_Curve'].pct_change().fillna(0)
    if 'Asset_Return' not in df_backtest.columns:
         df_backtest['Asset_Return'] = df_backtest['Close'].pct_change().fillna(0)

    end_time_backtest = time.time()
    return df_backtest, trades


def calculate_performance_metrics(equity_curve, trades, asset_returns, config_name="Strategy", config=None):
    # ... (Codice identico da v10.1/v8.0/v6.1) ...
    """Calcola le metriche di performance della strategia."""
    logging.debug(f"Calcolo metriche per: {config_name}")
    metrics = {'Name': config_name}
    if config:
         metrics['signal_type'] = config.get('signal_type')
         if 'indicator_div' in metrics['signal_type']:
             metrics['indicator_key'] = config.get('indicator_key')
             metrics['distance'] = config.get('distance')
             metrics['prominence'] = config.get('prominence', -1.0)
         metrics['sl_atr'] = config.get('sl_atr_multiplier')
         metrics['tp_atr'] = config.get('tp_atr_multiplier')

    if equity_curve is None or equity_curve.empty or asset_returns is None or asset_returns.empty or equity_curve.iloc[0] <= 0:
         logging.warning(f"Dati insufficienti o non validi per calcolare metriche per {config_name}")
         metrics.update({'CAGR (%)': np.nan, 'Sharpe Ratio': np.nan, 'Max Drawdown (%)': np.nan, 'Num Trades': 0, 'Error': 'Input Data Invalid'})
         return metrics

    try:
        total_return_strategy = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics['Total Return (%)'] = total_return_strategy * 100
        start_date, end_date = equity_curve.index[0], equity_curve.index[-1]
        years = max((end_date - start_date).days / 365.25, 1/365.25)
        if equity_curve.iloc[-1] > 1e-9 and equity_curve.iloc[0] > 1e-9:
            metrics['CAGR (%)'] = ((equity_curve.iloc[-1] / equity_curve.iloc[0])**(1/years) - 1) * 100
        else:
            metrics['CAGR (%)'] = -100.0 if equity_curve.iloc[-1] <= 1e-9 else 0.0

        strategy_daily_returns = equity_curve.pct_change().fillna(0)
        volatility_strategy = strategy_daily_returns.std() * np.sqrt(365)
        metrics['Volatility (%)'] = volatility_strategy * 100
        if volatility_strategy > 1e-9:
             metrics['Sharpe Ratio'] = metrics['CAGR (%)'] / 100 / volatility_strategy if metrics['CAGR (%)'] is not None else 0.0
        else: metrics['Sharpe Ratio'] = 0.0

        rolling_max = equity_curve.cummax()
        daily_drawdown = (equity_curve / rolling_max) - 1
        metrics['Max Drawdown (%)'] = daily_drawdown.min() * 100

        if trades:
            trades_df = pd.DataFrame(trades)
            metrics['Num Trades'] = len(trades_df)
            long_trades = trades_df[trades_df['type'] == 'LONG']
            short_trades = trades_df[trades_df['type'] == 'SHORT']
            winning_trades = trades_df[trades_df['net_pnl_pct'] > 0]
            losing_trades = trades_df[trades_df['net_pnl_pct'] <= 0]
            metrics['Num Long Trades'] = len(long_trades); metrics['Num Short Trades'] = len(short_trades)
            if len(trades_df) > 0: metrics['Win Rate (%)'] = (len(winning_trades) / len(trades_df)) * 100
            else: metrics['Win Rate (%)'] = 0.0
            if not winning_trades.empty: metrics['Avg Win (%)'] = winning_trades['net_pnl_pct'].mean() * 100
            else: metrics['Avg Win (%)'] = 0.0
            if not losing_trades.empty: metrics['Avg Loss (%)'] = losing_trades['net_pnl_pct'].mean() * 100
            else: metrics['Avg Loss (%)'] = 0.0
            gross_profit = winning_trades['net_pnl_pct'].sum(); gross_loss = abs(losing_trades['net_pnl_pct'].sum())
            if gross_loss > 1e-9: metrics['Profit Factor'] = gross_profit / gross_loss
            elif gross_profit > 1e-9: metrics['Profit Factor'] = np.inf
            else: metrics['Profit Factor'] = 0.0
            if len(long_trades) > 0: metrics['Win Rate Long (%)'] = (len(long_trades[long_trades['net_pnl_pct'] > 0]) / len(long_trades)) * 100
            else: metrics['Win Rate Long (%)'] = np.nan
            if len(short_trades) > 0: metrics['Win Rate Short (%)'] = (len(short_trades[short_trades['net_pnl_pct'] > 0]) / len(short_trades)) * 100
            else: metrics['Win Rate Short (%)'] = np.nan
        else:
             metrics.update({'Num Trades': 0, 'Num Long Trades': 0, 'Num Short Trades': 0,'Win Rate (%)': 0, 'Avg Win (%)': 0, 'Avg Loss (%)': 0,'Profit Factor': 0, 'Win Rate Long (%)': np.nan, 'Win Rate Short (%)': np.nan})

        logging.debug(f"Metriche calcolate per: {config_name}")
        return metrics

    except Exception as e:
        logging.error(f"Errore calcolo metriche per {config_name}: {e}", exc_info=True)
        metrics['Error'] = 'Metric Calculation Failed'
        return metrics


# --- ESECUZIONE PRINCIPALE (OOS Validation Finale) ---
if __name__ == "__main__":
    logging.info(f"Avvio Validazione OOS Finale per Strategie Candidate su {SYMBOL}...")
    start_run_time = time.time()

    all_oos_results = []
    final_trades_oos = {} # Dizionario per salvare i trade OOS per analisi

    # 1. Scarica Dati Completi
    symbol = SYMBOL
    logging.info(f"--- Download Dati Completi: {symbol} ---")
    ohlcv_df_full, used_symbol = fetch_ohlcv_data(EXCHANGE_ID, symbol, TIMEFRAME, START_DATE_STR, FETCH_LIMIT)

    if ohlcv_df_full is None or ohlcv_df_full.empty:
        logging.critical(f"Dati vuoti/errore fetch per {used_symbol}. Impossibile procedere.")
        exit()
    logging.info(f"Dati completi per {used_symbol} scaricati: {len(ohlcv_df_full)} righe.")

    # 2. Pre-calcola TUTTI gli indicatori necessari
    df_indicators_full = None
    try:
         all_indicator_params = [cfg['indicator_params'] for cfg in CONFIGS]
         all_strategy_params = [cfg['strategy_params'] for cfg in CONFIGS]
         df_indicators_full = calculate_indicators(ohlcv_df_full, all_indicator_params, all_strategy_params)
         if df_indicators_full is None or df_indicators_full.empty:
             raise ValueError("Calcolo indicatori fallito o dati insufficienti.")
    except Exception as e:
         logging.critical(f"Eccezione calcolo indicatori per {used_symbol}: {e}. Uscita.", exc_info=True)
         exit()

    # 3. Ciclo sulle Configurazioni da Validare
    logging.info(f"Inizio validazione su {len(CONFIGS)} configurazioni candidate...")

    # Definisci df_signals_oos fuori dal loop per usarlo nel report finale per la data
    df_signals_oos = pd.DataFrame()

    for config in CONFIGS:
        config_name = config.get('name', 'Unnamed_Config')
        logging.info(f"--- Validazione Config: '{config_name}' ---")

        results_is = {}; results_oos = {}; trades_oos_list = []

        try:
            # A. Genera Segnali
            required_keys = []
            if 'overbought_long' in config['signal_type']: required_keys = ['RSI', 'Stoch_K']
            elif 'indicator_div' in config['signal_type']: required_keys = [config.get('indicator_key')]

            # Verifica indicatori base (già fatto) + specifici
            if not all(k in df_indicators_full.columns for k in required_keys if k):
                 logging.warning(f"Indicatori richiesti {required_keys} non trovati per {config_name}. Salto.")
                 all_oos_results.append({'Name': config_name, 'IS': {'Error':'Missing Indicators'}, 'OOS': {'Error':'Missing Indicators'}})
                 continue

            df_signals_full = generate_signals(df_indicators_full.copy(), config)
            if df_signals_full is None or df_signals_full.empty:
                 all_oos_results.append({'Name': config_name, 'IS': {'Error':'Signal Gen Failed'}, 'OOS': {'Error':'Signal Gen Failed'}})
                 continue

            # B. Dividi Dati
            df_signals_is = df_signals_full[:OOS_START_DATE].iloc[:-1]
            df_signals_oos = df_signals_full[OOS_START_DATE:] # Aggiorna df_signals_oos qui

            if df_signals_is.empty or df_signals_oos.empty:
                 logging.warning(f"Periodo IS o OOS vuoto per {config_name} ({len(df_signals_is)} IS, {len(df_signals_oos)} OOS). Salto.")
                 all_oos_results.append({'Name': config_name, 'IS': {'Error':'Empty Period'}, 'OOS': {'Error':'Empty Period'}})
                 continue

            logging.info(f"Periodo In-Sample (IS): {df_signals_is.index.min().date()} - {df_signals_is.index.max().date()}")
            logging.info(f"Periodo Out-of-Sample (OOS): {df_signals_oos.index.min().date()} - {df_signals_oos.index.max().date()}")


            # C. Backtest IS
            logging.info(f"Esecuzione Backtest In-Sample per '{config_name}'...")
            df_result_is, trades_is = run_backtest(df_signals_is, config)
            if df_result_is is not None and not df_result_is.empty:
                 asset_returns_is = df_result_is.get('Asset_Return', pd.Series(dtype=float))
                 metrics_is = calculate_performance_metrics(df_result_is['Equity_Curve'], trades_is, asset_returns_is, f"{config_name}_IS", config)
                 results_is = metrics_is
            else: logging.warning(f"Backtest IS fallito per {config_name}.")


            # D. Backtest OOS
            logging.info(f"Esecuzione Backtest Out-of-Sample per '{config_name}'...")
            df_result_oos, trades_oos = run_backtest(df_signals_oos, config) # Salva i trade OOS
            if df_result_oos is not None and not df_result_oos.empty:
                 asset_returns_oos = df_result_oos.get('Asset_Return', pd.Series(dtype=float))
                 metrics_oos = calculate_performance_metrics(df_result_oos['Equity_Curve'], trades_oos, asset_returns_oos, f"{config_name}_OOS", config)
                 results_oos = metrics_oos
                 trades_oos_list = trades_oos
            else: logging.warning(f"Backtest OOS fallito per {config_name}.")

            # E. Aggrega e Salva
            all_oos_results.append({'Name': config_name, 'IS': results_is, 'OOS': results_oos})
            final_trades_oos[config_name] = trades_oos_list

        except Exception as e:
             logging.error(f"Errore GRACE durante validazione '{config_name}': {e}", exc_info=True)
             all_oos_results.append({'Name': config_name, 'IS': {'Error': str(e)}, 'OOS': {'Error': str(e)}})


    # --- Report Finale Validazione ---
    print("\n\n--- REPORT VALIDAZIONE OUT-OF-SAMPLE (OOS) ---")
    print(f"Simbolo: {used_symbol}")
    # Usa la data massima reale dal dataframe OOS (se esiste)
    oos_end_date_actual = df_signals_oos.index.max().date() if not df_signals_oos.empty else 'N/A'
    print(f"Periodo OOS: {OOS_START_DATE} -> {oos_end_date_actual}")
    print("-" * 80)

    cols_to_print = ['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Num Trades', 'Win Rate (%)', 'Profit Factor']

    for result in all_oos_results:
        config_name = result['Name']
        print(f"\nConfigurazione: {config_name}")
        print("-" * (len(config_name) + 15))
        metrics_is = result.get('IS', {})
        metrics_oos = result.get('OOS', {})

        print(f"{'Metrica':<20} | {'In-Sample (IS)':<20} | {'Out-of-Sample (OOS)':<20}")
        print("-" * 65)
        for col in cols_to_print:
            val_is = metrics_is.get(col, 'N/A')
            val_oos = metrics_oos.get(col, 'N/A')
            val_is_str = f"{val_is:.2f}" if isinstance(val_is, (int, float)) and pd.notna(val_is) else str(val_is)
            val_oos_str = f"{val_oos:.2f}" if isinstance(val_oos, (int, float)) and pd.notna(val_oos) else str(val_oos)
            print(f"{col:<20} | {val_is_str:<20} | {val_oos_str:<20}")

        # Stampa Dettagli Trade OOS per questa configurazione
        trades_list_for_config = final_trades_oos.get(config_name, [])
        if trades_list_for_config:
            print(f"\n--- Trade Dettagliati OOS per {config_name} ---")
            trades_df = pd.DataFrame(trades_list_for_config)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

            # Filtro data trade corretto e robusto
            if 'df_signals_oos' in locals() and not df_signals_oos.empty:
                oos_period_index = df_signals_oos.index
                trades_in_period = trades_df[(trades_df['exit_date'] >= oos_period_index.min()) & (trades_df['exit_date'] <= oos_period_index.max())]
            else:
                trades_in_period = pd.DataFrame()


            if not trades_in_period.empty:
                trades_in_period = trades_in_period.copy()
                trades_in_period['duration'] = (trades_in_period['exit_date'] - trades_in_period['entry_date']).dt.days
                trades_in_period['entry_date'] = trades_in_period['entry_date'].dt.date
                trades_in_period['exit_date'] = trades_in_period['exit_date'].dt.date
                print(trades_in_period[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'net_pnl_pct', 'duration']].round(4))
            else:
                 print("Nessun trade completato trovato nel periodo OOS.")
            print("-" * 50)
        else:
             print(f"\nNessun trade registrato nel periodo OOS per {config_name}.")


        if 'Error' in metrics_is or 'Error' in metrics_oos:
             print(f"\nErrore IS: {metrics_is.get('Error', 'Nessuno')}")
             print(f"Errore OOS: {metrics_oos.get('Error', 'Nessuno')}")
        print("-" * 80)


    end_run_time = time.time()
    logging.info(f"Validazione OOS completata in {(end_run_time - start_run_time)/60:.2f} minuti.")