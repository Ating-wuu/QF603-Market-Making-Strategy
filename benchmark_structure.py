# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from tqdm import tqdm
from volmodels import VolatilityModelBase


def round_to_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-12) * tick

def round_to_lot(qty: float, lot: float) -> float:
    return math.floor(qty / lot + 1e-12) * lot


@dataclass
class ExchangeSpec:
    tick_size: float = 0.01
    lot_size: float = 0.001
    maker_fee: float = -0.0002
    taker_fee: float =  0.0007

@dataclass
class ASConfig:
    # A-S param
    gamma: float = 20.0
    k: float = 8.0
    tau: float = 1.0

    # This is to make sure the spread stays within a range
    spread_cap_frac: float = 0.001   # 0.1% of mid
    min_spread_frac: float = 0.0002  # 0.02% of mid

    max_order_notional_frac: float = 0.01  # 1% of equity # max order size for each order
    min_cash_buffer_frac: float = 0.05 # 5% of total cash left in buffer

    vol_scale_k: float = 1.0               # higher -> smaller order qty
    target_inv_frac: float = 0.0           # neutral

    max_inv_frac: float = 0.25             # 25% * Equity
    hard_liq_frac: float = 0.35


# ============ A-S Simulator ============
class MMSimulator:
    """
    feed.step() -> dict:
      {'ts','bid_px','ask_px','bid_qty','ask_qty', 'sigma'(optional)}
    """
    def __init__(self, feed, ex: ExchangeSpec, cfg: ASConfig,
                 initial_cash: float = 100_000.0, step_seconds: float = 1.0,
                 vol_model: Optional[VolatilityModelBase] = None):
        self.feed, self.ex, self.cfg = feed, ex, cfg
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.inv = 0.0  # unit of the trading pair
        self.px_bid = self.px_ask = None
        self.total_fees = 0.0
        self.step_seconds, self._step_idx = step_seconds, 0
        self.history_rows: List[Dict[str, Any]] = []
        self.equity_prev = None
        self.trade_steps: List[int] = []
        self.trade_equity_delta: List[float] = []
        self.vol_model = vol_model

    def equity(self, mid: float) -> float:
        return self.cash + self.inv * mid

    # —— A-S quote ——
    def get_as_quotes(self, mid: float, sigma: float, drift: float = 0.0):
        # sigma is given as relative instead of actual

        # A-S Base Model
        q, g, k, tau = self.inv, self.cfg.gamma, self.cfg.k, self.cfg.tau
        r_t = mid - q * g * (sigma**2) * tau + drift
        half = (1.0 / g )*np.log(1.0 + g/k) + 0.5*g*(sigma**2)*tau
        bid, ask = r_t - half, r_t + half
        spr = ask - bid
        if self.cfg.spread_cap_frac is not None:
            # adjusting based on max and min spread configuration
            min_spread_abs = mid * self.cfg.min_spread_frac
            max_spread_abs = mid * self.cfg.spread_cap_frac
            tgt = max(min_spread_abs, min(max_spread_abs, spr))
            bid, ask = r_t - 0.5*tgt, r_t + 0.5*tgt

        bid = round_to_tick(bid, self.ex.tick_size)
        ask = round_to_tick(ask, self.ex.tick_size)

        if bid >= ask:
            # should not be happening, but will just log in case if happens
            print(f"bid {bid} more than ask {ask}!! warn, resetting ask to be above bid")
            ask = bid + self.ex.tick_size

        return bid, ask, (ask - bid), spr

    def max_qty_caps(self, bid_q: float, ask_q: float, sigma: float, mid: float) -> Tuple[float,float]:
        eq = self.equity(mid)
        # (1) notional_cap
        notional_cap = eq * self.cfg.max_order_notional_frac / (1.0 + self.cfg.vol_scale_k * max(sigma, 0.0))
        cap_risk_qty = max(0.0, notional_cap / max(mid, 1e-9))
        # (2) cash_buffer
        cash_buffer = eq * self.cfg.min_cash_buffer_frac
        cash_usable = max(0.0, self.cash - cash_buffer)
        buy_denom = bid_q * (1 + max(self.ex.maker_fee, 0.0))
        cap_buy_balance = max(0.0, cash_usable / max(buy_denom, 1e-9))
        cap_sell_balance = max(0.0, self.inv)  # 不能卖超过库存
        # (3) inv_notional limitation：|inv_notional ± q*mid| ≤ eq * max_inv_frac
        inv_notional = self.inv * mid
        lim_notional = eq * self.cfg.max_inv_frac
        cap_buy_inv  = max(0.0, (lim_notional - abs(inv_notional)) / max(mid, 1e-9)) if inv_notional >= 0 \
                       else max(0.0, (lim_notional - abs(inv_notional + 0.0)) / max(mid, 1e-9))
        cap_sell_inv = max(0.0, (lim_notional - abs(inv_notional)) / max(mid, 1e-9))
        # (4) inventory adjustment
        inv_frac = inv_notional / max(eq, 1e-9)
        buy_scale  = float(np.clip(1.0 - (inv_frac - self.cfg.target_inv_frac), 0.0, 1.0))
        sell_scale = float(np.clip(1.0 + (inv_frac - self.cfg.target_inv_frac), 0.0, 1.0))
        #
        max_buy_qty  = round_to_lot(min(cap_risk_qty, cap_buy_balance, cap_buy_inv)  * buy_scale,  self.ex.lot_size)
        max_sell_qty = round_to_lot(min(cap_risk_qty, cap_sell_balance, cap_sell_inv) * sell_scale, self.ex.lot_size)
        return max_buy_qty, max_sell_qty

    # —— hard liquidation：|inv_notional| >  hard_liq_frac*Equity ——
    def hard_liquidate_if_needed(self, mid: float) -> float:
        eq = self.equity(mid)
        inv_notional = self.inv * mid
        if abs(inv_notional) <= eq * self.cfg.hard_liq_frac:
            return 0.0
        # target_notional：target_inv_frac * Equity
        target_notional = self.cfg.target_inv_frac * eq
        delta_notional = inv_notional - target_notional
        qty_to_trade = abs(delta_notional) / max(mid, 1e-9)
        qty_to_trade = round_to_lot(qty_to_trade, self.ex.lot_size)
        if qty_to_trade <= 0: return 0.0
        if delta_notional > 0:
            # > target: sell
            proceeds = mid * qty_to_trade
            fee = proceeds * self.ex.taker_fee
            self.cash += (proceeds - fee)
            self.inv  -= qty_to_trade
            self.total_fees += -fee
            return -qty_to_trade
        else:
            # < target: buy
            cost = mid * qty_to_trade
            fee  = cost * self.ex.taker_fee
            self.cash -= (cost + fee)
            self.inv  += qty_to_trade
            self.total_fees += -fee
            return +qty_to_trade

    def step(self) -> Optional[Dict[str,Any]]:
        snap = self.feed.step()
        if snap is None: return None
        self._step_idx += 1

        # ============= Getting the raw timestamp  =============
        ts_raw = snap.get("ts")
        if ts_raw is None:
            ms = snap.get("exchTimeMs", snap.get("timeMs", None))
            if ms is not None:
                ts_raw = pd.to_datetime(int(ms), unit="ms", utc=True)
            else:
                base = getattr(self, "_t0", None)
                if base is None: base = datetime.now(timezone.utc); self._t0 = base
                ts_raw = base + timedelta(seconds=(self._step_idx-1)*self.step_seconds)

        ts = pd.to_datetime(ts_raw, utc=True)

        # ============= mid value from current row =============
        best_bid = float(snap["bid_px"]); best_ask = float(snap["ask_px"])
        best_bid_qty = float(snap.get("bid_qty", 0.0))
        best_ask_qty = float(snap.get("ask_qty", 0.0))
        mid = 0.5*(best_bid + best_ask)

        # σ
        if self.vol_model is not None:
            self.vol_model.update(ts, mid, best_bid, best_ask)
            sigma = float(self.vol_model.predict())
        else:
            print("Warning: missing vol_model")
            raw_sigma = snap.get("sigma", float("nan"))
            sigma = float(0.0 if (raw_sigma is None or np.isnan(raw_sigma)) else raw_sigma)

        self.px_bid, self.px_ask, dyn_spread, as_spread = self.get_as_quotes(mid, sigma)

        # clip thresholds (abs $)
        min_spread_abs = mid * self.cfg.min_spread_frac
        max_spread_abs = mid * self.cfg.spread_cap_frac
        # boolean flags → store as 0/1 floats for easy averaging later
        at_min_spread = float(dyn_spread <= min_spread_abs + 1e-9)
        at_max_spread = float(dyn_spread >= max_spread_abs - 1e-9)

        max_buy_qty, max_sell_qty = self.max_qty_caps(self.px_bid, self.px_ask, sigma, mid)

        trade_qty, trade_px, fee_this = 0.0, math.nan, 0.0

        # BUY
        if (self.px_bid >= best_bid) and best_bid_qty > 0 and max_buy_qty > 0:
            q = round_to_lot(min(max_buy_qty, best_bid_qty), self.ex.lot_size)
            if q > 0:
                cost = best_bid * q
                fee  = cost * self.ex.maker_fee
                if self.cash >= cost + max(fee, 0.0):
                    self.cash -= (cost + fee); self.inv += q
                    trade_qty, trade_px = +q, best_bid
                    fee_this += fee; self.total_fees += fee

        # SELL
        if (self.px_ask <= best_ask) and best_ask_qty > 0 and max_sell_qty > 0:
            q = round_to_lot(min(max_sell_qty, best_ask_qty), self.ex.lot_size)
            if q > 0 and self.inv >= q:
                proceeds = best_ask * q
                fee = proceeds * self.ex.maker_fee
                self.cash += (proceeds + fee); self.inv -= q
                if trade_qty == 0.0: trade_qty, trade_px = -q, best_ask
                fee_this += fee; self.total_fees += fee

        # hard liquidation if needed
        self.hard_liquidate_if_needed(mid)

        eq = self.equity(mid)
        pnl = eq - self.initial_cash

        if self.equity_prev is None: self.equity_prev = eq
        if trade_qty != 0.0:
            self.trade_steps.append(self._step_idx)
            self.trade_equity_delta.append(eq - self.equity_prev)
        self.equity_prev = eq

        self.history_rows.append({
            "ts": ts, "mid": mid,
            "bid": self.px_bid, "ask": self.px_ask,
            "dynamic_spread": dyn_spread, "as_spread": as_spread,
            "best_bid": best_bid, "best_ask": best_ask,
            "best_bid_qty": best_bid_qty, "best_ask_qty": best_ask_qty,
            "inv": self.inv, "cash": self.cash, "equity": eq, "pnl": pnl,
            "sigma": sigma, "trade_qty": trade_qty, "trade_price": trade_px, "fee_step": fee_this,
            "at_min_spread": at_min_spread, "at_max_spread": at_max_spread,
        })
        return snap

    def run(self) -> pd.DataFrame:
        self.feed.reset()
        self.cash, self.inv, self.total_fees = self.initial_cash, 0.0, 0.0
        self.history_rows.clear(); self.trade_steps.clear(); self.trade_equity_delta.clear()
        self.equity_prev, self._step_idx = None, 0

        total_steps = len(self.feed.rows)
        start_time = time.time()

        print(f"[MMSimulator] Starting simulation for {total_steps:,} ticks...")

        # tqdm automatically handles ETA, elapsed time, and rate
        with tqdm(total=total_steps, desc="Simulating", unit="ticks", ncols=100) as pbar:
            while not self.feed.done():
                self.step()
                pbar.update(1)  # advance the bar by one tick

        total_elapsed = time.time() - start_time
        print(f"[MMSimulator] Completed {self._step_idx:,} ticks in {total_elapsed:.1f}s "
              f"({self._step_idx / total_elapsed:.1f} ticks/sec)")

        df = pd.DataFrame(self.history_rows).sort_values("ts").reset_index(drop=True)
        return df

# ============ performence metrics & output ============
def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    eq = df["equity"].astype(float); ts = pd.to_datetime(df["ts"], utc=True)
    initial, final = float(eq.iloc[0]), float(eq.iloc[-1])
    total_seconds = (ts.iloc[-1] - ts.iloc[0]).total_seconds() or len(df)
    rets = eq.pct_change().fillna(0.0); mu = rets.mean(); sd = rets.std(ddof=1) or np.nan
    sec_per_year = 365*24*3600; step_sec = max(total_seconds/len(df), 1e-9)
    sharpe = (mu/(sd if sd>0 else np.nan))*math.sqrt(sec_per_year/step_sec) if sd==sd else float("nan")
    cumret = final/initial - 1.0
    cagr = (final/initial)**(sec_per_year/max(total_seconds,1e-9)) - 1.0 if final>0 and initial>0 else float("nan")
    run_max = eq.cummax(); dd = eq/run_max - 1.0; mdd = float(dd.min())
    end_i = int(dd.idxmin()); start_i = int((eq[:end_i]).idxmax()) if end_i>0 else 0
    dd_secs = (ts.iloc[end_i]-ts.iloc[start_i]).total_seconds() if end_i>start_i else 0.0

    # --- RMSE of volatility forecast ---
    eps = 1e-12
    realized_vol_window = 100  # Increased from 20 to better match GARCH window=1000
    realized_vol = rets.rolling(window=realized_vol_window).std(ddof=1)
    realized_var = np.maximum(realized_vol ** 2, eps)
    forecast_vol = df["sigma"].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    forecast_var = np.maximum(forecast_vol ** 2, eps)

    aligned = pd.concat([realized_var, forecast_var], axis=1, join="inner").dropna()
    realized_var, forecast_var = aligned.iloc[:, 0], aligned.iloc[:, 1]
    # Align lengths
    # min_len = min(len(realized_vol), len(forecast_vol))
    # realized_vol = realized_vol.iloc[-min_len:]
    # forecast_vol = forecast_vol.iloc[-min_len:]
    rmse_vol = np.sqrt(np.mean((np.sqrt(forecast_var) - np.sqrt(realized_var)) ** 2))
    ratio = np.clip(realized_var / forecast_var, eps, 1/eps)
    qlike_vol = np.mean(ratio - np.log(ratio) - 1)
    # --- Value at Risk (VaR) accuracy ---
    # 1-day VaR at 95% confidence: VaR = forecast_vol * norm.ppf(0.05)
    from scipy.stats import norm
    var_95 = -forecast_vol * norm.ppf(0.05) * eq.shift(1)  # VaR is negative, so use -ppf
    pnl = eq.diff().fillna(0.0)
    # Count breaches: actual loss > VaR
    var_breaches = ((-pnl) > var_95).sum()
    var_total = len(var_95)
    var_accuracy = 1.0 - (var_breaches / var_total) if var_total > 0 else float("nan")

    return {"CAGR": cagr, "Sharpe": sharpe, "Cumulative Return": cumret,
            "Max Drawdown": mdd, "Max DD Period (seconds)": dd_secs, "Final Equity": final,
            "RMSE Volatility Forecast": rmse_vol,
            "QLIKE Volatility Forecast": qlike_vol,
            "VaR Accuracy (95%)": var_accuracy}

def compute_win_ratio(trade_steps: List[int], trade_equity_delta: List[float]) -> float:
    if not trade_steps: return float("nan")
    return sum(1 for x in trade_equity_delta if x>0)/len(trade_equity_delta)

def save_csv(df: pd.DataFrame, path: str):
    cols = ["ts","mid","bid","ask","dynamic_spread", "as_spread",
            "inv","cash","equity","pnl",
            "sigma","trade_qty","trade_price","fee_step",
            "best_bid","best_ask","best_bid_qty","best_ask_qty",
            "at_min_spread","at_max_spread"]
    df[cols].to_csv(path, index=False)

def save_html_report(df: pd.DataFrame, metrics: Dict[str,Any], win_ratio: float, total_fees: float,
                     path: str, initial_cash: float = 100_000.0):
    ts = pd.to_datetime(df["ts"])
    price = df["mid"].astype(float)
    inv = df["inv"].astype(float)

    pnl = ((df["equity"] - float(df["equity"].iloc[0])) / initial_cash).astype(float)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Price","Inventory","PnL (%)"))

    fig.add_trace(go.Scatter(x=ts, y=price, name="Price", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=inv,   name="Inventory", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=pnl*100,
                             name="PnL (%)", mode="lines"), row=3, col=1)

    summary = (f"CAGR: {metrics['CAGR']:.2%} | Sharpe: {metrics['Sharpe']:.2f} | "
               f"Cumulative: {metrics['Cumulative Return']:.2%} | MDD: {metrics['Max Drawdown']:.2%} | "
               f"DD Period: {metrics['Max DD Period (seconds)']:.0f}s | Win: {win_ratio:.2%} | "
               f"Final Equity: {metrics['Final Equity']:.2f} | Total Fees: {total_fees:.2f}")

    fig.update_layout(template="ggplot2", height=900,
                      title="Market Making Backtest (A-S + Notional Risk Controls)",
                      margin=dict(l=60,r=30,t=60,b=80),
                      annotations=[dict(text=summary, xref="paper", yref="paper",
                                        x=0, y=0, showarrow=False, xanchor="left", yanchor="bottom",
                                        font=dict(size=12))])

    fig.write_html(path, include_plotlyjs="cdn")

def save_quotes_report(df: pd.DataFrame, path: str):
    ts = pd.to_datetime(df["ts"])

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=("Quotes Comparison: OB vs Algo"))

    # Order book best quotes
    fig.add_trace(go.Scatter(x=ts, y=df["best_bid"], name="OB Bid",
                             mode="lines", line=dict(color="blue", dash="dot")))
    fig.add_trace(go.Scatter(x=ts, y=df["best_ask"], name="OB Ask",
                             mode="lines", line=dict(color="red", dash="dot")))

    # Algo quotes
    fig.add_trace(go.Scatter(x=ts, y=df["bid"], name="Algo Bid",
                             mode="lines", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=ts, y=df["ask"], name="Algo Ask",
                             mode="lines", line=dict(color="red")))

    fig.update_layout(template="ggplot2", height=1200,
                      title="Algo vs Order Book Quotes",
                      margin=dict(l=60, r=30, t=60, b=80),
                      xaxis=dict(title="Time"),
                      yaxis=dict(title="Price"))

    fig.write_html(path, include_plotlyjs="cdn")


def save_volatility_comparison(all_results: Dict[str, pd.DataFrame], realized_vol_window: int = 100, path: str = None):
    first_model_name = list(all_results.keys())[0]
    df_first = all_results[first_model_name].copy()
    df_first = df_first.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(df_first["ts"])
    
    rets = df_first["mid"].astype(float).pct_change().fillna(0.0)
    realized_vol = rets.rolling(window=realized_vol_window).std(ddof=1)
    
    # Create figure
    fig = go.Figure()
    
    colors = {
        "EWMA": "#1f77b4",      
        "GARCH": "#ff7f0e",        
        "Realized": "#2ca02c",   
    }
    
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=realized_vol,
            name="Realized Volatility",
            mode="lines",
            line=dict(color=colors.get("Realized", "#2ca02c"), width=2, dash="dash"),
            opacity=1.0,
            hovertemplate="Realized Vol: %{y:.6f}<extra></extra>"
        )
    )
    
    for model_name, df in all_results.items():
        if model_name == "Realized":
            continue  
        
        df_model = df.copy().sort_values("ts").reset_index(drop=True)
        df_model_ts = pd.to_datetime(df_model["ts"])
        
        forecast_vol = df_model["sigma"].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        
        df_aligned = pd.DataFrame({"ts": ts})
        df_model_aligned = pd.merge_asof(
            df_aligned.sort_values("ts"),
            pd.DataFrame({"ts": df_model_ts, "sigma": forecast_vol}).sort_values("ts"),
            on="ts",
            direction="backward"
        )
        forecast_vol_aligned = df_model_aligned["sigma"].ffill().fillna(0.0)
        
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=forecast_vol_aligned,
                name=f"{model_name} Forecast",
                mode="lines",
                line=dict(color=colors.get(model_name, "#9467bd"), width=1.5),
                opacity=0.7,
                hovertemplate=f"{model_name}: %{{y:.6f}}<extra></extra>"
            )
        )
    
    fig.update_layout(
        template="plotly_white",
        height=600,
        title="Volatility Forecasting Comparison: EWMA, GARCH vs Realized Volatility",
        xaxis_title="Time",
        yaxis_title="Volatility",
        margin=dict(l=60, r=30, t=80, b=80),
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    if path:
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"Saved volatility comparison plot to {path}")
    else:
        return fig


def save_spread_analysis(df: pd.DataFrame, path:str, cfg: ASConfig):
    """
      Create a spread analysis report with 5 stacked plots:
        1) Market spread = best_ask - best_bid
        2) dynamic_spread
        3) as_spread
        4) min_spread = mid * min_spread_frac
        5) max_spread = mid * spread_cap_frac
      """
    ts = pd.to_datetime(df["ts"])

    best_bid = df["best_bid"]
    best_ask = df["best_ask"]
    dynamic_spread = df["dynamic_spread"]
    as_spread = df["as_spread"]

    # computed spreads
    market_spread = best_ask - best_bid
    min_spread = df["mid"] * cfg.min_spread_frac
    max_spread = df["mid"] * cfg.spread_cap_frac


    # figure with table row at the end
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],   # table in last row
        ],
        subplot_titles=(
            "Market Spread (best_ask - best_bid)",
            "Final Spread",
            "A-S Spread",
            f"Min Spread (mid × {cfg.min_spread_frac})",
            f"Max Spread (mid × {cfg.spread_cap_frac})",
            "Summary",
        ),
    )

    fig.add_trace(go.Scatter(x=ts, y=market_spread, name="Market Spread", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=dynamic_spread, name="Dynamic Spread", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=as_spread, name="A-S Spread", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=min_spread, name="Min Spread", mode="lines"), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts, y=max_spread, name="Max Spread", mode="lines"), row=5, col=1)

    # quick stats for footer
    series_map = {
        "Market Spread": market_spread,
        "Final Spread": dynamic_spread,
        "A-S Spread": as_spread,
        "Min Spread": min_spread,
        "Max Spread": max_spread,
        "Sigma": df['sigma']
    }

    names, means, medians, p95s, mins, maxs = [], [], [], [], [], []
    for name, s in series_map.items():
        names.append(name)
        means.append(f"{s.mean():.6f}")
        medians.append(f"{s.median():.6f}")
        p95s.append(f"{s.quantile(0.95):.6f}")
        mins.append(f"{s.min():.6f}")
        maxs.append(f"{s.max():.6f}")

    # add ratio of spread hitting min/max here
    min_clip = float(df["at_min_spread"].mean()) if "at_min_spread" in df else np.nan
    max_clip = float(df["at_max_spread"].mean()) if "at_max_spread" in df else np.nan

    # Append rows; only "Mean" is used for these
    names.extend(["% of Min Spread", "% of Max Spread"])
    mins.extend(["", ""])
    means.extend([f"{min_clip:.2%}", f"{max_clip:.2%}"])
    medians.extend(["", ""])
    p95s.extend(["", ""])
    maxs.extend(["", ""])


    fig.add_trace(
        go.Table(
            header=dict(values=["Series", "Min", "Mean", "Median", "p95", "Max"],
                        align="left"),
            cells=dict(values=[names, mins, means, medians, p95s, maxs],
                       align="left")
        ),
        row=6, col=1
    )

    fig.update_layout(
        template="ggplot2",
        height=1500,  # extra space for the table
        title="Spread Analysis",
        margin=dict(l=60, r=30, t=60, b=40),
    )

    fig.write_html(path, include_plotlyjs="cdn")


# ============ OKX order book CSV -> Feed ============
def okx_top1_csv_iter(path: str) -> Iterable[Dict[str,Any]]:
    df = pd.read_csv(path)
    def col(name):
        for c in df.columns:
            if c.lower() == name.lower(): return df[c]
        raise KeyError(f"missing column: {name}")

    ms = col("exchTimeMs") if "exchTimeMs".lower() in map(str.lower, df.columns) else col("timeMs")
    ts = pd.to_datetime(ms.astype(np.int64), unit="ms", utc=True)
    bid_px  = col("bid_1_px").astype(float); bid_qty = col("bid_1_qty").astype(float)
    ask_px  = col("ask_1_px").astype(float); ask_qty = col("ask_1_qty").astype(float)
    for i in range(len(df)):
        yield {"ts": ts.iloc[i],
               "bid_px": float(bid_px.iloc[i]), "ask_px": float(ask_px.iloc[i]),
               "bid_qty": float(bid_qty.iloc[i]), "ask_qty": float(ask_qty.iloc[i])}


class OKXTop1CSVFeed:
    def __init__(self, path: str):
        self.rows = list(okx_top1_csv_iter(path)); self._i = 0
    def reset(self): self._i = 0
    def step(self):
        if self._i >= len(self.rows): return None
        r = self.rows[self._i]; self._i += 1; return r
    def done(self): return self._i >= len(self.rows)
