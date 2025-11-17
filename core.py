# core.py
import datetime as dt
from datetime import time
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# -------------------- 공통 상수 (DD 구간) -------------------- #

DD_BINS = [-1000, -40, -30, -20, -10, -5, 0, 5, 10, 20, 1000]
DD_LABELS = [
    "< -40%",
    "-40 ~ -30%",
    "-30 ~ -20%",
    "-20 ~ -10%",
    "-10 ~ -5%",
    "-5 ~ 0%",
    "0 ~ +5%",
    "+5 ~ +10%",
    "+10 ~ +20%",
    ">= +20%",
]


# -------------------- 유틸 함수 -------------------- #

def ensure_1d_series(x, name: str = "series") -> pd.Series:
    """
    prices / close가 DataFrame, ndarray, Series 등 어떤 형태로 들어와도
    항상 1D Series로 만들어서 반환.
    """
    if isinstance(x, pd.Series):
        return x.dropna()

    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0]
        return s.dropna()

    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    return pd.Series(arr).dropna()


def calculate_cagr(start_value: float, end_value: float, start_date: dt.datetime, end_date: dt.datetime) -> float:
    if start_value <= 0:
        return 0.0
    if end_date <= start_date:
        return 0.0
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1.0


# -------------------- 드로우다운 / RSI -------------------- #

def compute_drawdown(prices) -> pd.Series:
    prices = ensure_1d_series(prices, "prices")
    if prices.empty:
        return prices * np.nan

    cumulative_max = prices.cummax()
    dd = (prices / cumulative_max - 1.0) * 100.0
    return dd


def compute_rsi(series, period: int = 14) -> pd.Series:
    prices = ensure_1d_series(series, "prices_for_rsi")
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# -------------------- 특수 리밸 백테스트 -------------------- #

def run_grid_rebal_backtest(
    prices,
    initial_capital: float,
    init_weight: float,
    up_pct: float,
    down_pct: float,
    buy_frac: float,
    sell_frac: float,
    trade_base: str = "total",     # "total" or "equity"
    rebalance_freq: str = "event", # "event", "1D", "1W", "1M", "4M"
):
    """
    단일 자산 특수 리밸 전략 백테스트.
    prices: 1D Series (index: DatetimeIndex, values: price)
    """

    prices = ensure_1d_series(prices, "prices")
    if not isinstance(prices.index, pd.DatetimeIndex):
        # 날짜 인덱스 보장 (없으면 일자 단위 가짜 인덱스)
        prices.index = pd.to_datetime(prices.index)

    prices = prices.sort_index()
    if len(prices) < 2:
        raise ValueError("가격 데이터가 너무 적습니다.")

    start_date = prices.index[0]
    end_date = prices.index[-1]

    # 초기 포지션
    p0 = prices.iloc[0]
    equity_init = initial_capital * (init_weight / 100.0)
    cash = initial_capital - equity_init
    shares = equity_init / p0 if p0 > 0 else 0.0
    ref_price = p0

    # Buy&Hold 기준
    bh_shares = equity_init / p0 if p0 > 0 else 0.0
    bh_cash = initial_capital - equity_init

    trade_log = []
    records = []

    last_trade_idx = 0

    def can_trade(current_idx: int) -> bool:
        nonlocal last_trade_idx
        if rebalance_freq == "event":
            return True
        # 인덱스 차이(캔들 개수) 기준으로 간단히 구현
        diff = current_idx - last_trade_idx
        if rebalance_freq == "1D":
            return diff >= 1
        elif rebalance_freq == "1W":
            return diff >= 5
        elif rebalance_freq == "1M":
            return diff >= 21
        elif rebalance_freq == "4M":
            return diff >= 84
        else:
            return True

    for i, (date, price) in enumerate(prices.items()):
        equity_value = shares * price
        total = cash + equity_value

        # Buy&Hold
        bh_total = bh_cash + bh_shares * price

        action = "HOLD"
        shares_change = 0.0
        trade_value = 0.0

        if i > 0 and total > 0 and ref_price > 0:
            diff_pct = (price / ref_price - 1.0) * 100.0

            if can_trade(i):
                # trade_base 선택
                if trade_base == "equity":
                    base_val = equity_value
                else:
                    base_val = total

                if diff_pct <= -down_pct:
                    # BUY
                    target_val = base_val * (buy_frac / 100.0)
                    trade_value = min(target_val, cash)
                    if trade_value > 0 and price > 0:
                        shares_delta = trade_value / price
                        shares += shares_delta
                        cash -= trade_value
                        shares_change = shares_delta
                        action = "BUY"
                        ref_price = price
                        last_trade_idx = i

                elif diff_pct >= up_pct:
                    # SELL
                    target_val = base_val * (sell_frac / 100.0)
                    max_sell_val = shares * price
                    trade_value = min(target_val, max_sell_val)
                    if trade_value > 0 and price > 0:
                        shares_delta = trade_value / price
                        shares -= shares_delta
                        cash += trade_value
                        shares_change = -shares_delta
                        action = "SELL"
                        ref_price = price
                        last_trade_idx = i

        # 레코드 저장
        equity_value = shares * price
        total = cash + equity_value
        bh_total = bh_cash + bh_shares * price

        records.append(
            {
                "Date": date,
                "Price": price,
                "Cash": cash,
                "Shares": shares,
                "Equity_Value": equity_value,
                "Total_Value": total,
                "BuyHold_Value": bh_total,
                "Ref_Price": ref_price,
                "Last_Action": action,
            }
        )

        if action in ("BUY", "SELL"):
            trade_log.append(
                {
                    "Date": date,
                    "Action": action,
                    "Price": price,
                    "Trade_Value": trade_value,
                    "Shares_Change": shares_change,
                    "Cash_After": cash,
                    "Total_After": total,
                }
            )

    result_df = pd.DataFrame(records).set_index("Date")
    result_df.rename(
        columns={
            "Total_Value": "Strategy_Value",
        },
        inplace=True,
    )

    final_total = result_df["Strategy_Value"].iloc[-1]
    bh_final = result_df["BuyHold_Value"].iloc[-1]

    final_return_pct = (final_total / initial_capital - 1.0) * 100.0
    bh_return_pct = (bh_final / initial_capital - 1.0) * 100.0

    cagr_strat = calculate_cagr(initial_capital, final_total, start_date, end_date) * 100.0
    cagr_bh = calculate_cagr(initial_capital, bh_final, start_date, end_date) * 100.0

    summary = {
        "initial_capital": initial_capital,
        "final_total": final_total,
        "final_return_pct": final_return_pct,
        "buyhold_final": bh_final,
        "buyhold_return_pct": bh_return_pct,
        "cagr_strat_pct": cagr_strat,
        "cagr_bh_pct": cagr_bh,
        "num_trades": len(trade_log),
    }

    final_state = {
        "final_total": final_total,
        "final_cash": cash,
        "final_shares": shares,
        "last_price": prices.iloc[-1],
        "ref_price": ref_price,
    }

    return result_df, trade_log, final_state, summary


# -------------------- DD 기반 1D 통계 -------------------- #

def compute_signal_stats(prices, horizon: int = 20) -> Optional[pd.DataFrame]:
    prices = ensure_1d_series(prices, "prices_for_signal_stats")
    if len(prices) < horizon + 5:
        return None

    dd = compute_drawdown(prices)

    fwd_rets = []
    buckets = []

    for i in range(len(prices) - horizon):
        dd_now = dd.iloc[i]
        if pd.isna(dd_now):
            continue
        p_now = prices.iloc[i]
        p_fwd = prices.iloc[i + horizon]
        if p_now <= 0:
            continue

        ret = (p_fwd / p_now - 1.0) * 100.0
        bucket = pd.cut(
            [dd_now],
            bins=DD_BINS,
            labels=DD_LABELS,
            right=False,
        )[0]
        if pd.isna(bucket):
            continue

        fwd_rets.append(ret)
        buckets.append(str(bucket))

    if not fwd_rets:
        return None

    df = pd.DataFrame({"DD_Bucket": buckets, "FwdRet_Pct": fwd_rets})
    grouped = df.groupby("DD_Bucket")["FwdRet_Pct"]

    stats_df = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "median": grouped.median(),
            "count": grouped.count(),
            "positive_ratio": (grouped.apply(lambda x: (x > 0).mean()) * 100.0),
        }
    )
    return stats_df


# -------------------- DD × RSI 2D 통계 -------------------- #

def compute_dd_rsi_2d_stats(
    prices,
    rsi_period: int = 14,
    horizon: int = 20,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    prices = ensure_1d_series(prices, "prices_for_dd_rsi_2d")
    if len(prices) < max(rsi_period, horizon) + 10:
        return None, None

    dd = compute_drawdown(prices)
    rsi = compute_rsi(prices, rsi_period)

    df = pd.DataFrame(
        {
            "Price": prices,
            "DD": dd,
            "RSI": rsi,
        }
    ).dropna()

    if len(df) < horizon + 5:
        return None, None

    # 버킷 정의
    def rsi_bucket_func(v):
        if v < 30:
            return "RSI<30"
        elif v < 50:
            return "30~50"
        elif v < 70:
            return "50~70"
        else:
            return ">=70"

    dd_buckets = pd.cut(df["DD"], bins=DD_BINS, labels=DD_LABELS, right=False)
    rsi_buckets = df["RSI"].apply(rsi_bucket_func)

    df["DD_Bucket"] = dd_buckets.astype(str)
    df["RSI_Bucket"] = rsi_buckets

    fwd_returns = []
    dd_list = []
    rsi_list = []

    df = df.sort_index()
    idx_list = df.index.to_list()
    for i in range(len(df) - horizon):
        idx_now = idx_list[i]
        idx_fwd = idx_list[i + horizon]

        p_now = df.at[idx_now, "Price"]
        p_fwd = df.at[idx_fwd, "Price"]
        if p_now <= 0:
            continue

        dd_b = df.at[idx_now, "DD_Bucket"]
        rsi_b = df.at[idx_now, "RSI_Bucket"]

        if dd_b not in DD_LABELS:
            continue

        ret = (p_fwd / p_now - 1.0) * 100.0
        fwd_returns.append(ret)
        dd_list.append(dd_b)
        rsi_list.append(rsi_b)

    if not fwd_returns:
        return None, None

    raw_df = pd.DataFrame(
        {
            "DD_Bucket": dd_list,
            "RSI_Bucket": rsi_list,
            "FwdRet_Pct": fwd_returns,
        }
    )

    pivot_mean = raw_df.pivot_table(
        index="DD_Bucket",
        columns="RSI_Bucket",
        values="FwdRet_Pct",
        aggfunc="mean",
    )

    return pivot_mean, raw_df


# -------------------- VIX 분석 -------------------- #

def analyze_vix(start: str, end: str):
    """
    VIX (^VIX) 지수 다운로드 후,
    - 마지막 값
    - 과거 대비 분위수 (%)
    - 전체 시계열
    """
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    except Exception:
        return None, None, None

    if vix is None or vix.empty or "Close" not in vix.columns:
        return None, None, None

    series = ensure_1d_series(vix["Close"], "vix_close")
    if series.empty:
        return None, None, None

    v_now = float(series.iloc[-1])
    # 분위수: 지금 값 이하가 전체 중 몇 %인지
    pct = (series <= v_now).mean() * 100.0

    return v_now, pct, series


# -------------------- 파라미터 그리드 탐색 -------------------- #

def optimize_param_grid(
    prices,
    initial_capital: float,
    init_weight: float,
    rebalance_freq: str,
    trade_base: str,
):
    """
    down_pct, up_pct, buy_frac, sell_frac에 대해 간단한 그리드 탐색.
    너무 무겁지 않게 범위 제한.
    """
    prices = ensure_1d_series(prices, "prices_for_opt")

    down_grid = [5, 10, 15, 20]
    up_grid = [5, 10, 15, 20]
    buy_grid = [5, 10, 15]
    sell_grid = [5, 10, 15]

    best_combo = None
    best_final = -np.inf
    total_tests = 0
    success_tests = 0

    for d in down_grid:
        for u in up_grid:
            for bf in buy_grid:
                for sf in sell_grid:
               

                    total_tests += 1
                    try:
                        _, _, _, summary = run_grid_rebal_backtest(
                            prices,
                            initial_capital,
                            init_weight,
                            u,  # up_pct
                            d,  # down_pct
                            bf,
                            sf,
                            trade_base=trade_base,
                            rebalance_freq=rebalance_freq,
                        )
                        success_tests += 1
                    except Exception:
                        continue

                    final_total = summary["final_total"]
                    if final_total > best_final:
                        best_final = final_total
                        best_combo = (d, u, bf, sf, summary)

    return best_combo, total_tests, success_tests


# -------------------- 프리장 vs 본장 통계 분석 -------------------- #

def analyze_premarket_vs_regular(
    ticker: str,
    start: str,
    end: str,
    interval: str = "30m",
    strong_move_thresh: float = 1.0,
):
    """
    프리장(04:00~09:30) / 본장(09:30~16:00) 변동률을 일 단위로 뽑아서
    - PreRet: 프리장 %수익률
    - RegRet: 본장 %수익률
    를 계산하고,
    각종 조건부 확률 + 상관계수 + 단순 회귀 기울기 등을 반환.

    반환:
      daily_df (index=Date, cols: PreRet, RegRet)
      stats (dict)
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            prepost=True,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise ValueError(f"yfinance 인트라데이 다운로드 실패: {e}")

    if df is None or df.empty or "Open" not in df.columns or "Close" not in df.columns:
        raise ValueError("인트라데이 데이터가 비어 있거나 컬럼이 부족합니다.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("인트라데이 데이터 인덱스가 DatetimeIndex가 아닙니다.")

    # 타임존을 미국 동부시간으로 정규화 (NYSE 기준)
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = idx.tz_convert("America/New_York")

    df = df.sort_index()

    df["date"] = df.index.date
    # 그룹핑용
    daily_rows = []

    for d, grp in df.groupby("date"):
        # pre-market: 04:00 ~ 09:30
        pre_mask = (grp.index.time >= time(4, 0)) & (grp.index.time < time(9, 30))
        reg_mask = (grp.index.time >= time(9, 30)) & (grp.index.time <= time(16, 0))

        pre = grp.loc[pre_mask]
        reg = grp.loc[reg_mask]

        if pre.empty or reg.empty:
            continue

        pre_open = float(pre["Open"].iloc[0])
        pre_close = float(pre["Close"].iloc[-1])
        reg_open = float(reg["Open"].iloc[0])
        reg_close = float(reg["Close"].iloc[-1])

        if pre_open <= 0 or reg_open <= 0:
            continue

        pre_ret = (pre_close / pre_open - 1.0) * 100.0
        reg_ret = (reg_close / reg_open - 1.0) * 100.0

        daily_rows.append(
            {
                "Date": pd.to_datetime(d),
                "PreRet": pre_ret,
                "RegRet": reg_ret,
            }
        )

    if not daily_rows:
        raise ValueError("프리장/본장 데이터가 충분하지 않습니다. 기간을 줄이거나 다른 티커를 시도해 보세요.")

    daily_df = pd.DataFrame(daily_rows)
    daily_df = daily_df.sort_values("Date")
    daily_df.set_index("Date", inplace=True)

    if len(daily_df) < 10:
        raise ValueError(f"유효한 일 수가 {len(daily_df)}일밖에 없습니다. (최소 10일 권장)")

    pre = daily_df["PreRet"]
    reg = daily_df["RegRet"]

    # 상관계수 / 회귀 기울기
    if pre.std() > 0 and reg.std() > 0:
        corr = float(pre.corr(reg))
        var_pre = float(pre.var())
        if var_pre > 0:
            cov = float(np.cov(pre, reg, ddof=1)[0, 1])
            slope = cov / var_pre  # Reg = a + slope * Pre
        else:
            slope = None
    else:
        corr = None
        slope = None

    # 조건부 확률들
    reg_up = reg > 0
    reg_down = reg < 0
    pre_up = pre > 0
    pre_down = pre < 0

    def safe_ratio(num, den):
        num = float(num)
        den = float(den)
        if den <= 0:
            return None
        return num / den * 100.0

    p_up = safe_ratio(reg_up.sum(), len(reg))
    p_down = safe_ratio(reg_down.sum(), len(reg))

    p_up_given_pre_up = safe_ratio((reg_up & pre_up).sum(), pre_up.sum())
    p_up_given_pre_down = safe_ratio((reg_up & pre_down).sum(), pre_down.sum())

    same_sign = (pre * reg) > 0
    p_same_sign = safe_ratio(same_sign.sum(), len(reg))

    # 절댓값 기준 "강한 프리장" 조건
    thr = float(abs(strong_move_thresh))
    strong_up = pre >= thr
    strong_down = pre <= -thr

    p_up_given_strong_up = safe_ratio((reg_up & strong_up).sum(), strong_up.sum())
    p_up_given_strong_down = safe_ratio((reg_up & strong_down).sum(), strong_down.sum())

    stats = {
        "n_days": int(len(daily_df)),
        "start_date": str(daily_df.index.min().date()),
        "end_date": str(daily_df.index.max().date()),
        "corr_pre_reg": corr,
        "reg_slope_per_1pct_pre": slope,  # 프리장 1% 변화당 본장 평균 변화(%)
        "p_reg_up": p_up,
        "p_reg_down": p_down,
        "p_reg_up_given_pre_up": p_up_given_pre_up,
        "p_reg_up_given_pre_down": p_up_given_pre_down,
        "p_same_sign": p_same_sign,
        "strong_move_thresh": thr,
        "p_reg_up_given_strong_pre_up": p_up_given_strong_up,
        "p_reg_up_given_strong_pre_down": p_up_given_strong_down,
    }

    return daily_df, stats

# -------------------- HYBRID SIGNAL ENGINE: '그래서 오늘 살까요?' -------------------- #

def _build_state_features(prices: pd.Series, spy: pd.Series, vix_series: pd.Series) -> pd.DataFrame:
    """공통 인덱스 구간으로 정렬한 뒤 상태 벡터를 계산한다."""
    # 공통 인덱스 교집합
    idx = prices.index.intersection(spy.index).intersection(vix_series.index)
    df = pd.DataFrame(
        {
            "Price": prices.loc[idx],
            "SPY": spy.loc[idx],
            "VIX": vix_series.loc[idx],
        }
    ).dropna()

    if len(df) < 80:
        return pd.DataFrame()

    # 이동평균
    df["MA20"] = df["Price"].rolling(20).mean()
    df["MA60"] = df["Price"].rolling(60).mean()

    df["z20"] = (df["Price"] - df["MA20"]) / df["MA20"] * 100.0
    df["z60"] = (df["Price"] - df["MA60"]) / df["MA60"] * 100.0

    # RSI, DD
    df["RSI"] = compute_rsi(df["Price"], 14)
    df["DD"] = compute_drawdown(df["Price"])

    # SPY 상태
    df["SPY_MA20"] = df["SPY"].rolling(20).mean()
    df["SPY_RSI"] = compute_rsi(df["SPY"], 14)
    df["SPY_DD"] = compute_drawdown(df["SPY"])
    df["SPY_z20"] = (df["SPY"] - df["SPY_MA20"]) / df["SPY_MA20"] * 100.0

    # VIX 분위수 (시계열 내 순위 기준)
    df["VIX_pct"] = df["VIX"].rank(pct=True) * 100.0

    # 단기 수익률
    df["ret_3d"] = df["Price"].pct_change(3) * 100.0
    df["ret_10d"] = df["Price"].pct_change(10) * 100.0

    return df


def _compute_future_returns(prices: pd.Series) -> pd.DataFrame:
    """미래 1/5/20 거래일 수익률 (단위: %) 계산"""
    s = prices.sort_index()
    df = pd.DataFrame({"Price": s})
    df["ret_1d"] = (df["Price"].shift(-1) / df["Price"] - 1.0) * 100.0
    df["ret_5d"] = (df["Price"].shift(-5) / df["Price"] - 1.0) * 100.0
    df["ret_20d"] = (df["Price"].shift(-20) / df["Price"] - 1.0) * 100.0
    return df


def _find_similar_states(
    state_df: pd.DataFrame,
    today_row: pd.Series,
    min_samples: int = 80,
) -> pd.DataFrame:
    """상태 DF와 오늘 상태를 기준으로 유사 상태들을 찾는다."""
    base = state_df.copy()

    # 오늘 자신은 제외
    if today_row.name in base.index:
        base = base.drop(index=today_row.name)

    # 1차 조건 (타이트)
    cond = (
        base["DD"].between(today_row["DD"] - 5, today_row["DD"] + 5)
        & base["RSI"].between(today_row["RSI"] - 10, today_row["RSI"] + 10)
        & base["z20"].between(today_row["z20"] - 5, today_row["z20"] + 5)
        & base["z60"].between(today_row["z60"] - 5, today_row["z60"] + 5)
        & base["ret_3d"].between(today_row["ret_3d"] - 5, today_row["ret_3d"] + 5)
        & base["SPY_RSI"].between(today_row["SPY_RSI"] - 10, today_row["SPY_RSI"] + 10)
        & base["VIX_pct"].between(today_row["VIX_pct"] - 15, today_row["VIX_pct"] + 15)
    )

    candidates = base[cond].dropna()

    # 표본이 너무 적으면 조건 완화
    if len(candidates) < min_samples:
        cond = (
            base["DD"].between(today_row["DD"] - 8, today_row["DD"] + 8)
            & base["RSI"].between(today_row["RSI"] - 15, today_row["RSI"] + 15)
            & base["z20"].between(today_row["z20"] - 8, today_row["z20"] + 8)
            & base["VIX_pct"].between(today_row["VIX_pct"] - 20, today_row["VIX_pct"] + 20)
        )
        candidates = base[cond].dropna()

    return candidates


def _compute_probability_from_df(similar_df: pd.DataFrame) -> dict:
    """similar_df 안에 ret_1d, ret_5d, ret_20d 컬럼이 포함되어 있다고 가정."""
    if similar_df.empty:
        return {
            "p1": None,
            "p5": None,
            "p20": None,
            "avg1": None,
            "avg5": None,
            "avg20": None,
            "count": 0,
        }

    sub = similar_df.dropna(subset=["ret_1d", "ret_5d", "ret_20d"])

    if sub.empty:
        return {
            "p1": None,
            "p5": None,
            "p20": None,
            "avg1": None,
            "avg5": None,
            "avg20": None,
            "count": 0,
        }

    def pct_pos(col: pd.Series) -> float:
        return float((col > 0).mean() * 100.0)

    result = {
        "p1": pct_pos(sub["ret_1d"]),
        "p5": pct_pos(sub["ret_5d"]),
        "p20": pct_pos(sub["ret_20d"]),
        "avg1": float(sub["ret_1d"].mean()),
        "avg5": float(sub["ret_5d"].mean()),
        "avg20": float(sub["ret_20d"].mean()),
        "count": int(len(sub)),
    }
    return result


def analyze_hybrid_signal(
    prices: pd.Series,
    spy_prices: pd.Series,
    vix_series: pd.Series,
    min_samples: int = 80,
):
    """네 번째 탭 '그래서 오늘 살까요?'용 하이브리드 분석 엔진."""
    prices = ensure_1d_series(prices, "prices_for_hybrid")
    spy_prices = ensure_1d_series(spy_prices, "spy_for_hybrid")
    vix_series = ensure_1d_series(vix_series, "vix_for_hybrid")

    state_df = _build_state_features(prices, spy_prices, vix_series)
    if state_df.empty or len(state_df) < 80:
        return None, None

    fut_df = _compute_future_returns(prices)

    # 상태 DF와 미래 수익률 DF를 합침 (공통 인덱스 & NaN 제거)
    merged = state_df.join(
        fut_df[["ret_1d", "ret_5d", "ret_20d"]], how="inner"
    ).dropna(subset=["ret_1d", "ret_5d", "ret_20d"])

    if merged.empty or len(merged) < 80:
        return None, None

    # 오늘 상태 (가장 마지막 일자) - state_df 기준
    today_state = state_df.iloc[-1]

    # 유사 상태 탐색 (미래 수익률 있는 구간 내에서만)
    similar_states = _find_similar_states(merged, today_state, min_samples=min_samples)
    if similar_states.empty:
        return None, today_state

    probs = _compute_probability_from_df(similar_states)

    return probs, today_state

