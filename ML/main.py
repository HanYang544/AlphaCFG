#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alpha_backtest_shift20_ic_cumret.py
-----------------------------------
* Calculate IC / RankIC / ICIR / RankICIR
* Backtest with signals shifted 20 trading days backward
* Print the tail of cumulative returns (compounded)
"""

import numpy as np
import pandas as pd
import qlib
from qlib.backtest import backtest, executor as exec
from qlib.config import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.model.xgboost import XGBModel
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest, executor as bt_exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.model.pytorch_alstm import ALSTM
from qlib.contrib.model.pytorch_lstm import LSTM
from qlib.contrib.model.pytorch_tcn import TCN
from qlib.contrib.model.pytorch_transformer import TransformerModel

class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float

# ----------------- 0. Initialization -----------------
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_rolling", region="cn")
print("Qlib initialization completed.")
# ----------------- 1. Dataset -----------------
def prepare_dataset(train, valid, test):
    """
    Construct DatasetH:
    - features: 6 basic market fields * recent 60 days Ref
    - label: 20 days later return
    - All missing values are filled with 0 to avoid object dtype
    """
    base  = ["$high", "$low", "$open", "$close", "$volume", "$vwap"]
    feats = [f"Ref({f}, {lag})" for lag in range(60) for f in base]
    label = "Ref($close, -20) / $close - 1"

    handler = DataHandlerLP(
        instruments="csi300",
        start_time=min(train[0], valid[0], test[0]),
        end_time=max(train[1], valid[1], test[1]),

        # ---------- Data Loading ----------
        data_loader={
            "class": "QlibDataLoader",
            "kwargs": {"config": {"feature": feats, "label": [label]}},
        },

        # ---------- Processing Pipeline ----------
        # Keep consistent filling strategy for both training and inference
        shared_processors=[
            {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
            {"class": "Fillna", "kwargs": {"fields_group": "label",   "fill_value": 0}},
        ],
        learn_processors=[
            {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
            {"class": "Fillna", "kwargs": {"fields_group": "label",   "fill_value": 0}},
        ],
        infer_processors=[
            {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
            {"class": "Fillna", "kwargs": {"fields_group": "label",   "fill_value": 0}},
        ],
    )

    return DatasetH(
        handler=handler,
        segments={"train": train, "valid": valid, "test": test},
    )


# ----------------- 2. Risk -----------------
def calc_risk(series):
    try:
        return risk_analysis(series, mode="product")["risk"]
    except TypeError:
        return risk_analysis(series)["risk"]


# ----------------- 3. IC Tools -----------------
def ic_metrics(pred, label):
    df = pd.concat([
        pred.rename(columns={pred.columns[0]: "pred"}),
        label.rename(columns={label.columns[0]: "label"})
    ], axis=1).dropna()

    daily_ic  = df.groupby(level="datetime").apply(
        lambda g: g["pred"].corr(g["label"]))
    daily_ric = df.groupby(level="datetime").apply(
        lambda g: g["pred"].rank().corr(g["label"].rank()) if
        (g["pred"].nunique() > 1 and g["label"].nunique() > 1) else np.nan)

    ic_mean,  ic_std  = daily_ic.mean(),  daily_ic.std()
    ric_mean, ric_std = daily_ric.mean(), daily_ric.std()
    icir  = ic_mean  / ic_std  if ic_std  else np.nan
    ricir = ric_mean / ric_std if ric_std else np.nan
    return ic_mean, ric_mean, icir, ricir


def analyze_report(report: pd.DataFrame) -> BacktestResult:
    """
    Use qlib's built-in risk analysis function to extract final statistics 
    (such as annualized return, max drawdown, etc.) from the report.
    """
    # Excess return
    excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
    # Total return
    returns = risk_analysis(report["return"] - report["cost"])["risk"]

    return BacktestResult(
        sharpe=returns.loc["information_ratio"],
        annual_return=returns.loc["annualized_return"],
        max_drawdown=returns.loc["max_drawdown"],
        information_ratio=excess.loc["information_ratio"],
        annual_excess_return=excess.loc["annualized_return"],
        excess_max_drawdown=excess.loc["max_drawdown"]
    )

# ----------------- 4. Backtest (shift 20d) -----------------
def backtest_shift20(model_cls, params,
                     ds: DatasetH,
                     start: str, end: str,
                     seed: int = 0):
    """
    1) model.fit(DatasetH) internally automatically splits into train/valid
    2) Predict test segment → calculate IC / RankIC
    3) Shift signals 20 trading days backward then backtest
    4) Print the tail of cumulative returns
    """
    # ---------- Training ----------
    np.random.seed(seed)
    model = model_cls(**params, random_state=seed)

    # Adjust data format passing during training phase
    model.fit(ds)

    # ---------- Prediction ----------
    # Ensure the input to the model is DatasetH type, not DataFrame
    X_test = ds.prepare(segments="test", col_set="feature", data_key=DataHandlerLP.DK_I)
    # Still use DatasetH for prediction, not converting to DataFrame
    pred = pd.DataFrame(model.predict(ds), index=X_test.index, columns=["score"])

    # ---------- IC ----------
    y_test = ds.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_L)
    ic_mean, ric_mean, icir, ricir = ic_metrics(pred, y_test)

    print("\n=== IC / RankIC ===")
    print(f"IC={ic_mean:.4f}, RankIC={ric_mean:.4f}, "
          f"ICIR={icir:.4f}, RankICIR={ricir:.4f}")

    # ---------- Signal shifted 20d ----------
    signal = pred.shift(20).dropna()

    strategy = TopkDropoutStrategy(
        signal=signal["score"],
        topk=60, n_drop=5
    )
    executor = exec.SimulatorExecutor(
                time_per_step="day", 
                generate_portfolio_metrics=True
            )

    backtest_impl = backtest(
        strategy=strategy,
        executor=executor,
        start_time=start,
        end_time=end,
        account=100_000_000,
        benchmark="SH000300",
        exchange_kwargs=dict(
            limit_threshold=0.095,
            deal_price="close",
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5,
        )
    )[0]

    report, _ = backtest_impl["1day"]
    # Plot portfolio analysis chart
    graph = report_graph(report, show_notebook=False)[0]
    result = analyze_report(report)
    print("=== Report Head ===")
    print(report.head())
    print("=== Backtest Metrics ===")
    print(result)

    return report


def compute_cumulative_returns(report: pd.DataFrame) -> pd.Series:
    """
    Calculate cumulative returns:
    - daily_net_ret: daily net return (return - cost)
    - cum_net: cumulative return (compounded)
    """
    # 1) Data check and preprocessing
    if not set(["return", "cost"]).issubset(report.columns):
        raise ValueError("report DataFrame must contain 'return' and 'cost' columns.")

    # Sort and reset index to ensure iloc order
    report = report.sort_index().reset_index(drop=True)

    # 2) Calculate daily net return and cumulative net value
    daily_net_ret = report["return"] - report["cost"]
    cum_net = (1 + daily_net_ret).cumprod() - 1

    return cum_net



# ----------------- 5. Main Process -----------------
if __name__ == "__main__":
    train = ("2010-01-01", "2017-12-31")
    valid = ("2018-01-01", "2019-12-31")
    test  = ("2021-01-01", "2024-12-31")

    seed=1
    ds = prepare_dataset(train, valid, test)
    print(f"Type of ds in backtest: {type(ds)}")
    start, end = "2021-01-01", "2024-12-31"

    print("\n========== XGBModel ==========")
    r1 = backtest_shift20(XGBModel, {}, ds, start, end, seed=seed)
    print(compute_cumulative_returns(r1))

    print("\n========== LGBModel ==========")
    r2 = backtest_shift20(LGBModel, {}, ds, start, end, seed=seed)
    print(compute_cumulative_returns(r2))

    print("\n========== ALSTM ==========")
    r3 = backtest_shift20(ALSTM, {}, ds, start, end, seed=seed)
    print(compute_cumulative_returns(r3))

    print("\n========== TCN ==========")
    r4 = backtest_shift20(TCN, {}, ds, start, end, seed=seed)
    print(compute_cumulative_returns(r4))

    print("\n========== LSTM ==========")
    r5 = backtest_shift20(LSTM, {}, ds, start, end, seed=seed)
    print(compute_cumulative_returns(r5))

    
    print("\n========== TransformerModel ==========")
    r6 = backtest_shift20(TransformerModel, {}, ds, start, end, seed=seed)
    print(compute_cumulative_returns(r6))
