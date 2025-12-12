"""
Generalized performance driver analysis for an equity ticker.

1. Download daily adjusted close prices with yfinance.
2. Convert to log returns and fit an OLS regression to explain the target.
3. Attribute R^2 across driver groups via Shapley (LMG-style) shares.
4. Create a dashboard figure (bars + factor correlations) and a rebased price plot.

Edit the CONFIG block to map factors to your preferred tickers. Run:
    python performance_drivers.py --months 12 --target IBE.MC
Accepted lookback windows: 6, 12, or 24 months.

Missing tickers in the configuration are skipped automatically with warnings,
and the rebased chart highlights the most correlated available drivers.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from matplotlib import pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration section (edit these tickers to analyze a different company)
# --------------------------------------------------------------------------- #

CONFIG = {
    "default_months": 12,
    "rebased_top_n": 3,
    "groups": {
        "Market": [
            {"name": "global_market", "ticker": "URTH"},
            {"name": "regional_market", "ticker": "^STOXX50E"},
            {"name": "local_market", "ticker": "^IBEX"},
        ],
        "Sector": [
            {"name": "utilities_europe", "ticker": "EXH9.DE"},
            {"name": "utilities_us", "ticker": "XLU"},
        ],
        "Macro": [
            {"name": "eur_gbp", "ticker": "EURGBP=X"},
            {"name": "eur_usd", "ticker": "EURUSD=X"},
            {"name": "clean_energy", "ticker": "INRG.L"},
            {"name": "rates_2y_us", "ticker": "^IRX"},
            {"name": "rates_10y_us", "ticker": "^TNX"},
        ],
        "Style": [
            {"name": "momentum", "ticker": "MTUM"},
            {"name": "low_volatility", "ticker": "USMV"},
            {"name": "quality", "ticker": "QUAL"},
            {"name": "size", "ticker": "SIZE"},
        ],
    },
}

CORRELATION_WINDOWS = [6, 12]


# --------------------------------------------------------------------------- #
# Helper data structures and utilities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriverConfig:
    target: str
    groups: Dict[str, List[Dict[str, str]]]
    rebased_top_n: int

    def iter_predictors(self) -> Iterable[Tuple[str, str, str]]:
        for group, entries in self.groups.items():
            for entry in entries:
                yield entry["name"], entry.get("ticker"), group

    @property
    def predictor_to_ticker(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for name, ticker, _ in self.iter_predictors():
            if ticker:
                mapping[name] = ticker
        return mapping

    @property
    def ticker_to_predictor(self) -> Dict[str, str]:
        return {ticker: predictor for predictor, ticker in self.predictor_to_ticker.items()}

    @property
    def factor_tickers(self) -> List[str]:
        return sorted({ticker for _, ticker, _ in self.iter_predictors() if ticker})

    @property
    def group_order(self) -> List[str]:
        return list(self.groups.keys())

    def active_predictors(
        self, available_columns: Iterable[str]
    ) -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
        mapping: Dict[str, str] = {}
        group_lookup: Dict[str, str] = {}
        missing: List[Tuple[str, str]] = []
        available = set(available_columns)
        for name, ticker, group in self.iter_predictors():
            if not ticker:
                missing.append((name, "<unspecified>"))
                continue
            if ticker in available:
                mapping[name] = ticker
                group_lookup[name] = group
            else:
                missing.append((name, ticker))
        return mapping, group_lookup, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain equity performance drivers via multi-factor regression.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="IBE.MC",
        help="Ticker for the dependent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--months",
        type=int,
        choices=[6, 12, 24],
        default=CONFIG["default_months"],
        help="Lookback window in months (default: %(default)s)",
    )
    return parser.parse_args()


def determine_date_range(months: int, corr_windows: Iterable[int]) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(datetime.now().date())
    lookback_months = max([months, *corr_windows])
    history_start = end - pd.DateOffset(months=lookback_months)
    analysis_start = end - pd.DateOffset(months=months)
    return history_start, analysis_start, end


def download_prices(tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    tickers = sorted(set(tickers))
    raw = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data. Check ticker spelling or date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        adj_close = raw["Adj Close"]
    else:
        adj_close = raw["Adj Close"].to_frame(name=tickers[0])

    adj_close = adj_close.sort_index()
    adj_close.index = pd.to_datetime(adj_close.index)
    return adj_close


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_rets = np.log(prices / prices.shift(1))
    log_rets.index.name = "date"
    return log_rets.dropna(how="all")


def build_model_frame(
    log_returns: pd.DataFrame,
    config: DriverConfig,
    active_mapping: Dict[str, str],
) -> pd.DataFrame:
    if not active_mapping:
        raise ValueError("No predictors with available data.")
    columns_needed = [config.target, *active_mapping.values()]
    frame = log_returns[columns_needed].dropna().reset_index()
    rename_map = {ticker: predictor for predictor, ticker in active_mapping.items()}
    model_df = frame.rename(columns=rename_map)
    model_df = model_df.rename(columns={config.target: "r"})
    return model_df


def fit_ols(model_df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    y = model_df["r"]
    X = model_df.drop(columns=["date", "r"])
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop").fit()
    return model


def r_squared(model: sm.regression.linear_model.RegressionResultsWrapper) -> float:
    return float(model.rsquared)


def compute_shapley_r2(model_df: pd.DataFrame) -> Tuple[pd.Series, float]:
    y = model_df["r"]
    X = model_df.drop(columns=["date", "r"])
    columns = list(X.columns)
    n = len(columns)
    cache: Dict[Tuple[str, ...], float] = {}

    def subset_r2(cols: Tuple[str, ...]) -> float:
        if cols in cache:
            return cache[cols]
        if not cols:
            cache[cols] = 0.0
            return 0.0
        X_subset = sm.add_constant(X[list(cols)])
        model = sm.OLS(y, X_subset).fit()
        cache[cols] = float(model.rsquared)
        return cache[cols]

    shapley_values = pd.Series(0.0, index=columns)
    factorial = math.factorial

    for predictor in columns:
        others = [c for c in columns if c != predictor]
        for k in range(len(others) + 1):
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            for subset in itertools.combinations(others, k):
                with_pred = tuple(sorted((*subset, predictor)))
                marginal = subset_r2(with_pred) - subset_r2(tuple(sorted(subset)))
                shapley_values[predictor] += weight * marginal

    full_model = sm.OLS(y, sm.add_constant(X)).fit()
    return shapley_values, float(full_model.rsquared)


def summarize_groups(shapley_values: pd.Series, group_lookup: Dict[str, str]) -> pd.DataFrame:
    df = (
        pd.DataFrame(
            {
                "predictor": shapley_values.index,
                "share": shapley_values.values,
                "group": [group_lookup.get(p, "Other") for p in shapley_values.index],
            }
        )
        .groupby("group", as_index=False)["share"]
        .sum()
    )
    return df


def select_rebased_series(
    model_df: pd.DataFrame,
    active_mapping: Dict[str, str],
    target_ticker: str,
    top_n: int,
) -> List[str]:
    if len(model_df) == 0:
        return [target_ticker]
    corrs = model_df.drop(columns=["date"]).corr()["r"].drop("r", errors="ignore")
    corrs = corrs.dropna().abs().sort_values(ascending=False)
    selected_predictors = corrs.head(top_n).index.tolist()
    tickers = [target_ticker]
    for predictor in selected_predictors:
        ticker = active_mapping.get(predictor)
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def compute_factor_correlations(
    log_returns: pd.DataFrame,
    usable_mapping: Dict[str, str],
    target_ticker: str,
    windows: Iterable[int],
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not usable_mapping:
        return pd.DataFrame()

    results: List[Dict[str, float]] = []
    for predictor, ticker in usable_mapping.items():
        row: Dict[str, float] = {"predictor": predictor}
        for window in windows:
            window_start = end_date - pd.DateOffset(months=window)
            window_df = log_returns.loc[log_returns.index >= window_start, [target_ticker, ticker]].dropna()
            if len(window_df) < 5:
                corr = np.nan
            else:
                corr = window_df[target_ticker].corr(window_df[ticker])
            row[f"{window}M"] = corr
        results.append(row)
    return pd.DataFrame(results)


def build_correlation_table(
    corr_df: pd.DataFrame,
    group_lookup: Dict[str, str],
    mapping: Dict[str, str],
    group_order: List[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame(columns=["Group", "Factor", *[f"{w}M Corr" for w in windows]])

    corr_df = corr_df.copy()
    corr_df["group"] = corr_df["predictor"].map(group_lookup)
    corr_df["factor"] = corr_df["predictor"].map(lambda p: f"{p} ({mapping[p]})")

    order_map = {group: idx for idx, group in enumerate(group_order)}
    corr_df["group_order"] = corr_df["group"].map(order_map).fillna(len(order_map))
    corr_df = corr_df.sort_values(["group_order", "factor"]).drop(columns=["group_order"])

    display_rows: List[Dict[str, object]] = []
    for group, grouped in corr_df.groupby("group", sort=False):
        for idx, row in enumerate(grouped.iterrows()):
            _, series = row
            entry: Dict[str, object] = {
                "Group": group if idx == 0 else "",
                "Factor": series["factor"],
            }
            for window in windows:
                val = series.get(f"{window}M")
                entry[f"{window}M Corr"] = val
            display_rows.append(entry)
    return pd.DataFrame(display_rows)


def plot_performance_overview(
    group_shares: pd.DataFrame,
    total_r2: float,
    corr_table: pd.DataFrame,
    windows: Iterable[int],
    group_order: List[str],
    output_dir: Path,
) -> None:
    bars = group_shares.copy()
    bars["pct"] = 100 * bars["share"]
    order_map = {group: idx for idx, group in enumerate(group_order)}
    bars = bars.sort_values(
        by="group",
        key=lambda col: col.map(order_map).fillna(len(order_map)),
    )
    idiosyncratic_pct = max(0.0, 100 * (1 - total_r2))
    bars = pd.concat(
        [bars, pd.DataFrame([{"group": "Idiosyncratic", "pct": idiosyncratic_pct}])],
        ignore_index=True,
    )

    fig = plt.figure(figsize=(9, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)

    ax_bar = fig.add_subplot(gs[0])
    ax_bar.barh(bars["group"], bars["pct"], color="#d81923")
    for y, pct in zip(bars["group"], bars["pct"]):
        ax_bar.text(pct + 1, y, f"{pct:.0f}%", va="center", fontsize=10)
    axis_max = max(5 + bars["pct"].max(), 10)
    ax_bar.set_xlim(0, axis_max)
    ax_bar.set_xlabel("Share of R^2 (%)")
    ax_bar.set_title("Performance Drivers", loc="left", fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    ax_table = fig.add_subplot(gs[1])
    ax_table.set_title("Factor Correlations", loc="left", fontweight="bold")
    ax_table.axis("off")
    col_labels = ["Group", "Factor", *[f"{w}M Corr" for w in windows]]
    if corr_table.empty:
        ax_table.text(0.01, 0.6, "Insufficient data to compute correlations.", fontsize=10)
    else:
        cell_text = []
        row_colors = []
        for idx, row in corr_table.iterrows():
            cols = [row["Group"], row["Factor"]]
            for window in windows:
                val = row[f"{window}M Corr"]
                cols.append("" if pd.isna(val) else f"{val:.2f}")
            cell_text.append(cols)
            row_colors.append("#f6f6f6" if (idx % 2 == 0) else "#ffffff")

        total_rows = len(cell_text) + 1  # include header
        row_height = 0.065
        table_height = min(0.9, total_rows * row_height)
        table_y = max(0.02, 0.5 - table_height / 2)

        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="left",
            colLoc="left",
            loc="upper left",
            bbox=[0, table_y, 0.94, table_height],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.05, 1.15)
        col_widths = {0: 0.08, 1: 0.2}
        narrow_width = 0.13
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_height(row_height)
            cell.set_width(col_widths.get(col_idx, narrow_width))
            cell.set_edgecolor("#ccccccff")
            cell.set_linewidth(0.5)
            if col_idx >= 1:
                cell.get_text().set_ha("center")
            if row_idx == 0:
                cell.set_facecolor("#d4d4d4")
                cell.set_height(0.1)
                cell.set_fontsize(10)
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor(row_colors[row_idx - 1])
    fig.savefig(output_dir / "performance_drivers.png", dpi=250)
    plt.close(fig)


def plot_rebased(prices: pd.DataFrame, series: List[str], output_dir: Path) -> None:
    missing = [s for s in series if s not in prices.columns]
    if missing:
        print(f"Skipping rebased plot because these series are missing: {missing}")
        return

    subset = prices[series].dropna()
    if subset.empty:
        print("Skipping rebased plot because no overlapping history exists for selected series.")
        return
    base = subset.iloc[0]
    rebased = subset / base * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    for ticker in series:
        ax.plot(rebased.index, rebased[ticker], linewidth=1.2, label=ticker)
    ax.set_title("Rebased Performance (100 = first common date)")
    ax.set_ylabel("Index level")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / "rebased_prices.png", dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    driver_config = DriverConfig(
        target=args.target,
        groups=CONFIG["groups"],
        rebased_top_n=CONFIG["rebased_top_n"],
    )

    history_start, analysis_start, end = determine_date_range(args.months, CORRELATION_WINDOWS)
    tickers_needed = {driver_config.target, *driver_config.factor_tickers}

    prices = download_prices(tickers_needed, history_start, end)
    log_returns = compute_log_returns(prices)
    model_returns = log_returns.loc[log_returns.index >= analysis_start]

    active_mapping, group_lookup, missing_defs = driver_config.active_predictors(log_returns.columns)
    for name, ticker in missing_defs:
        print(f"Warning: no usable data for '{name}' ({ticker}); excluding from regression.")

    usable_mapping: Dict[str, str] = {}
    for predictor, ticker in active_mapping.items():
        series = log_returns[ticker].dropna()
        if series.empty:
            print(f"Warning: ticker '{ticker}' for '{predictor}' has no return history in window; removing.")
        else:
            usable_mapping[predictor] = ticker

    if not usable_mapping:
        raise RuntimeError("No predictors with available return history. Adjust configuration or date range.")

    usable_group_lookup = {predictor: group_lookup[predictor] for predictor in usable_mapping}

    model_df = build_model_frame(model_returns, driver_config, usable_mapping)

    print(f"Observations used for regression: {len(model_df):,}")

    model = fit_ols(model_df)
    print(model.summary())

    shapley_values, total_r2 = compute_shapley_r2(model_df)
    group_shares = summarize_groups(shapley_values, usable_group_lookup)
    group_shares = (
        group_shares.set_index("group")
        .reindex(driver_config.group_order, fill_value=0)
        .reset_index()
    )

    corr_df = compute_factor_correlations(
        log_returns,
        usable_mapping,
        driver_config.target,
        CORRELATION_WINDOWS,
        end,
    )
    corr_table = build_correlation_table(
        corr_df,
        usable_group_lookup,
        usable_mapping,
        driver_config.group_order,
        CORRELATION_WINDOWS,
    )

    print("\nGroup contribution table (share of R^2):")
    print(group_shares.assign(share=lambda df: 100 * df["share"]).rename(columns={"share": "share_%"}))
    print(f"\nTotal R^2: {total_r2:.3f}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    plot_performance_overview(
        group_shares,
        total_r2,
        corr_table,
        CORRELATION_WINDOWS,
        driver_config.group_order,
        output_dir,
    )
    rebased_series = select_rebased_series(model_df, usable_mapping, driver_config.target, driver_config.rebased_top_n)
    prices_for_rebase = prices.loc[prices.index >= analysis_start]
    plot_rebased(prices_for_rebase, rebased_series, output_dir)

    print(f"\nCharts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
