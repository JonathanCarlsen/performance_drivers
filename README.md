# Performance Drivers with Shapley LMG

## Why this matters
Understanding what actually drives a stock’s performance is crucial for portfolio construction, attribution, and communication. A simple regression coefficient rarely tells the full story—multicollinearity and overlapping risk factors obscure which themes really explain returns. By decomposing R² into intuitive buckets (market, sector, macro, style) you can:

- Translate complex factor exposures into a narrative suitable for investment committees, client decks, or research notes.
- Spot shifts in dominant drivers (for example, when a utility starts behaving more like a macro play than a sector proxy).
- Benchmark discretionary theses against objective data before allocating capital.

## What the code does

`performance_drivers.py` automates an end-to-end attribution workflow:

1. **Data download** with `yfinance`: pulls daily adjusted prices for a target equity plus any factors defined in `CONFIG`.
2. **Log-return transformation** on rolling windows (6, 12, 24 months).
3. **OLS regression** to explain the target’s returns with the chosen factors.
4. **Shapley/LMG R² decomposition**:
   - The script computes the marginal contribution of every predictor across all subsets (LMG method) to create fair R² shares even when factors overlap.
   - Shares are aggregated into the configured groups (Market, Sector, Macro, Style, etc.).
5. **Correlation diagnostics**: calculates 6M and 12M Pearson correlations for each factor to show short- and medium-term co-movement strength.
6. **Visual output**:
   - A combined PNG (`outputs/performance_drivers.png`) with grouped R² bars and a factor correlation table.
   - A secondary chart (`outputs/rebased_prices.png`) that rebases the target and its most-correlated drivers to 100 at the common start date.

Key design choices:

- The `CONFIG` block is the only place you need to edit tickers or groups, making the workflow robust when you swap in different equities or factors.
- Missing tickers are skipped gracefully with warnings so experiments don’t break.
- `CORRELATION_WINDOWS` controls which lookbacks appear in the table—extend it if you need 3M/24M, etc.

## How to use it

1. **Clone and install dependencies**
   ```bash
   git clone <repo-url>
   cd performance_drivers
   python -m venv .venv
   .\.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
   python -m pip install -U pip
   pip install -r requirements.txt  # or pip install yfinance pandas numpy statsmodels matplotlib
   ```

2. **Edit `CONFIG` (optional)**
   - Set the default target ticker via `CONFIG["default_target"]`, change factor groups, or add new entries using the `{ "name": "...", "ticker": "..." }` format.
   - Adjust `rebased_top_n` to control how many highly correlated drivers appear in the rebased chart.

3. **Run the script**
   ```bash
   python performance_drivers.py --target IBE.MC --months 12
   ```
   Flags:
   - `--target`: overrides the dependent equity (defaults to whatever you put in `CONFIG["default_target"]`).
   - `--months` (choices: 6, 12, 24): lookback window for the regression; longer windows are still downloaded for correlation analysis.

4. **Interpret the outputs**
   - Console shows regression summary, Shapley group shares, and warnings for any missing tickers.
   - `outputs/performance_drivers.png`: single-page overview suitable for slides or emails.
   - `outputs/rebased_prices.png`: sanity-check chart to see how the target moved versus the strongest drivers.

5. **Swap to a new equity**
   - Update `CONFIG["default_target"]` (or just pass `--target NEW_TICKER` on the command line).
   - Add/remove factor entries as needed; the script automatically handles missing data.
   - Re-run the script; no other edits required.

This setup lets you quickly replicate an analyst-friendly attribution pack for any listed equity without touching R or Excel. Feel free to extend it with additional factor groups, custom lags, or alternative importance metrics if you need to scale the workflow further.
