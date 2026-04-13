# Data

Place backtest results and TRUF stream data here.

## Expected files

- `full_history_results.json` — 51-year FRED backtest output (from `backtests/`)
- `truf_streams_*.csv` — TRUF stream historical data (when available from Stefan)

## Format

Backtest results should contain an `updates` or `innovation_history` key with a list of:
```json
{
  "stream_key": "US_CPI_YOY",
  "observed": 0.032,
  "predicted": 0.029,
  "innovation": 0.003,
  "innovation_zscore": 0.45,
  "kalman_gain": [0.12, 0.01, ...],
  "state_before": [...],
  "state_after": [...]
}
```
