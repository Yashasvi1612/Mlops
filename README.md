# MLOps Batch Job — Rolling Mean Signal Pipeline

A minimal, reproducible MLOps-style batch job that reads OHLCV market data,
computes a rolling-mean signal, and emits structured metrics + detailed logs.

---

## Project Structure

```
.
├── run.py            # Main pipeline
├── config.yaml       # Job configuration
├── data.csv          # Input OHLCV dataset (10 000 rows)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── metrics.json      # Sample output (successful run)
├── run.log           # Sample log (successful run)
└── README.md
```

---

## Local Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Execute the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

The final metrics JSON is printed to **stdout**; detailed logs go to `run.log`.

---

## Docker Build & Run

### Build the image

```bash
docker build -t mlops-task .
```

### Run the container

```bash
docker run --rm mlops-task
```

`data.csv` and `config.yaml` are baked into the image.  
`metrics.json` and `run.log` are written inside the container;
the final metrics JSON is printed to stdout before exit.

### Retrieve output files (optional)

```bash
docker run --rm -v "$(pwd)/out:/app" mlops-task
# metrics.json and run.log will appear in ./out/
```

---

## Configuration (`config.yaml`)

| Key       | Type    | Description                              |
|-----------|---------|------------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility    |
| `window`  | int     | Rolling mean window size (rows)          |
| `version` | string  | Pipeline version tag in metrics output   |

```yaml
seed: 42
window: 5
version: "v1"
```

---

## Signal Logic

| Condition                  | Signal |
|----------------------------|--------|
| `close > rolling_mean`     | 1      |
| `close <= rolling_mean`    | 0      |

The first `window - 1` rows have no rolling mean and are **excluded** from
signal computation (their signal value stays NaN and they are not counted
in `rows_processed`).

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 32,
  "seed": 42,
  "status": "success"
}
```

`rows_processed` = total rows − (window − 1) excluded warm-up rows.

### Error output shape

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found.",
  "latency_ms": 5
}
```

---

## Determinism

Running the pipeline twice with the same `config.yaml` and `data.csv`
always produces identical `metrics.json` output.  
The NumPy seed is set at startup via `numpy.random.seed(seed)`.

---

## Dependencies

| Package   | Version  |
|-----------|----------|
| numpy     | 1.24.4   |
| pandas    | 2.0.3    |
| PyYAML    | 6.0.1    |
