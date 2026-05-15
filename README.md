# AI Whale Intelligence Terminal

AI Whale Intelligence Terminal is a Streamlit dashboard for tracking crypto market movement, headline sentiment, and short-term BUY / SELL / HOLD bias for coins such as BTC, ETH, SOL, XRP, and DOGE.

The project combines live market candles, crypto news headlines, sentiment analysis, and dashboard visuals into one simple intelligence terminal.


---

## 🚀 Live Demo

🔗 **[Open Live App → ai-whale-intelligence.streamlit.app](https://ai-whale-intelligence.streamlit.app/)**

---


## What It Does

- Fetches live hourly market data from Coinbase Exchange
- Scrapes crypto headlines from CoinDesk RSS
- Scores news sentiment with VADER NLP
- Aligns news and market data by hour
- Builds a master dataset for analysis and ML workflows
- Shows price, volume, sentiment, conviction, and market bias in Streamlit
- Supports coin focus for BTC, ETH, SOL, XRP, and DOGE
- Handles low-price coins like DOGE with decimal-safe price formatting
- Auto-refreshes live data in the dashboard
- Syncs notebook-selected coin with the Streamlit dashboard

## Project Structure

```text
.
|-- app.py                  # Main Streamlit dashboard
|-- styles.css              # Custom dashboard styling
|-- requirements.txt        # Python dependencies
|-- runtime.txt             # Python version hint for deployment
|-- .streamlit/
|   `-- config.toml         # Streamlit theme/server config
|-- .gitignore              # Ignored generated files
`-- README.md               # Project documentation
```

Generated runtime files are intentionally ignored:

```text
ai_master_data.csv
master_news_database.csv
master_news_database_*.csv
last_selected_coin.txt
```

The app can recreate these files while running.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Requests
- Feedparser
- VADER Sentiment

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-whale-intelligence.git
cd ai-whale-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Open the app:

```text
http://localhost:8501
```

## Run From Jupyter Notebook

Use this cell if you want to start Streamlit from your notebook:

```python
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\imdka\OneDrive\Documents\New project")
APP_FILE = PROJECT_DIR / "app.py"

streamlit_process = subprocess.Popen([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(APP_FILE),
    "--server.port",
    "8501"
])
```

If you select a coin in the notebook, write it to `last_selected_coin.txt` so the dashboard opens the same coin first:

```python
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\imdka\OneDrive\Documents\New project")
coin = "DOGE"

(PROJECT_DIR / "last_selected_coin.txt").write_text(coin, encoding="utf-8")
```

## Dashboard Modes

The sidebar has two data modes:

```text
Live fetch
```

The app fetches fresh market and news data by itself.

```text
Local / Upload
```

The app reads a local CSV or an uploaded master table.

## Supported Coins

```text
BTC
ETH
SOL
XRP
DOGE
```

DOGE and other low-price coins are displayed with extra decimal precision, so values like `$0.183421` do not get rounded to `$0`.

## Deployment

This project is ready for Streamlit Community Cloud.

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Select the branch, usually `main`.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app.

No API keys are required for the current version.

## GitHub Push Commands

After creating an empty GitHub repository, run:

```bash
git init
git add app.py styles.css requirements.txt runtime.txt README.md .gitignore .streamlit/config.toml
git commit -m "Initial Streamlit crypto dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-whale-intelligence.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Troubleshooting

If Streamlit shows BTC after running DOGE in the notebook, refresh the app or restart Streamlit. The dashboard reads `last_selected_coin.txt` on startup and uses it as the default coin.

If `localhost:8501` does not open, make sure Streamlit is running:

```bash
streamlit run app.py
```

If DOGE price shows `$0`, restart the app so the latest formatting code is loaded.

If dependencies are missing, reinstall:

```bash
pip install -r requirements.txt
```

## Disclaimer

This project is for education, research, and dashboard-building practice. It is not financial advice. Crypto markets are volatile, and model or sentiment outputs should not be used as the only basis for trading decisions.
