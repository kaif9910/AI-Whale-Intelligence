from __future__ import annotations

from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


st.set_page_config(
    page_title="Whale Intelligence Terminal",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
STYLE_PATH = APP_DIR / "styles.css"
MASTER_OUTPUT_PATH = APP_DIR / "ai_master_data.csv"
NEWS_OUTPUT_PATH = APP_DIR / "master_news_database.csv"
SELECTED_COIN_PATH = APP_DIR / "last_selected_coin.txt"

if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

MASTER_FILE_CANDIDATES = [
    "ai_master_data.csv",
    "ai_master_table.csv",
    "ml_ready_data.csv",
    "master_table.csv",
]
NEWS_FILE_CANDIDATES = ["master_news_database.csv"]

COIN_CONFIG = {
    "BTC": {
        "coinbase_symbol": "BTC-USD",
        "keywords": ["bitcoin", "btc"],
    },
    "ETH": {
        "coinbase_symbol": "ETH-USD",
        "keywords": ["ethereum", "eth"],
    },
    "SOL": {
        "coinbase_symbol": "SOL-USD",
        "keywords": ["solana", "sol"],
    },
    "XRP": {
        "coinbase_symbol": "XRP-USD",
        "keywords": ["ripple", "xrp"],
    },
    "DOGE": {
        "coinbase_symbol": "DOGE-USD",
        "keywords": ["dogecoin", "doge"],
    },
}

analyzer = SentimentIntensityAnalyzer()


def read_synced_coin() -> str:
    try:
        coin = SELECTED_COIN_PATH.read_text(encoding="utf-8").strip().upper()
    except OSError:
        coin = "BTC"

    return coin if coin in COIN_CONFIG else "BTC"


def write_synced_coin(coin: str) -> None:
    coin = str(coin).upper().strip()
    if coin not in COIN_CONFIG:
        return

    try:
        SELECTED_COIN_PATH.write_text(coin, encoding="utf-8")
    except OSError:
        pass


@st.cache_data(show_spinner=False)
def generate_demo_data(hours: int = 120, coin: str = "BTC") -> tuple[pd.DataFrame, pd.DataFrame]:
    coin = coin.upper()
    rng = np.random.default_rng(42)
    time_index = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("h"),
        periods=hours,
        freq="h",
    )

    sentiment = np.clip(np.cumsum(rng.normal(0, 0.05, hours)), -1.25, 1.25)
    whale_burst = rng.integers(0, 6, hours)
    price_noise = rng.normal(0, 0.009, hours)
    momentum = 0.0025 * np.tanh(sentiment) + 0.001 * whale_burst
    price = 76000 * np.cumprod(1 + price_noise + momentum)
    volume = 2200 + whale_burst * 700 + rng.normal(0, 180, hours)
    trades = 6000 + whale_burst * 850 + rng.normal(0, 260, hours)

    master_df = pd.DataFrame(
        {
            "Time": time_index,
            "Coin": coin,
            "Price": price,
            "Volume": np.clip(volume, 250, None),
            "Trades": np.clip(trades, 1000, None),
            "News_Count": np.clip(whale_burst + rng.integers(0, 3, hours), 0, None),
            "Avg_Sentiment": np.clip(sentiment, -1, 1),
        }
    )
    master_df["Positive_News"] = np.where(master_df["Avg_Sentiment"] > 0.08, master_df["News_Count"] * 0.7, master_df["News_Count"] * 0.25).round().astype(int)
    master_df["Negative_News"] = np.where(master_df["Avg_Sentiment"] < -0.08, master_df["News_Count"] * 0.7, master_df["News_Count"] * 0.2).round().astype(int)
    master_df["Neutral_News"] = (master_df["News_Count"] - master_df["Positive_News"] - master_df["Negative_News"]).clip(lower=0)
    master_df["Has_News"] = (master_df["News_Count"] > 0).astype(int)
    master_df["Price_Change_%"] = master_df["Price"].pct_change() * 100
    master_df["Volume_Change_%"] = master_df["Volume"].pct_change() * 100
    master_df["Trades_Change_%"] = master_df["Trades"].pct_change() * 100
    master_df["Sentiment_Change"] = master_df["Avg_Sentiment"].diff()
    master_df["News_Sentiment_Impact"] = master_df["News_Count"] * master_df["Avg_Sentiment"]
    master_df = master_df.fillna(0)

    story_bank = [
        "Whale wallet rotation points to stealth accumulation",
        "Traders absorb heavy sell wall as sentiment improves",
        "Macro calm gives room for risk-on positioning",
        "Liquidity pocket opens after a burst of news activity",
        "Large spot orders reshape short-term directional bias",
        "Funding cools while on-chain flows stay constructive",
    ]
    news_rows = []
    for idx, row in master_df.tail(40).iterrows():
        if row["News_Count"] <= 0:
            continue
        news_rows.append(
            {
                "Published_Time": row["Time"],
                "Title": f"{coin}: {story_bank[idx % len(story_bank)]}",
                "Coin": coin,
                "Sentiment_Score": row["Avg_Sentiment"],
                "Link": f"https://example.com/demo-story-{idx}",
            }
        )
    return master_df, pd.DataFrame(news_rows)


def _first_existing_path(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = APP_DIR / candidate
        if path.exists():
            return path
    return None


def _normalize_master_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Time" not in df.columns:
        for fallback in ["Published_Time", "timestamp", "Date"]:
            if fallback in df.columns:
                df["Time"] = df[fallback]
                break
    if "Price" not in df.columns:
        for fallback in ["Close", "close", "price"]:
            if fallback in df.columns:
                df["Price"] = df[fallback]
                break
    for column in ["Volume", "Trades", "News_Count", "Avg_Sentiment"]:
        if column not in df.columns:
            df[column] = 0
    if "Coin" not in df.columns:
        df["Coin"] = "BTC"

    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df["Coin"] = df["Coin"].fillna("BTC").astype(str).str.upper()
    numeric_columns = [
        "Price", "Volume", "Trades", "News_Count", "Avg_Sentiment",
        "Positive_News", "Negative_News", "Neutral_News", "Has_News",
        "Price_Change_%", "Volume_Change_%", "Trades_Change_%",
        "Sentiment_Change", "News_Sentiment_Impact",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Time", "Price"]).sort_values("Time").reset_index(drop=True)
    if "Positive_News" not in raw_df.columns and "Negative_News" not in raw_df.columns:
        df["Positive_News"] = np.where(df["Avg_Sentiment"] > 0.1, df["News_Count"], 0)
        df["Negative_News"] = np.where(df["Avg_Sentiment"] < -0.1, df["News_Count"], 0)
        df["Neutral_News"] = (df["News_Count"] - df["Positive_News"] - df["Negative_News"]).clip(lower=0)
    if "Has_News" not in raw_df.columns:
        df["Has_News"] = (df["News_Count"] > 0).astype(int)
    if "Price_Change_%" not in raw_df.columns:
        df["Price_Change_%"] = df["Price"].pct_change() * 100
    if "Volume_Change_%" not in raw_df.columns:
        df["Volume_Change_%"] = df["Volume"].pct_change() * 100
    if "Trades_Change_%" not in raw_df.columns:
        df["Trades_Change_%"] = df["Trades"].pct_change() * 100
    if "Sentiment_Change" not in raw_df.columns:
        df["Sentiment_Change"] = df["Avg_Sentiment"].diff()
    if "News_Sentiment_Impact" not in raw_df.columns:
        df["News_Sentiment_Impact"] = df["News_Count"] * df["Avg_Sentiment"]
    return df.fillna(0)


def _normalize_news_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Published_Time", "Title", "Coin", "Sentiment_Score", "Link"])
    df = raw_df.copy()
    if "Published_Time" not in df.columns:
        for fallback in ["Time", "timestamp", "Date"]:
            if fallback in df.columns:
                df["Published_Time"] = df[fallback]
                break
    if "Title" not in df.columns:
        df["Title"] = "Untitled news pulse"
    if "Coin" not in df.columns:
        df["Coin"] = "BTC"
    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = 0
    if "Link" not in df.columns:
        df["Link"] = ""

    df["Published_Time"] = pd.to_datetime(df["Published_Time"], utc=True, errors="coerce")
    df["Coin"] = df["Coin"].fillna("BTC").astype(str).str.upper()
    df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce").fillna(0)
    return df.dropna(subset=["Published_Time"]).sort_values("Published_Time", ascending=False).reset_index(drop=True)

def fetch_live_news(coin: str, max_entries: int = 40) -> pd.DataFrame:
    coin = coin.upper()
    keywords = COIN_CONFIG[coin]["keywords"]
    rows = []

    try:
        response = requests.get(
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            headers={"User-Agent": "Whale-Intelligence-Terminal/1.0"},
            timeout=12,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception:
        return pd.DataFrame(columns=["Published_Time", "Title", "Coin", "Sentiment_Score", "Link"])

    for entry in getattr(feed, "entries", [])[:max_entries]:
        title = str(entry.get("title", "")).strip()
        summary = str(entry.get("summary", "")).strip()
        link = str(entry.get("link", "")).strip()
        published = pd.to_datetime(entry.get("published", None), utc=True, errors="coerce")
        text = f"{title} {summary}".lower()

        if not title or not link or pd.isna(published):
            continue
        if not any(keyword in text for keyword in keywords):
            continue

        rows.append(
            {
                "Published_Time": published,
                "Title": title,
                "Coin": coin,
                "Sentiment_Score": analyzer.polarity_scores(title)["compound"],
                "Link": link,
            }
        )

    news_df = pd.DataFrame(rows)
    if news_df.empty:
        return _normalize_news_df(news_df)

    news_df = news_df.drop_duplicates(subset=["Link"])
    return _normalize_news_df(news_df)


def fetch_coinbase_market_data(coin: str, limit: int = 120) -> pd.DataFrame:
    coin = coin.upper()
    product_id = COIN_CONFIG[coin]["coinbase_symbol"]
    empty_df = pd.DataFrame(
        columns=[
            "Time", "Coin", "Open", "High", "Low", "Price",
            "Volume", "Trades", "Price_Change_%", "Volume_Change_%",
            "Trades_Change_%",
        ]
    )

    try:
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            params={"granularity": 3600},
            headers={"User-Agent": "Whale-Intelligence-Terminal/1.0"},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return empty_df

    if not isinstance(data, list) or len(data) == 0:
        return empty_df

    df = pd.DataFrame(data, columns=["Unix_Time", "Low", "High", "Open", "Price", "Volume"])
    df["Time"] = pd.to_datetime(df["Unix_Time"], unit="s", utc=True).dt.floor("h")
    df["Coin"] = coin
    for column in ["Open", "High", "Low", "Price", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Trades"] = 0
    df = df.dropna(subset=["Time", "Open", "High", "Low", "Price", "Volume"])
    df = df.sort_values("Time").tail(limit).reset_index(drop=True)
    df["Price_Change_%"] = df["Price"].pct_change() * 100
    df["Volume_Change_%"] = df["Volume"].pct_change() * 100
    df["Trades_Change_%"] = 0.0

    final_df = df[
        [
            "Time", "Coin", "Open", "High", "Low", "Price",
            "Volume", "Trades", "Price_Change_%", "Volume_Change_%",
            "Trades_Change_%",
        ]
    ].fillna(0)
    return final_df


def build_hourly_sentiment(news_df: pd.DataFrame, coin: str) -> pd.DataFrame:
    coin = coin.upper()
    empty_df = pd.DataFrame(
        columns=[
            "Time", "Coin", "News_Count", "Avg_Sentiment",
            "Positive_News", "Negative_News", "Neutral_News",
        ]
    )
    if news_df.empty:
        return empty_df

    df = news_df.copy()
    df["Published_Time"] = pd.to_datetime(df["Published_Time"], utc=True, errors="coerce")
    df = df.dropna(subset=["Published_Time", "Title"])
    if df.empty:
        return empty_df

    df["Time"] = df["Published_Time"].dt.floor("h")
    df["Positive_Flag"] = (df["Sentiment_Score"] > 0.05).astype(int)
    df["Negative_Flag"] = (df["Sentiment_Score"] < -0.05).astype(int)
    df["Neutral_Flag"] = ((df["Sentiment_Score"] >= -0.05) & (df["Sentiment_Score"] <= 0.05)).astype(int)

    hourly = df.groupby("Time").agg(
        News_Count=("Title", "count"),
        Avg_Sentiment=("Sentiment_Score", "mean"),
        Positive_News=("Positive_Flag", "sum"),
        Negative_News=("Negative_Flag", "sum"),
        Neutral_News=("Neutral_Flag", "sum"),
    ).reset_index()
    hourly["Coin"] = coin
    return hourly[["Time", "Coin", "News_Count", "Avg_Sentiment", "Positive_News", "Negative_News", "Neutral_News"]]


def build_live_master_table(market_df: pd.DataFrame, sentiment_df: pd.DataFrame, coin: str) -> pd.DataFrame:
    coin = coin.upper()
    columns = [
        "Time", "Coin", "Open", "High", "Low", "Price", "Volume", "Trades",
        "Price_Change_%", "Volume_Change_%", "Trades_Change_%", "News_Count",
        "Avg_Sentiment", "Positive_News", "Negative_News", "Neutral_News",
        "Has_News", "Sentiment_Change", "News_Sentiment_Impact",
    ]
    empty_df = pd.DataFrame(columns=columns)
    if market_df.empty:
        return empty_df

    base = market_df.copy()
    base["Time"] = pd.to_datetime(base["Time"], utc=True, errors="coerce")
    base = base.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    if base.empty:
        return empty_df

    if sentiment_df.empty:
        merged = base.copy()
        merged["News_Count"] = 0
        merged["Avg_Sentiment"] = 0.0
        merged["Positive_News"] = 0
        merged["Negative_News"] = 0
        merged["Neutral_News"] = 0
    else:
        sent = sentiment_df.copy()
        sent["Time"] = pd.to_datetime(sent["Time"], utc=True, errors="coerce")
        sent = sent.dropna(subset=["Time"])
        merged = pd.merge(base, sent.drop(columns=["Coin"], errors="ignore"), on="Time", how="left")
        merged["News_Count"] = merged["News_Count"].fillna(0).astype(int)
        merged["Avg_Sentiment"] = merged["Avg_Sentiment"].fillna(0.0)
        merged["Positive_News"] = merged["Positive_News"].fillna(0).astype(int)
        merged["Negative_News"] = merged["Negative_News"].fillna(0).astype(int)
        merged["Neutral_News"] = merged["Neutral_News"].fillna(0).astype(int)

    merged["Coin"] = coin
    merged["Has_News"] = (merged["News_Count"] > 0).astype(int)
    merged["Sentiment_Change"] = merged["Avg_Sentiment"].diff().fillna(0)
    merged["News_Sentiment_Impact"] = merged["News_Count"] * merged["Avg_Sentiment"]
    return merged[columns].fillna(0)


def persist_live_files(master_df: pd.DataFrame, news_df: pd.DataFrame) -> None:
    if not master_df.empty:
        master_df.to_csv(MASTER_OUTPUT_PATH, index=False)
    if news_df is not None:
        news_df.to_csv(NEWS_OUTPUT_PATH, index=False)


def fetch_live_dashboard_data(coin: str, lookback_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    market_df = fetch_coinbase_market_data(coin, limit=max(lookback_hours, 72))
    news_df = fetch_live_news(coin)
    sentiment_df = build_hourly_sentiment(news_df, coin)
    master_df = build_live_master_table(market_df, sentiment_df, coin)

    if master_df.empty:
        fallback_master, fallback_news = generate_demo_data(hours=max(lookback_hours, 72), coin=coin)
        persist_live_files(fallback_master, fallback_news)
        return _normalize_master_df(fallback_master), _normalize_news_df(fallback_news), "Demo fallback"

    persist_live_files(master_df, news_df)
    return _normalize_master_df(master_df), _normalize_news_df(news_df), "Live fetch: Coinbase + CoinDesk RSS"


@st.cache_data(show_spinner=False)
def load_dashboard_data(uploaded_master: bytes | None, uploaded_news: bytes | None, default_coin: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if uploaded_master:
        master_df = pd.read_csv(pd.io.common.BytesIO(uploaded_master))
        news_df = pd.read_csv(pd.io.common.BytesIO(uploaded_news)) if uploaded_news else pd.DataFrame()
        return _normalize_master_df(master_df), _normalize_news_df(news_df), "Uploaded CSV"

    local_master = _first_existing_path(MASTER_FILE_CANDIDATES)
    if local_master is not None:
        master_df = pd.read_csv(local_master)
        news_path = _first_existing_path(NEWS_FILE_CANDIDATES)
        news_df = pd.read_csv(news_path) if news_path is not None else pd.DataFrame()
        return _normalize_master_df(master_df), _normalize_news_df(news_df), f"Local file: {local_master.name}"

    master_df, news_df = generate_demo_data(coin=default_coin)
    return _normalize_master_df(master_df), _normalize_news_df(news_df), "Demo mode"

def compute_market_state(df: pd.DataFrame) -> dict[str, object]:
    latest = df.iloc[-1]
    price_change = float(latest.get("Price_Change_%", 0.0))
    volume_change = float(latest.get("Volume_Change_%", 0.0))
    trades_change = float(latest.get("Trades_Change_%", 0.0))
    sentiment = float(latest.get("Avg_Sentiment", 0.0))
    news_count = float(latest.get("News_Count", 0.0))

    components = {
        "Price Momentum": 19 * np.tanh(price_change / 2.4),
        "Sentiment Pulse": 18 * np.tanh(sentiment * 2.7),
        "Whale Volume": 15 * np.tanh(volume_change / 8.5),
        "Trade Burst": 11 * np.tanh(trades_change / 8.5),
        "News Pressure": 7 * np.tanh(news_count / 3.0),
    }
    bullish_score = float(np.clip(50 + sum(components.values()), 1, 99))
    confidence = float(abs(bullish_score - 50) * 2)

    if bullish_score >= 68:
        signal = "BUY"
        tone = "Constructive"
    elif bullish_score <= 32:
        signal = "SELL"
        tone = "Defensive"
    else:
        signal = "HOLD"
        tone = "Balanced"

    risk_score = float(np.clip(abs(price_change) * 7 + abs(sentiment) * 25 + max(0, 6 - confidence), 0, 100))
    risk_label = "Low" if risk_score < 30 else "Medium" if risk_score < 60 else "High"
    driver_df = pd.DataFrame({"Driver": list(components.keys()), "Impact": list(components.values())}).sort_values("Impact")

    headline = {
        "BUY": "Whales are leaning risk-on with supportive momentum.",
        "SELL": "Distribution pressure is dominating the current tape.",
        "HOLD": "The tape is mixed, so patience looks stronger than force.",
    }[signal]

    return {
        "bullish_score": bullish_score,
        "confidence": confidence,
        "signal": signal,
        "tone": tone,
        "risk_label": risk_label,
        "headline": headline,
        "drivers": driver_df,
    }


def metric_card(title: str, value: str, delta: str, accent: str, delta_class: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {accent}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["Time"], y=df["Price"], name="Price", mode="lines",
            line=dict(color="#cf6a3d", width=3),
            fill="tozeroy", fillcolor="rgba(207,106,61,0.12)",
            hovertemplate="%{x}<br>Price: $%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Time"], y=df["Avg_Sentiment"], name="Sentiment", mode="lines+markers",
            line=dict(color="#1d6b71", width=2), marker=dict(size=5),
            hovertemplate="%{x}<br>Sentiment: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(margin=dict(l=12, r=12, t=16, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=0), height=360)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(18,52,59,0.10)", title_text="Price", secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text="Sentiment", secondary_y=True)
    return fig


def build_flow_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["Time"], y=df["Volume"], name="Volume",
            marker_color="rgba(29,107,113,0.72)",
            hovertemplate="%{x}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Time"], y=df["News_Count"], name="News Count", mode="lines+markers",
            line=dict(color="#cf6a3d", width=2.4),
            hovertemplate="%{x}<br>News: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(margin=dict(l=12, r=12, t=16, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=0), height=330)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(18,52,59,0.10)", secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    return fig


def build_driver_chart(driver_df: pd.DataFrame) -> go.Figure:
    colors = ["#b24444" if value < 0 else "#1f8f66" for value in driver_df["Impact"]]
    fig = go.Figure(go.Bar(x=driver_df["Impact"], y=driver_df["Driver"], orientation="h", marker_color=colors, hovertemplate="%{y}<br>Impact: %{x:.2f}<extra></extra>"))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=310, xaxis_title="Directional impact", yaxis_title="")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(18,52,59,0.10)")
    fig.update_yaxes(showgrid=False)
    return fig


def build_sentiment_mix(latest_row: pd.Series) -> go.Figure:
    values = [int(latest_row.get("Positive_News", 0)), int(latest_row.get("Neutral_News", 0)), int(latest_row.get("Negative_News", 0))]
    if sum(values) == 0:
        values = [1, 1, 1]
    fig = go.Figure(go.Pie(labels=["Positive", "Neutral", "Negative"], values=values, hole=0.62, marker=dict(colors=["#1f8f66", "#d0c2a6", "#b24444"]), textinfo="label+percent"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", height=300, showlegend=False)
    return fig


def render_news_wire(news_df: pd.DataFrame, selected_coin: str) -> None:
    if news_df.empty:
        st.info("No news feed detected yet. Live mode will still work with market-only data.")
        return
    filtered = news_df.copy()
    if "Coin" in filtered.columns:
        filtered = filtered[filtered["Coin"].str.upper() == selected_coin]
    if filtered.empty:
        st.info(f"No recent news items found for {selected_coin}.")
        return

    for _, row in filtered.head(6).iterrows():
        score = float(row.get("Sentiment_Score", 0))
        tone = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
        tone_color = "#1f8f66" if score > 0.1 else "#b24444" if score < -0.1 else "#7a6860"
        timestamp = pd.to_datetime(row["Published_Time"], utc=True, errors="coerce")
        timestamp_text = timestamp.strftime("%d %b %Y | %H:%M UTC") if pd.notna(timestamp) else "Unknown time"
        title = row.get("Title", "Untitled market note")
        link = row.get("Link", "")
        st.markdown(f"""
            <div class="news-card">
                <div class="news-time">{timestamp_text}</div>
                <div class="news-headline">{title}</div>
                <div class="news-score" style="color:{tone_color};">{tone} sentiment | score {score:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        if isinstance(link, str) and link.strip():
            st.markdown(f"[Open Source]({link})")


def format_market_price(value: float) -> str:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return "$0.00"

    abs_value = abs(value)
    if abs_value >= 1000:
        decimals = 2
    elif abs_value >= 1:
        decimals = 4
    else:
        decimals = 6

    return f"${value:,.{decimals}f}"


def format_delta(value: float, prefix: str = "", suffix: str = "") -> tuple[str, str]:
    if prefix == "$":
        formatted_value = format_market_price(abs(value))
        if value > 0:
            return f"+{formatted_value}{suffix}", "positive"
        if value < 0:
            return f"-{formatted_value}{suffix}", "negative"
        return f"{formatted_value}{suffix}", "neutral"

    if value > 0:
        return f"+{prefix}{value:,.2f}{suffix}", "positive"
    if value < 0:
        return f"{prefix}{value:,.2f}{suffix}", "negative"
    return f"{prefix}{value:,.2f}{suffix}", "neutral"


def get_live_state(coin: str, lookback_hours: int, refresh_counter: int, force_refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    needs_fetch = (
        force_refresh
        or "live_master_df" not in st.session_state
        or st.session_state.get("live_coin") != coin
        or st.session_state.get("live_lookback") != lookback_hours
        or st.session_state.get("live_refresh_counter") != refresh_counter
    )

    if needs_fetch:
        with st.spinner(f"Fetching latest {coin} market pulse..."):
            master_df, news_df, source_label = fetch_live_dashboard_data(coin, lookback_hours)
        st.session_state["live_master_df"] = master_df
        st.session_state["live_news_df"] = news_df
        st.session_state["live_source_label"] = source_label
        st.session_state["live_coin"] = coin
        st.session_state["live_lookback"] = lookback_hours
        st.session_state["live_refresh_counter"] = refresh_counter

    return (
        st.session_state.get("live_master_df", pd.DataFrame()),
        st.session_state.get("live_news_df", pd.DataFrame()),
        st.session_state.get("live_source_label", "Live fetch"),
    )


data_mode = st.sidebar.radio("Data mode", ["Live fetch", "Local / Upload"], index=0)

coin_options = sorted(COIN_CONFIG.keys())
synced_coin = read_synced_coin()
if (
    "coin_focus" not in st.session_state
    or st.session_state.get("last_synced_coin") != synced_coin
):
    st.session_state["coin_focus"] = synced_coin if synced_coin in coin_options else coin_options[0]
    st.session_state["last_synced_coin"] = synced_coin

preferred_coin = st.sidebar.selectbox("Coin focus", coin_options, key="coin_focus")
if preferred_coin != st.session_state.get("last_synced_coin"):
    write_synced_coin(preferred_coin)
    st.session_state["last_synced_coin"] = preferred_coin

lookback_hours = st.sidebar.slider("Lookback window (hours)", min_value=24, max_value=240, value=72, step=12)
show_raw_table = st.sidebar.checkbox("Show raw table", value=False)

master_df = pd.DataFrame()
news_df = pd.DataFrame()
source_label = ""

if data_mode == "Live fetch":
    auto_refresh_minutes = st.sidebar.slider("Auto refresh (minutes)", min_value=0, max_value=30, value=5, step=5)
    manual_refresh = st.sidebar.button("Fetch latest data", use_container_width=True)
    if auto_refresh_minutes > 0 and st_autorefresh is not None:
        refresh_counter = st_autorefresh(interval=auto_refresh_minutes * 60 * 1000, key="live-refresh")
    else:
        refresh_counter = 0
        if auto_refresh_minutes > 0 and st_autorefresh is None:
            st.sidebar.caption("Install `streamlit-autorefresh` to enable timed refresh.")

    master_df, news_df, source_label = get_live_state(preferred_coin, lookback_hours, refresh_counter, manual_refresh)
else:
    uploaded_master = st.sidebar.file_uploader("Load master table CSV", type="csv")
    uploaded_news = st.sidebar.file_uploader("Load news feed CSV", type="csv")
    master_df, news_df, source_label = load_dashboard_data(
        uploaded_master.getvalue() if uploaded_master else None,
        uploaded_news.getvalue() if uploaded_news else None,
        preferred_coin,
    )

if master_df.empty:
    st.error("No master data available. Try live fetch or upload a CSV.")
    st.stop()

available_coins = sorted(master_df["Coin"].dropna().astype(str).str.upper().unique().tolist())
selected_coin = preferred_coin if preferred_coin in available_coins else available_coins[0]
filtered_df = master_df[master_df["Coin"].str.upper() == selected_coin].copy().tail(lookback_hours).reset_index(drop=True)
if filtered_df.empty:
    st.error(f"No rows available for {selected_coin}.")
    st.stop()

latest = filtered_df.iloc[-1]
previous = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest
state = compute_market_state(filtered_df)
price_delta_text, price_delta_class = format_delta(float(latest["Price"] - previous["Price"]), prefix="$")
volume_delta_text, volume_delta_class = format_delta(float(latest.get("Volume_Change_%", 0.0)), suffix="% vs prev hour")
sentiment_delta_text, sentiment_delta_class = format_delta(float(latest.get("Avg_Sentiment", 0.0) - previous.get("Avg_Sentiment", 0.0)), suffix=" shift")
confidence_delta_text, confidence_delta_class = format_delta(state["confidence"], suffix="% conviction")

st.markdown(f"""
<div class="hero-shell">
  <div class="hero-grid">
    <div>
      <div class="eyebrow">AI Whale Intelligence System</div>
      <h1 class="hero-title">Whale Intelligence Terminal</h1>
      <p class="hero-subtitle">
        A live command center for market structure, headline pressure, whale activity,
        and short-horizon conviction. Built to help you read whether the tape is being
        accumulated, distributed, or simply digesting noise.
      </p>
    </div>
    <div class="hero-rail">
      <div class="mini-chip"><div class="mini-label">Data Source</div><div class="mini-value">{source_label}</div></div>
      <div class="mini-chip"><div class="mini-label">Coin Focus</div><div class="mini-value">{selected_coin} | {len(filtered_df)} tracked hours</div></div>
      <div class="mini-chip"><div class="mini-label">Current Bias</div><div class="mini-value">{state['signal']} | {state['tone']} tape</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Pulse Board</div>', unsafe_allow_html=True)
st.markdown('<p class="section-copy">A fast read on price, flow, sentiment, and conviction before you dive into the deeper charts.</p>', unsafe_allow_html=True)
metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Spot Price", format_market_price(float(latest["Price"])), price_delta_text, "copper", price_delta_class)
with metric_cols[1]:
    metric_card("Whale Flow", f"{float(latest['Volume']):,.0f}", volume_delta_text, "teal", volume_delta_class)
with metric_cols[2]:
    metric_card("Sentiment Pulse", f"{float(latest['Avg_Sentiment']):.2f}", sentiment_delta_text, "ink", sentiment_delta_class)
with metric_cols[3]:
    metric_card("Conviction", f"{state['confidence']:.0f}%", confidence_delta_text, "teal", confidence_delta_class)

left_col, right_col = st.columns([1.35, 0.85], gap="large")
with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Price Vs. Sentiment Rhythm</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Watch how spot price reacts while headline tone slides from fearful to constructive.</p>', unsafe_allow_html=True)
    st.plotly_chart(build_price_chart(filtered_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with right_col:
    st.markdown(f"""
        <div class="signal-panel">
            <div class="signal-tag">Live Market Call</div>
            <div class="signal-value">{state['signal']}</div>
            <div class="signal-copy">{state['headline']}</div>
            <div style="margin-top:1rem; display:grid; gap:0.7rem;">
                <div><strong>Bullish Score:</strong> {state['bullish_score']:.1f}/100</div>
                <div><strong>Risk Regime:</strong> {state['risk_label']}</div>
                <div><strong>Last Snapshot:</strong> {pd.to_datetime(latest['Time'], utc=True).strftime('%d %b %Y | %H:%M UTC')}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

tab_overview, tab_news, tab_data = st.tabs(["Market Story", "News Wire", "Data Deck"])
with tab_overview:
    chart_col, driver_col = st.columns([1.15, 0.85], gap="large")
    with chart_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Volume + News Pressure</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">High-volume bursts alongside rising news counts usually mark the hours when whales leave a footprint.</p>', unsafe_allow_html=True)
        st.plotly_chart(build_flow_chart(filtered_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with driver_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What Is Driving The Bias</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">This panel shows the latest directional forces shaping the BUY, HOLD, or SELL tone.</p>', unsafe_allow_html=True)
        st.plotly_chart(build_driver_chart(state['drivers']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    mix_col, summary_col = st.columns([0.78, 1.22], gap="large")
    with mix_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Headline Mix</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Positive, neutral, and negative story balance for the latest tracked hour.</p>', unsafe_allow_html=True)
        st.plotly_chart(build_sentiment_mix(latest), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with summary_col:
        rolling_sentiment = filtered_df['Avg_Sentiment'].tail(12).mean()
        hourly_range = filtered_df['Price'].tail(24).max() - filtered_df['Price'].tail(24).min()
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Desk Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <p class="section-copy">
                Over the last <strong>{lookback_hours} hours</strong>, the desk is reading a
                <strong>{state['tone'].lower()}</strong> structure for <strong>{selected_coin}</strong>.
                The latest hour printed <strong>{int(latest['News_Count'])}</strong> tracked headlines,
                a sentiment score of <strong>{float(latest['Avg_Sentiment']):.2f}</strong>,
                and a rolling 12-hour mood average of <strong>{rolling_sentiment:.2f}</strong>.
                Spot has rotated through a 24-hour range of <strong>{format_market_price(hourly_range)}</strong>,
                which tells you how much room the market still has before the next expansion or fade.
            </p>
            <p class="footer-note">
                Live fetch mode writes fresh local CSV files too, so your notebook and dashboard can share the same dataset.
            </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_news:
    st.markdown('<div class="section-title">News Wire</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Latest headlines feeding the sentiment engine. In live mode the app fetches this feed automatically.</p>', unsafe_allow_html=True)
    render_news_wire(news_df, selected_coin)

with tab_data:
    st.markdown('<div class="section-title">Data Deck</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Use the raw view when you want to inspect the exact rows powering the terminal.</p>', unsafe_allow_html=True)
    display_columns = [col for col in ["Time", "Coin", "Price", "Volume", "Trades", "News_Count", "Avg_Sentiment", "Price_Change_%", "Volume_Change_%", "Trades_Change_%", "Sentiment_Change", "News_Sentiment_Impact"] if col in filtered_df.columns]
    styled_df = filtered_df[display_columns].copy()
    st.dataframe(styled_df.sort_values("Time", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download current view as CSV", styled_df.to_csv(index=False).encode("utf-8"), file_name=f"whale_terminal_{selected_coin.lower()}.csv", mime="text/csv")

if show_raw_table:
    st.markdown('---')
    st.markdown('<div class="section-title">Expanded Raw Table</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
