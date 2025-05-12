import streamlit as st
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import re
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
import spacy
import requests
import time
import pandas as pd
from difflib import get_close_matches
import re
from prophet import Prophet
import pdfplumber
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Streamlit app setup
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from transformers import pipeline
llm = pipeline("text-generation", model="gpt2")  

# FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.warning("FAISS not available. RAG features will be limited.")

# Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    st.warning("SentenceTransformer not available. RAG features will be limited.")

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Prediction features will be limited.")

alpha_vantage_api_key = "HCG6M9ET81906BAF"
NEWS_API_KEY = "647dc961385847ddbe418167ecc13fc6"
DEFAULT_TICKER_SUFFIX = ".NS"  
NEWS_REFRESH_INTERVAL = 900  # 15 minutes
NEWS_QUERY = "stocks+finance+market+india"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Initialize RAG components
document_store = []
PDF_FOLDER = r"C:\Users\HP\Desktop\AI Agent for Market Analysis"
PDF_FILENAME = "Market_reference.pdf" 
PDF_PATH = os.path.join(PDF_FOLDER, PDF_FILENAME)

if FAISS_AVAILABLE and SENTENCE_TRANSFORMER_AVAILABLE:
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        if os.path.exists(PDF_PATH):
            with pdfplumber.open(PDF_PATH) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if not page_text:
                        continue

                    chunks = [page_text[i:i+500] for i in range(0, len(page_text), 500)]

                    for chunk in chunks:
                        document_store.append({
                            "filename": f"{PDF_FILENAME}_page{page_num+1}",
                            "content": chunk
                        })
                # st.success(f"Loaded PDF '{PDF_FILENAME}' into the document store!")

            texts = [doc['content'] for doc in document_store]
            embeddings = embed_model.encode(texts, convert_to_numpy=True)
            index.add(embeddings)


        else:
            st.warning(f"PDF file '{PDF_FILENAME}' not found in folder '{PDF_FOLDER}'.")

    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        embed_model = None
        index = None
else:
    embed_model = None
    index = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "news_thread_started" not in st.session_state:
    st.session_state.news_thread_started = False
if "last_news_fetch" not in st.session_state:
    st.session_state.last_news_fetch = datetime.now() - timedelta(hours=1)
if "recent_news" not in st.session_state:
    st.session_state.recent_news = []
if "stock_cache" not in st.session_state:
    st.session_state.stock_cache = {}

# Fetch news
def fetch_news():
    try:
        url = f"https://newsapi.org/v2/everything?q={NEWS_QUERY}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=25"
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"News API error: {response.status_code}")
            return []
            
        data = response.json()
        
        if data.get('status') != 'ok':
            st.error(f"News API error: {data.get('message', 'Unknown error')}")
            return []
            
        articles = []
        for article in data.get('articles', []):
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            source = article.get('source', {}).get('name', 'Unknown Source')
            published_at = article.get('publishedAt', datetime.now().isoformat())
            url = article.get('url', '')
            
            combined_text = f"Title: {title}. Description: {description}. Source: {source}. Date: {published_at}. Link: {url}".strip()
            if combined_text and len(combined_text) > 50:  # Filter out very short articles
                articles.append({
                    'text': combined_text,
                    'title': title,
                    'source': source,
                    'date': published_at,
                    'url': url
                })
        
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def update_rag(news_articles):
    global document_store, index
    if not FAISS_AVAILABLE or not SENTENCE_TRANSFORMER_AVAILABLE or not news_articles:
        return False
        
    try:
        texts = [article['text'] for article in news_articles]
        
        embeddings = embed_model.encode(texts)
        
        document_store.extend(news_articles)
        index.add(np.array(embeddings, dtype=np.float32))
        
        return True
    except Exception as e:
        st.error(f"Error updating RAG index: {str(e)}")
        return False

def retrieve_docs(query, k=1):
    if not FAISS_AVAILABLE or not SENTENCE_TRANSFORMER_AVAILABLE:
        return []

    if index.ntotal == 0:
        return []

    try:
        query_embedding = embed_model.encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(document_store) and idx >= 0:
                doc = document_store[idx]
                result_entry = {
                    'score': float(distances[0][i])
                }

                if 'text' in doc:
                    result_entry.update({
                        'text': doc['text'],
                        'title': doc.get('title', 'Untitled'),
                        'source': doc.get('source', 'Unknown'),
                        'date': doc.get('date', ''),
                        'url': doc.get('url', '')
                    })
                elif 'content' in doc:
                    result_entry.update({
                        'text': doc['content'],
                        'title': doc.get('filename', 'Document'),
                        'source': 'Reference PDF',
                        'date': '',
                        'url': ''
                    })

                results.append(result_entry)

        return results

    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def is_valid_ticker(ticker):
    return bool(re.match(r'^[A-Z]{1,5}$', ticker))

# Extract tickers from query
def extract_tickers(query):
    """
    Extract NSE tickers from a user query based on official NSE equity list.
    
    Args:
        query (str): User query text (e.g., "What's the latest on Reliance Industries?")
        
    Returns:
        list: List of detected NSE tickers (e.g., ['RELIANCE.NS'])
    """
    try:
        csv_path = r"C:\Users\HP\Desktop\AI Agent for Market Analysis\EQUITY_L.csv"
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df['SYMBOL'] = df['SYMBOL'].str.strip()
        df['NAME OF COMPANY'] = df['NAME OF COMPANY'].str.strip()

        query_lower = query.lower()
        tickers_found = []

        for symbol in df['SYMBOL']:
            pattern = r'\b' + re.escape(symbol.lower()) + r'\b'
            if re.search(pattern, query_lower):
                # tickers_found.append(symbol + ".NS")
                tickers_found.append(symbol)

        if not tickers_found:
            for _, row in df.iterrows():
                company_name = row['NAME OF COMPANY'].lower()
                if company_name in query_lower:
                    # tickers_found.append(row['SYMBOL'] + ".NS")
                    tickers_found.append(row['SYMBOL'])

        return tickers_found

    except Exception as e:
        print(f"Ticker extraction error: {e}")
        return []

    
def fetch_stock_data(ticker, period="1d", force_refresh=False):
    cache_key = f"{ticker}_{period}"
    
    if not force_refresh and cache_key in st.session_state.stock_cache:
        cache_entry = st.session_state.stock_cache[cache_key]
        cache_age = datetime.now() - cache_entry['timestamp']
        
        if cache_age.total_seconds() < 300:
            return cache_entry['data']
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
            
        if period == "1d":
            last = hist.iloc[-1]
            data = {
                "price": last["Close"], 
                "open": last["Open"],
                "high": last["High"], 
                "low": last["Low"],
                "volume": last["Volume"],
                "change": (last["Close"] - hist.iloc[0]["Open"]) / hist.iloc[0]["Open"] * 100 if len(hist) > 0 else 0
            }
        else:
            data = hist
            
        st.session_state.stock_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Get stock info
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None),
            'eps': info.get('trailingEPS', None),
            'dividend_yield': info.get('dividendYield', None),
            'target_high': info.get('targetHighPrice', None),
            'target_low': info.get('targetLowPrice', None),
            'target_mean': info.get('targetMeanPrice', None),
            'recommendation': info.get('recommendationKey', 'Unknown')
        }
    except:
        return {
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown'
        }

# Stock prediction using Prophet
def stock_prediction(ticker, days=30):
    if not PROPHET_AVAILABLE:
        return None
        
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")

        if hist is None or hist.empty:
            print(f"No historical data found for {ticker}")
            return None

        if len(hist) < 60:
            print(f"Not enough data for {ticker}. Only {len(hist)} rows.")
            return None

        df = pd.DataFrame({
            "ds": hist.index.tz_localize(None),
            "y": hist["Close"]
        })

        try:
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
        except Exception as model_error:
            print(f"Error while fitting or forecasting for {ticker}: {model_error}")
            traceback.print_exc()
            return None


        last_price = hist["Close"].iloc[-1]
        predicted_price = forecast["yhat"].iloc[-1]
        change = predicted_price - last_price
        change_pct = (change / last_price) * 100

        return {
            'current_price': last_price,
            'predicted_price': predicted_price,
            'change': change,
            'change_pct': change_pct,
            'forecast': forecast,
            'history': hist
        }

    except Exception as e:
        print(f"Prediction error for {ticker}: {str(e)}")
        traceback.print_exc()
        return None

def analyze_sentiment(texts, ticker=None):
    if not texts:
        return 0.5  
        
    if isinstance(texts, list):
        combined_text = " ".join(texts).lower()
    else:
        combined_text = texts.lower()
        
    positive_keywords = [
        'growth', 'profit', 'gain', 'rally', 'bullish', 'positive', 'outperform',
        'buy', 'strong buy', 'upgrade', 'increase', 'good', 'great', 'excellent',
        'beat', 'exceed', 'up', 'upside', 'opportunity', 'success', 'momentum',
        'recommended', 'advantage', 'promising', 'impressive', 'boost'
    ]
    
    negative_keywords = [
        'loss', 'down', 'bearish', 'downgrade', 'sell', 'strong sell', 'decline',
        'drop', 'fall', 'underperform', 'weak', 'warning', 'bad', 'poor',
        'concern', 'risk', 'volatile', 'avoid', 'miss', 'below', 'disappointing',
        'challenge', 'trouble', 'negative', 'crash', 'plunge', 'struggle'
    ]
    
    positive_count = sum(combined_text.count(word) for word in positive_keywords)
    negative_count = sum(combined_text.count(word) for word in negative_keywords)
    
    if ticker:
        ticker_context = 20  
        clean_ticker = ticker.replace(".NS", "")  
        
        ticker_positions = [m.start() for m in re.finditer(clean_ticker, combined_text)]
        
        for pos in ticker_positions:
            start = max(0, pos - ticker_context)
            end = min(len(combined_text), pos + ticker_context)
            context = combined_text[start:end]
            
            positive_count += sum(context.count(word) for word in positive_keywords) * 2  
            negative_count += sum(context.count(word) for word in negative_keywords) * 2  
    
    total = positive_count + negative_count
    if total == 0:
        return 0.5  
    else:
        return min(1.0, max(0.0, positive_count / total))

def trade_signal(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if len(hist) < 20:
            return "Neutral", 0.5 
            
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        
        last_price = hist['Close'].iloc[-1]
        last_sma20 = hist['SMA20'].iloc[-1]
        last_sma50 = hist['SMA50'].iloc[-1]
        
        prev_sma20 = hist['SMA20'].iloc[-2]
        prev_sma50 = hist['SMA50'].iloc[-2]
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        current_rsi = hist['RSI'].iloc[-1]
        
        momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[-6]) - 1
        
        score = 0.5  
        signal = "Neutral"
        
        if last_sma20 > last_sma50 and prev_sma20 <= prev_sma50:
            signal = "Buy"
            score += 0.2
        elif last_sma20 < last_sma50 and prev_sma20 >= prev_sma50:
            signal = "Sell"
            score -= 0.2
        
        if current_rsi > 70:
            score -= 0.1
        elif current_rsi < 30:
            score += 0.1
            
        if momentum > 0.05:
            score += 0.1
        elif momentum < -0.05:
            score -= 0.1
        
        print(f"{current_rsi},{last_sma20},{last_sma50},{momentum},{score}")
        if score >= 0.6:
            return "Buy", score
        elif score <= 0.4:
            return "Sell", score
        else:
            return "Hold", score
            
    except Exception as e:
        print(f"Trade signal error for {ticker}: {str(e)}")
        return "Neutral", 0.5


from difflib import get_close_matches

def detect_intent(query):
    q_lower = query.lower()

    keywords = {
        "comparison": ["compare", "vs", "versus", "better"],
        "recommendation": ["recommend", "should i", "good buy", "invest in"],
        "price_lookup": ["price", "quote", "trading at", "current"],
        "prediction": ["predict", "forecast", "future", "outlook"],
        "news": ["news", "headlines", "latest", "developments"],
        "technical": ["technical", "chart", "indicator"],
        "trend": ["trend", "performance", "history"],
        "sentiment": ["sentiment", "feeling", "market sentiment"],
        "help": ["help", "can you", "what can"]
    }

    for intent, words in keywords.items():
        for word in words:
            if word in q_lower:
                return intent

    return "general"


def create_comparison_chart(tickers):
    if not tickers or len(tickers) < 1:
        return None
        
    try:
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if not hist.empty:
                first_price = hist['Close'].iloc[0]
                hist['NormalizedReturn'] = (hist['Close'] / first_price - 1) * 100
                data[ticker] = hist
                
        if not data:
            return None
            
        fig = go.Figure()
        
        for ticker, hist in data.items():
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['NormalizedReturn'],
                mode='lines',
                name=ticker,
                hovertemplate=
                '<b>%{x}</b><br>' +
                'Return: %{y:.2f}%<br>'
            ))
            
        fig.update_layout(
            title='Comparative Performance (% Return)',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            hovermode='x unified',
            legend_title='Tickers',
            height=500
        )
        
        return fig
    except Exception as e:
        print(f"Chart error: {str(e)}")
        return None

# Generate technical analysis chart
def create_technical_chart(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
            
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA200'] = hist['Close'].rolling(window=200).mean()
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=(f'{ticker} Price & Moving Averages', 'RSI (14)'),
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA20'],
            line=dict(color='blue', width=1),
            name='SMA20'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA50'],
            line=dict(color='orange', width=1),
            name='SMA50'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA200'],
            line=dict(color='red', width=1),
            name='SMA200'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['RSI'],
            line=dict(color='purple', width=1),
            name='RSI'
        ), row=2, col=1)
        
        fig.add_shape(
            type="line", line_color="red", line_width=1, opacity=0.3,
            x0=hist.index[0], x1=hist.index[-1], y0=70, y1=70,
            row=2, col=1
        )
        
        fig.add_shape(
            type="line", line_color="green", line_width=1, opacity=0.3,
            x0=hist.index[0], x1=hist.index[-1], y0=30, y1=30,
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    except Exception as e:
        print(f"Technical chart error: {str(e)}")
        return None

def format_number(num):
    if num is None:
        return "N/A"
        
    if abs(num) >= 1e9:
        return f"₹{num/1e9:.2f}B" if str(ticker).endswith(".NS") else f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"₹{num/1e6:.2f}M" if str(ticker).endswith(".NS") else f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"₹{num/1e3:.2f}K" if str(ticker).endswith(".NS") else f"${num/1e3:.2f}K"
    else:
        return f"₹{num:.2f}" if str(ticker).endswith(".NS") else f"${num:.2f}"

def format_pct(num):
    if num is None:
        return "N/A"
    return f"{num:.2f}%"

# Final decision combining all signals
def combined_decision(ticker, retrieved_docs=None):
    print(f"\n📊 Running decision engine for {ticker}")

    stock_data = fetch_stock_data(ticker, period="5d")
    print(f"Stock Data: {stock_data}")

    prediction_data = stock_prediction(ticker)
    print(f"Prediction Data: {prediction_data}")

    technical_signal, technical_score = trade_signal(ticker)
    print(f"Technical Signal: {technical_signal}, Score: {technical_score}")

    doc_texts = [doc['text'] for doc in retrieved_docs] if retrieved_docs else []
    sentiment_score = analyze_sentiment(doc_texts, ticker)
    print(f"Sentiment Score: {sentiment_score}")

    market_component = 0.5
    try:
        market_ticker = "^NSEI" if ticker.endswith(".NS") else "^GSPC"
        market = yf.Ticker(market_ticker)
        market_hist = market.history(period="5d")
        print(f"Market Data ({market_ticker}): {market_hist}")

        if not market_hist.empty:
            change = (market_hist['Close'].iloc[-1] - market_hist['Open'].iloc[0]) / market_hist['Open'].iloc[0]
            market_component = 0.5 + min(0.5, max(-0.5, change))
    except Exception as e:
        print(f"Market data fetch error: {e}")

    print(f"Market Component: {market_component}")

    w_sentiment = 0.25
    w_technical = 0.35
    w_prediction = 0.25
    w_market = 0.15

    sentiment_component = sentiment_score
    technical_component = technical_score

    prediction_component = 0.5  
    if prediction_data:
        change_pct = prediction_data['change_pct']
        if change_pct > 5:
            prediction_component = 0.8
        elif change_pct > 2:
            prediction_component = 0.65
        elif change_pct < -5:
            prediction_component = 0.2
        elif change_pct < -2:
            prediction_component = 0.35
        else:
            prediction_component = 0.5
    print(f"Prediction Component: {prediction_component}")

    final_score = (
        sentiment_component * w_sentiment +
        technical_component * w_technical +
        prediction_component * w_prediction +
        market_component * w_market
    )

    print(f"\n✅ Final decision score for {ticker}: {final_score:.2f}")

    if final_score >= 0.7:
        recommendation = "Strong Buy"
    elif final_score >= 0.6:
        recommendation = "Buy"
    elif final_score <= 0.3:
        recommendation = "Sell"
    elif final_score <= 0.4:
        recommendation = "Hold (Negative Bias)"
    else:
        recommendation = "Hold"

    print(f"📈 Recommendation: {recommendation}\n")

    return {
        'score': final_score,
        'recommendation': recommendation,
        'components': {
            'sentiment': sentiment_component,
            'technical': technical_component,
            'prediction': prediction_component,
            'market': market_component
        }
    }

def news_fetch_thread():
    while True:
        try:
            time_since_last = (datetime.now() - st.session_state.last_news_fetch).total_seconds()
            
            if time_since_last > NEWS_REFRESH_INTERVAL:
                print("Fetching fresh news...")
                news = fetch_news()
                
                if news:
                    st.session_state.recent_news = news
                    st.session_state.last_news_fetch = datetime.now()
                    update_rag(news)
                    print(f"Fetched {len(news)} news articles")
        except Exception as e:
            print(f"Error in news thread: {str(e)}")
            
        # Sleep to avoid excessive CPU use
        time.sleep(30)  

# Format news for display
def format_news_item(news_item):
    return f"📰 **{news_item['title']}**\n" \
           f"*Source: {news_item['source']} - {news_item['date']}*\n" \
           f"[Read more]({news_item['url']})"

def get_help_message():
    return """
    **🤖 Financial Chatbot Help**
    
    Here are some things you can ask me:
    
    - **Price Lookup**: "What's the current price of RELIANCE?"
    - **Recommendations**: "Should I buy INFY?"
    - **Comparisons**: "Compare HDFC and ICICI"
    - **Predictions**: "Predict TATA stock in the next month"
    - **News**: "Latest news about Indian market"
    - **Technical Analysis**: "Show technical chart for WIPRO"
    - **Sentiment**: "What's the sentiment on TCS?"
    
    Just ask any question about Indian stocks or the market!
    """

if not st.session_state.news_thread_started:
    threading.Thread(target=news_fetch_thread, daemon=True).start()
    st.session_state.news_thread_started = True
    news = fetch_news()
    if news:
        st.session_state.recent_news = news
        st.session_state.last_news_fetch = datetime.now()
        update_rag(news)

st.title("AI Financial Assistant")
st.markdown("""
This intelligent chatbot helps you analyze Indian stocks, make comparisons, get price quotes, and receive market insights backed by news and data analysis.
""")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.write(msg["content"])
        elif isinstance(msg["content"], dict) and "chart" in msg["content"]:
            st.write(msg["content"].get("text", ""))
            st.plotly_chart(msg["content"]["chart"], use_container_width=True)
            if "additional_text" in msg["content"]:
                st.write(msg["content"]["additional_text"])

query = st.chat_input("Ask anything about stocks or markets...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            intent = detect_intent(query)
            tickers = extract_tickers(query)
            
            docs = retrieve_docs(query)
            
            if intent == "help":
                response = get_help_message()
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            elif intent == "news":
                if docs:
                    response = "📰 **Latest Financial News**\n\n"
                    for doc in docs[:5]:
                        response += f"- **{doc['title']}**\n  *Source: {doc['source']}*\n\n"
                else:
                    response = "I don't have any recent news articles matching your query. Try asking about specific stocks or broader market topics."
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            elif intent == "comparison" and len(tickers) >= 2:
                unique_tickers = list(set(tickers))[:3]  
                
                results = []
                for ticker in unique_tickers:
                    stock_data = fetch_stock_data(ticker)
                    if not stock_data:
                        continue
                        
                    decision = combined_decision(ticker, docs)
                    info = get_stock_info(ticker)
                    
                    results.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'price': stock_data['price'],
                        'change': stock_data['change'],
                        'score': decision['score'],
                        'recommendation': decision['recommendation']
                    })
                
                if len(results) < 2:
                    response = "⚠️ I couldn't find enough valid stock data for comparison. Please check the ticker symbols and try again."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    text = "📊 **Stock Comparison Analysis**\n\n"
                    
                    for result in results:
                        change_icon = "📈" if result['change'] >= 0 else "📉"
                        score_icon = "🟢" if result['score'] >= 0.6 else ("🔴" if result['score'] <= 0.4 else "🟡")
                        
                        text += f"**{result['ticker']} ({result['name']})**\n"
                        text += f"- Current Price: ₹{result['price']:.2f} {change_icon} {result['change']:.2f}%\n"
                        text += f"- Recommendation: {score_icon} {result['recommendation']} (Score: {result['score']:.2f})\n\n"
                    
                    comparison_chart = create_comparison_chart(list(r['ticker'] for r in results))
                    
                    best_stock = max(results, key=lambda x: x['score'])
                    text += f"🏆 **Conclusion**: Based on comprehensive analysis, **{best_stock['ticker']}** appears to be the better investment choice with a score of {best_stock['score']:.2f}."
                    
                    st.write(text)
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)
                        
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": {
                            "text": text,
                            "chart": comparison_chart
                        }
                    })
                    
            elif intent == "technical" and tickers:
                ticker = tickers[0]  
                
                technical_chart = create_technical_chart(ticker)
                info = get_stock_info(ticker)
                
                if technical_chart:
                    text = f"📈 **Technical Analysis for {ticker} ({info['name']})**\n\n"
                    
                    signal, score = trade_signal(ticker)
                    
                    text += f"**Technical Outlook**: "
                    if signal == "Buy":
                        text += "The technical indicators show a bullish trend. The price is above key moving averages, suggesting upward momentum."
                    elif signal == "Sell":
                        text += "The technical indicators show a bearish trend. The price is below key moving averages, suggesting downward pressure."
                    else:
                        text += "The technical indicators show a neutral trend. The price is moving sideways without clear direction."
                    
                    st.write(text)
                    st.plotly_chart(technical_chart, use_container_width=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": {
                            "text": text,
                            "chart": technical_chart
                        }
                    })
                else:
                    response = f"I couldn't generate a technical chart for {ticker}. Please check if this is a valid ticker symbol."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            elif intent == "prediction" and tickers:
                ticker = tickers[0]  
                
                prediction = stock_prediction(ticker)
                info = get_stock_info(ticker)
                
                if prediction:
                    change_pct = prediction['change_pct']
                    direction = "📈" if change_pct > 0 else "📉"
                    
                    text = f"🔮 **30-Day Price Forecast for {ticker} ({info['name']})**\n\n"
                    text += f"Current Price: ₹{prediction['current_price']:.2f}\n"
                    text += f"Forecast Price: ₹{prediction['predicted_price']:.2f}\n"
                    text += f"Expected Change: {direction} {abs(change_pct):.2f}%\n\n"
                    
                    text += "**Forecast Analysis**: "
                    if change_pct > 5:
                        text += "The model predicts a strong bullish trend in the next 30 days, suggesting a good buying opportunity."
                    elif change_pct > 2:
                        text += "The model predicts a moderate upward movement, which could present a potential entry point."
                    elif change_pct < -5:
                        text += "The model predicts a significant decline, suggesting caution if you're holding this stock."
                    elif change_pct < -2:
                        text += "The model predicts a moderate decline, which might indicate a wait-and-see approach."
                    else:
                        text += "The model predicts relatively flat price action, suggesting a neutral outlook for the next month."
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=prediction['history'].index,
                        y=prediction['history']['Close'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    forecast = prediction['forecast']
                    future_dates = forecast['ds'][len(prediction['history']):]
                    future_values = forecast['yhat'][len(prediction['history']):]
                    future_lower = forecast['yhat_lower'][len(prediction['history']):]
                    future_upper = forecast['yhat_upper'][len(prediction['history']):]
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_values,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates.tolist() + future_dates.tolist()[::-1],
                        y=future_upper.tolist() + future_lower.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(231,107,243,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence'
                    ))
                    
                    fig.update_layout(
                        title=f'Price Forecast for {ticker}',
                        xaxis_title='Date',
                        yaxis_title='Price (₹)' if ticker.endswith('.NS') else 'Price ($)',
                        hovermode='x unified',
                        legend_title='Data',
                        height=500
                    )
                    
                    st.write(text)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("⚠️ Disclaimer: These predictions are based on historical patterns and may not accurately reflect future market movements. Please consult a financial advisor before making investment decisions.")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": {
                            "text": text,
                            "chart": fig,
                            "additional_text": "⚠️ Disclaimer: These predictions are based on historical patterns and may not accurately reflect future market movements."
                        }
                    })
                else:
                    response = f"I couldn't generate a forecast for {ticker}. This might be due to insufficient historical data."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            elif intent == "recommendation" and tickers:
                ticker = tickers[0]  
                
                stock_data = fetch_stock_data(ticker)
                info = get_stock_info(ticker)
                
                if stock_data:
                    decision = combined_decision(ticker, docs)
                    
                    score = decision['score']
                    components = decision['components']
                    
                    text = f"🔍 **Investment Analysis for {ticker} ({info['name']})**\n\n"
                    
                    if 'change' in stock_data:
                        change_icon = "📈" if stock_data['change'] >= 0 else "📉"
                        text += f"Current Price: ₹{stock_data['price']:.2f} {change_icon} {stock_data['change']:.2f}%\n\n"
                    else:
                        text += f"Current Price: ₹{stock_data['price']:.2f}\n\n"
                    
                    rec_icon = "🟢" if score >= 0.6 else ("🔴" if score <= 0.4 else "🟡")
                    text += f"**Recommendation: {rec_icon} {decision['recommendation']} (Score: {score:.2f})**\n\n"
                    
                    text += "**Analysis Components:**\n"
                    text += f"- Technical Indicators: {components['technical']:.2f}\n"
                    text += f"- News Sentiment: {components['sentiment']:.2f}\n"
                    text += f"- Price Prediction: {components['prediction']:.2f}\n"
                    text += f"- Market Conditions: {components['market']:.2f}\n\n"
                    
                    text += "**Analysis Summary:** "
                    if score >= 0.7:
                        text += "Technical indicators, sentiment analysis, and forecasts all align positively. This stock shows strong bullish signals across multiple factors."
                    elif score >= 0.6:
                        text += "Most indicators are positive, suggesting good upside potential. Some caution is warranted, but overall outlook is favorable."
                    elif score <= 0.3:
                        text += "Multiple negative signals detected. Technical indicators and forecasts suggest a downward trend that may continue."
                    elif score <= 0.4:
                        text += "Some concerning signals present. While not strongly negative, the outlook shows more downside risk than upside potential."
                    else:
                        text += "Mixed signals with no clear direction. Some indicators are positive while others are negative, suggesting a wait-and-see approach."
                    
                    if docs:
                        text += "\n\n**Related News:**\n"
                        for doc in docs[:2]:
                            text += f"- {doc['title']} *(Source: {doc['source']})*\n"
                    
                    st.write(text)
                    
                    st.caption("⚠️ Disclaimer: This is an algorithmic recommendation and should not be the sole basis for investment decisions. Please consult with a financial advisor.")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": text + "\n\n⚠️ This is an algorithmic recommendation only."
                    })
                else:
                    response = f"I couldn't find data for {ticker}. Please check if this is a valid ticker symbol."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            elif intent == "price_lookup" and tickers:
                results = []
                
                for ticker in tickers[:5]:  
                    stock_data = fetch_stock_data(ticker)
                    info = get_stock_info(ticker)
                    
                    if stock_data:
                        results.append({
                            'ticker': ticker,
                            'name': info['name'],
                            'price': stock_data['price'],
                            'change': stock_data.get('change', 0),
                            'change_pct': stock_data.get('change', 0),
                            'volume': stock_data.get('volume', "N/A"),
                            'high': stock_data.get('high', stock_data['price']),
                            'low': stock_data.get('low', stock_data['price'])
                        })
                
                if results:
                    text = "💰 **Current Stock Prices**\n\n"
                    
                    for result in results:
                        change_icon = "📈" if result['change'] >= 0 else "📉"
                        name = result['name'] if result['name'] != result['ticker'] else ""
                        
                        if name:
                            text += f"**{result['ticker']}** ({name}):\n"
                        else:
                            text += f"**{result['ticker']}**:\n"
                            
                        text += f"- Price: ₹{result['price']:.2f} {change_icon} {result['change_pct']:.2f}%\n"
                        if result['high'] != "N/A" and result['low'] != "N/A":
                            text += f"- Day Range: ₹{result['low']:.2f} - ₹{result['high']:.2f}\n"
                        text += "\n"
                    
                    current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                    text += f"*Prices as of {current_time}*"
                    
                    st.write(text)
                    st.session_state.messages.append({"role": "assistant", "content": text})
                else:
                    response = "I couldn't find price data for the mentioned stocks. Please check if the ticker symbols are correct."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            elif intent == "sentiment" and tickers:
                ticker = tickers[0]  
                
                ticker_clean = ticker.replace(".NS", "")
                ticker_news = [doc for doc in docs if ticker_clean.lower() in doc['text'].lower()]
                
                if not ticker_news and st.session_state.recent_news:
                    ticker_news = [news for news in st.session_state.recent_news 
                                  if ticker_clean.lower() in news['text'].lower()]
                
                sentiment_score = analyze_sentiment([doc['text'] for doc in ticker_news], ticker) if ticker_news else 0.5
                
                info = get_stock_info(ticker)
                name = info['name'] if info['name'] != ticker else ticker
                
                sentiment_text = "Very Negative" if sentiment_score < 0.2 else (
                                "Negative" if sentiment_score < 0.4 else (
                                "Neutral" if sentiment_score < 0.6 else (
                                "Positive" if sentiment_score < 0.8 else "Very Positive")))
                
                sentiment_icon = "😡" if sentiment_score < 0.2 else (
                                "😟" if sentiment_score < 0.4 else (
                                "😐" if sentiment_score < 0.6 else (
                                "🙂" if sentiment_score < 0.8 else "😄")))
                
                text = f"📊 **Market Sentiment Analysis for {ticker}** ({name})\n\n"
                text += f"**Overall Sentiment: {sentiment_icon} {sentiment_text}** (Score: {sentiment_score:.2f})\n\n"
                
                if ticker_news:
                    text += "**Recent News Affecting Sentiment:**\n"
                    for i, news in enumerate(ticker_news[:3]):
                        text += f"{i+1}. **{news['title']}**\n   *Source: {news['source']}*\n\n"
                else:
                    text += "*No specific news found for this ticker in recent articles.*\n\n"
                
                text += "**Sentiment Interpretation:** "
                if sentiment_score > 0.7:
                    text += "The market has a strongly positive outlook on this stock. Recent news and discussions show enthusiasm and confidence."
                elif sentiment_score > 0.6:
                    text += "There's a generally positive sentiment around this stock, with more positive than negative news."
                elif sentiment_score < 0.3:
                    text += "The market sentiment is quite negative. Recent news and discussions indicate significant concerns."
                elif sentiment_score < 0.4:
                    text += "There's a slightly negative bias in recent coverage and market sentiment for this stock."
                else:
                    text += "The market sentiment is relatively neutral, with mixed signals or limited coverage."
                
                st.write(text)
                st.session_state.messages.append({"role": "assistant", "content": text})
            else:
                # General query - use RAG for financial questions
                if not document_store or not FAISS_AVAILABLE or not SENTENCE_TRANSFORMER_AVAILABLE:
                    response = "Sorry, I don't have financial reference material loaded right now. Try asking about stock prices or market news."
                else:
                    query_embedding = embed_model.encode([query], convert_to_numpy=True)
                    D, I = index.search(query_embedding, 3)  

                    if I[0][0] == -1:
                        response = "I couldn't find relevant information in my financial references for your question."
                    else:
                        retrieved_texts = []
                        for idx in I[0]:
                            if idx == -1 or idx >= len(document_store):
                                continue
                            doc_content = document_store[idx]['content']
                            retrieved_texts.append(doc_content.strip())

                        cleaned_chunks = [chunk.replace("\n", " ").strip() for chunk in retrieved_texts]
                        context_text = "\n\n".join(cleaned_chunks)
                        rag_prompt = f"""You are a financial market assistant AI. 
                        Answer the following question using only the information from the provided financial reference material. 
                        Do not add extra information or assumptions. Be clear, accurate, and concise. Limit your answer to 3-4 sentences.

                        Reference Material:
                        \"\"\"
                        {context_text}
                        \"\"\"

                        Question: {query}

                        Answer:"""

                        if len(rag_prompt.split()) > 1000:
                            rag_prompt = ' '.join(rag_prompt.split()[-1000:]) 

                        llm_response = llm(
                            rag_prompt, 
                            max_new_tokens=150,  
                            do_sample=True, 
                            temperature=0.3
                        )[0]['generated_text']

                        if "Answer:" in llm_response:
                            answer_text = llm_response.split("Answer:")[-1].strip()
                        else:
                            answer_text = llm_response.strip()

                        response = f"📚 **AI-powered Answer using reference material:**\n\n{answer_text}"

                if tickers:
                    ticker = tickers[0]
                    stock_data = fetch_stock_data(ticker)
                    if stock_data:
                        response += f"\n\n💹 Regarding **{ticker}**: Currently trading at ₹{stock_data['price']:.2f}."
                        if 'change' in stock_data:
                            direction = "up" if stock_data['change'] >= 0 else "down"
                            response += f" It's {direction} by {abs(stock_data['change']):.2f}% today."

                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})



if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        welcome_msg = """
        👋 **Welcome to the Financial Assistant!**
        
        I can help you with:
        
        - 📊 Stock price lookups (e.g., "What's the price of Reliance Communications Limited?")
        - 💰 Investment recommendations (e.g., "Should I buy Raymond Limited?")
        - 📈 Stock comparisons (e.g., "Compare IDFC First Bank Limited and ICICI Bank Limited")
        - 🔮 Price predictions (e.g., "Predict Tata Motors Limited stock")
        - 📰 Market news and General finance related questions
        
        Just type your question to get started!
        """
        st.write(welcome_msg)
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

with st.sidebar:
    st.header("📰 Recent Market News")
    
    if st.session_state.recent_news:
        for news in st.session_state.recent_news[:5]:
            st.markdown(f"**{news['title']}**")
            st.caption(f"Source: {news['source']} - {news['date'].split('T')[0]}")
            st.markdown("---")
    else:
        st.write("Loading recent news...")
    
    last_update = st.session_state.last_news_fetch.strftime("%d-%b-%Y %H:%M")
    st.caption(f"Last updated: {last_update}")
    
    st.header("📊 Market Indices")
    
    try:
        nifty = fetch_stock_data("^NSEI")
        sensex = fetch_stock_data("^BSESN")
        
        if nifty:
            nifty_change = nifty.get('change', 0)
            nifty_icon = "📈" if nifty_change >= 0 else "📉"
            st.markdown(f"**NIFTY 50:** {nifty['price']:.2f} {nifty_icon} {nifty_change:.2f}%")
            
        if sensex:
            sensex_change = sensex.get('change', 0)
            sensex_icon = "📈" if sensex_change >= 0 else "📉"
            st.markdown(f"**SENSEX:** {sensex['price']:.2f} {sensex_icon} {sensex_change:.2f}%")
    except:
        st.write("Index data not available")