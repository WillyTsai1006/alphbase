# 構建 ETL Pipeline
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
import time

# 1. 資料庫連線設定
# 格式: postgresql://user:password@localhost:port/dbname
DB_URI = 'postgresql://quant:password@localhost:5432/alphabase'
engine = create_engine(DB_URI)

def init_symbols(ticker_list):
    """初始化股票清單 (寫入 symbols 表)"""
    # 先檢查資料庫中是否已存在這些股票代號
    existing_symbols = []
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT symbol FROM symbols"))
            existing_symbols = [row[0] for row in result]
    except Exception as e:
        print(f"Error checking existing symbols: {e}")
    
    # 只新增不存在的股票
    new_symbols = [ticker for ticker in ticker_list if ticker not in existing_symbols]
    
    if new_symbols:
        data = []
        for ticker in new_symbols:
            data.append({
                'symbol': ticker,
                'asset_type': 'Stock', 
                'is_active': True
            })
        df = pd.DataFrame(data)
        try:
            df.to_sql('symbols', engine, if_exists='append', index=False)
            print(f"Added new symbols: {new_symbols}")
        except Exception as e:
            print(f"Error adding symbols: {e}")
    else:
        print("All symbols already exist in database")

def fetch_and_store_data(symbol, start_date="2020-01-01"):
    """
    從 yfinance 下載數據並透過 pandas to_sql 寫入資料庫
    """
    print(f"Downloading data for {symbol}...")
    # 方法1：使用 yf.Ticker 避免多級列索引問題
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(start=start_date)
        if df.empty:
            print(f"No data found for {symbol}")
            return
        # 檢查是否有多級列索引，如果有則扁平化
        if isinstance(df.columns, pd.MultiIndex):
            # 扁平化多級列索引
            df.columns = df.columns.get_level_values(0)
        # 重置索引
        df.reset_index(inplace=True)
        # 確保我們有正確的列名
        print(f"Columns after download: {list(df.columns)}")
        # 標準化列名
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower:
                column_mapping[col] = 'time'
            elif 'open' in col_lower:
                column_mapping[col] = 'open'
            elif 'high' in col_lower:
                column_mapping[col] = 'high'
            elif 'low' in col_lower:
                column_mapping[col] = 'low'
            elif 'close' in col_lower:
                column_mapping[col] = 'close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'volume'
        df.rename(columns=column_mapping, inplace=True)
        # 添加股票代號欄位
        df['symbol'] = symbol
        # 只保留我們需要的欄位
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        available_columns = [col for col in required_columns if col in df.columns]
        if 'time' not in df.columns:
            # 如果沒有time欄位，檢查是否有Date或index
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'time'}, inplace=True)
                available_columns.append('time')
            elif df.index.name == 'Date':
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'time'}, inplace=True)
                available_columns.append('time')
        df = df[available_columns]
        # 顯示前幾行數據以進行調試
        print(f"First few rows of data for {symbol}:")
        print(df.head())
        # 3. 寫入資料庫 (Bulk Insert)
        # 首先檢查表格是否存在，如果不存在則創建
        try:
            df.to_sql('market_data', engine, if_exists='append', index=False, chunksize=1000)
            print(f"Stored {len(df)} rows for {symbol}.")
        except Exception as e:
            print(f"Error storing data for {symbol}: {e}")
            # 顯示更多錯誤細節
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame dtypes: {df.dtypes}")
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")

# 替代方案：更簡單的方法，直接處理多級列索引
def fetch_and_store_data_simple(symbol, start_date="2020-01-01"):
    """
    簡單版本：直接處理 yf.download 返回的多級列索引
    """
    print(f"Downloading data for {symbol} (simple method)...")
    try:
        # 使用 yf.download 但只下載一個股票
        df = yf.download(symbol, start=start_date, progress=False)
        if df.empty:
            print(f"No data found for {symbol}")
            return
        # 重置索引
        df.reset_index(inplace=True)
        # 如果有多級列索引，扁平化它
        if isinstance(df.columns, pd.MultiIndex):
            # 獲取第一層列名（技術指標名稱）
            df.columns = df.columns.get_level_values(0)
        # 現在應該有單級列名：Date, Open, High, Low, Close, Volume
        print(f"Columns after processing: {list(df.columns)}")
        # 重命名列以匹配數據庫
        rename_dict = {
            'Date': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        # 只重命名存在的列
        rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
        df.rename(columns=rename_dict, inplace=True)
        # 添加股票代號欄位
        df['symbol'] = symbol
        # 確保列順序
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        existing_cols = [col for col in required_cols if col in df.columns]
        df = df[existing_cols]
        # 寫入數據庫
        df.to_sql('market_data', engine, if_exists='append', index=False, chunksize=1000)
        print(f"Stored {len(df)} rows for {symbol}.")
    except Exception as e:
        print(f"Error in simple method for {symbol}: {e}")

if __name__ == "__main__":
    # 定義你想抓取的股票 (建議選不同板塊以利後續多因子分析)
    target_tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'JPM', 'XOM']
    # 初始化股票清單
    init_symbols(target_tickers)
    # 使用簡單方法下載數據
    for ticker in target_tickers:
        fetch_and_store_data_simple(ticker)
        time.sleep(1)  # 避免觸發 API rate limit

# Docker 容器化部署：標準化開發環境。
# TimescaleDB/PostgreSQL：針對金融時間序列數據優化的資料庫選型。
# Schema Design：正確使用 Primary Key, Foreign Key, Indexing 和 Data Types (Numeric vs Float)。
# ETL process (Extract, Transform, Load)：使用 Python 自動化數據管道。