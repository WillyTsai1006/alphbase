import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

class ProBacktester:
    def __init__(self, data, model, feature_cols, initial_capital=100000, transaction_cost=0.001):
        """
        :param data: 包含 OHLCV 的完整數據 (必須包含 High, Low, Open)
        :param model: 訓練好的 LightGBM 模型
        :param feature_cols: 模型需要的特徵欄位
        """
        self.data = data.copy()
        self.model = model
        self.features = feature_cols
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.trade_log = []
        self.equity_curve = []
        
    def generate_signals(self, threshold=0.55):
        """
        使用模型生成預測信號
        """
        print("Generating Alpha Signals...")
        X = self.data[self.features]
        # 獲取預測機率
        self.data['prob'] = self.model.predict(X)
        # 生成信號: 1 = Long, 0 = Neutral (這裡暫不做空)
        self.data['signal'] = (self.data['prob'] > threshold).astype(int)
        return self.data

    def run_backtest(self, horizon_days=5, sl_mult=1.0, tp_mult=1.5):
        """
        執行事件驅動回測 (Event-Driven Loop)
        這比向量化慢，但能準確模擬 Triple Barrier 的路徑依賴
        """
        print("Running Event-Driven Simulation...")
        cash = self.initial_capital
        position = 0 # 持倉數量
        entry_price = 0
        entry_date = None
        stop_loss_price = 0
        take_profit_price = 0
        days_held = 0
        # 為了計算動態波動率 (用於 SL/TP)
        self.data['volatility'] = self.data['close'].pct_change().ewm(span=20).std()
        # 使用 itertuples 加速遍歷
        # row 包含: Index(Time), open, high, low, close, volatility, signal...
        for row in self.data.itertuples():
            # --- 1. 更新資產淨值 (Mark to Market) ---
            current_value = cash + (position * row.close)
            self.equity_curve.append({'time': row.Index, 'equity': current_value})
            # --- 2. 持倉管理 (檢查是否觸發出場) ---
            if position > 0:
                days_held += 1
                exit_price = None
                exit_reason = ""
                # A. 檢查止損 (Stop Loss) - 檢查 Low 是否穿過 SL
                if row.low <= stop_loss_price:
                    exit_price = stop_loss_price # 假設在 SL 價格成交 (實際可能有滑價)
                    exit_reason = "Stop Loss"
                # B. 檢查止盈 (Take Profit) - 檢查 High 是否穿過 TP
                elif row.high >= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = "Take Profit"
                # C. 檢查時間到期 (Time Barrier)
                elif days_held >= horizon_days:
                    exit_price = row.close
                    exit_reason = "Time Exit"
                # 執行平倉
                if exit_price:
                    # 扣除交易成本
                    commission = (position * exit_price) * self.tc
                    revenue = (position * exit_price) - commission
                    cash += revenue                    
                    # 紀錄交易
                    pnl = (exit_price - entry_price) / entry_price
                    self.trade_log.append({
                        'entry_date': entry_date,
                        'exit_date': row.Index,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'reason': exit_reason,
                        'holding_days': days_held
                    })                    
                    # 重置狀態
                    position = 0
                    days_held = 0
                    continue # 平倉當天不開新倉
            # --- 3. 進場管理 (Entry Logic) ---
            # 只有空手時才開倉 (簡單起見，不加碼)
            if position == 0 and row.signal == 1:
                # 為了避免 Look-ahead bias，我們在「收盤後」看到信號
                # 理論上要在「隔天開盤」買入
                # 為了簡化 Event Loop，假設以「當天收盤價」買入 (這是一個強假設)
                # 更嚴謹的做法是：Signal 產生在 t，而在 t+1 的 Open 買入
                # 修正：檢查是否還有下一筆數據可用 (避免最後一天報錯)
                # 由於 itertuples 很難拿下一行，這裡妥協：
                # 假設我們能在收盤前一刻買入 (MOC - Market On Close)
                price = row.close 
                vol = row.volatility
                # 計算可買股數 (全倉 All-in 模式，實際建議用 Risk Parity)
                # 預留 1% 現金避免誤差
                shares_to_buy = int((cash * 0.99) / price)
                if shares_to_buy > 0:
                    commission = (shares_to_buy * price) * self.tc
                    cost = (shares_to_buy * price) + commission
                    cash -= cost
                    position = shares_to_buy
                    entry_price = price
                    entry_date = row.Index
                    # 設定動態止盈止損 (根據 TBM 邏輯)
                    stop_loss_price = price * (1 - vol * sl_mult)
                    take_profit_price = price * (1 + vol * tp_mult)
                    days_held = 0
        # 回測結束，強制平倉
        if position > 0:
            cash += position * self.data.iloc[-1]['close']
        # 整理結果
        self.equity_df = pd.DataFrame(self.equity_curve).set_index('time')
        self.trades_df = pd.DataFrame(self.trade_log)
        return self.equity_df, self.trades_df

    def analyze_performance(self):
        """生成專業績效報告"""
        if self.equity_df.empty:
            print("No trades executed.")
            return
        # 計算指標
        final_equity = self.equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        # 夏普值
        returns = self.equity_df['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        # 最大回撤
        cum_max = self.equity_df['equity'].cummax()
        drawdown = (self.equity_df['equity'] - cum_max) / cum_max
        max_dd = drawdown.min()
        # 勝率與盈虧比
        if not self.trades_df.empty:
            win_rate = len(self.trades_df[self.trades_df['pnl'] > 0]) / len(self.trades_df)
            avg_win = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean())
            profit_factor = avg_win / avg_loss if avg_loss != 0 else np.inf
        else:
            win_rate = 0
            profit_factor = 0
        print("="*40)
        print("   AlphaBase Backtest Report")
        print("="*40)
        print(f"Initial Capital:   ${self.initial_capital:,.0f}")
        print(f"Final Equity:      ${final_equity:,.2f}")
        print(f"Total Return:      {total_return:.2%}")
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print(f"Max Drawdown:      {max_dd:.2%}")
        print("-" * 40)
        print(f"Total Trades:      {len(self.trades_df)}")
        print(f"Win Rate:          {win_rate:.2%}")
        print(f"Profit Factor:     {profit_factor:.2f}")
        # 顯示出場原因分佈
        if not self.trades_df.empty:
            print("-" * 40)
            print("Exit Reasons:")
            print(self.trades_df['reason'].value_counts())
        print("="*40)
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(12, 8))
        # 資金曲線
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.equity_df.index, self.equity_df['equity'], label='Strategy Equity')
        ax1.set_title('Equity Curve')
        ax1.grid(True)
        ax1.legend()
        # 標註買賣點 (選做)
        #if not self.trades_df.empty:
        #    buys = self.trades_df.set_index('entry_date')
        #    sells = self.trades_df.set_index('exit_date')
        #    ax1.scatter(buys.index, buys['entry_price'] * (self.equity_df.loc[buys.index]['equity'] / self.data.loc[buys.index]['close']), marker='^', color='green', alpha=0.5)
        # 回撤圖
        ax2 = plt.subplot(2, 1, 2)
        cum_max = self.equity_df['equity'].cummax()
        drawdown = (self.equity_df['equity'] - cum_max) / cum_max
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

# 整合測試
if __name__ == "__main__":
    # 假設已經跑完了 quant_engine_pro.py 並拿到了 model 和 df
    # 這裡模擬一個簡單的環境
    # 1. 重新載入數據 (必須包含 OHLC)
    from quant_engine import QuantDataManager, engine, ModelTrainer
    import joblib
    ticker = 'AAPL'
    dm = QuantDataManager(engine)
    df = dm.load_data(ticker)
    # 2. 載入訓練好的模型
    # model = joblib.load('alphabase_lgbm.pkl')
    # 為了演示，這裡我們假設 model 已經在記憶體中 (來自上一個腳本)
    # 如果是獨立運行，請取消註解上面的 joblib.load
    # 模擬: 如果沒有 model，這裡會報錯，所以請確保先執行 quant_engine_pro.py
    # 這裡假設 features 是之前定義好的
    features = ['rsi_14', 'bollinger_upper', 'bollinger_lower', 'log_return', 'ma_20']
    # 注意: 確保 df 包含 features 列
    try:
        model = joblib.load('alphabase_lgbm.pkl')
        print("Model loaded successfully.")
        # 3. 初始化回測器
        bt = ProBacktester(df, model, features)
        # 4. 生成信號
        bt.generate_signals(threshold=0.55)
        # 5. 執行回測 
        # sl_mult=1.0, tp_mult=1.5 意思是: 
        # 止損 = 進場價 * (1 - 1.0 * 當前波動率)
        # 止盈 = 進場價 * (1 + 1.5 * 當前波動率)
        bt.run_backtest(horizon_days=5, sl_mult=1.0, tp_mult=1.5)
        # 6. 分析結果
        bt.analyze_performance()
    except FileNotFoundError:
        print("Error: 'alphabase_lgbm.pkl' not found. Please run quant_engine_pro.py first.")