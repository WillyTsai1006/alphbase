import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import optuna  # 引入自動調參框架
import joblib  # 用於模型保存
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report, precision_score

# 忽略繁雜的警告
warnings.filterwarnings('ignore')
# 設定隨機種子以確保可重現性
np.random.seed(42)
# 資料庫連線
DB_URI = 'postgresql://quant:password@localhost:5432/alphabase'
engine = create_engine(DB_URI)

class QuantDataManager:
    """處理數據加載與清洗"""
    def __init__(self, db_engine):
        self.engine = db_engine

    def load_data(self, symbol):
        print(f"Loading data for {symbol}...")
        # 注意：需要 High/Low 來做真正的 Triple Barrier
        # 假設 features_view 或者 raw table 可以 join 出 high/low
        # 為了演示，直接 join market_data 拿 high/low
        query = f"""
        SELECT 
            f.time, f.close, f.log_return, f.ma_20, f.rsi_14, 
            f.bollinger_upper, f.bollinger_lower,
            m.high, m.low
        FROM features_view f
        JOIN market_data m ON f.time = m.time AND f.symbol = m.symbol
        WHERE f.symbol = '{symbol}'
        ORDER BY f.time ASC
        """
        df = pd.read_sql(query, self.engine)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df

class FactorAnalyzer:
    """
    負責因子有效性檢驗 (IC Test)
    """
    @staticmethod
    def analyze_ic(df, factor_col, forward_days=5):
        """
        計算因子的 Information Coefficient (IC)
        """
        # 計算未來 N 日的收益率作為 Target
        # 注意：這裡使用 shift(-N) 會導致最後 N 天變成 NaN，這是正常的
        future_return = df['close'].shift(-forward_days) / df['close'] - 1
        # 建立一個臨時 DataFrame 來處理去空值，避免影響原始 df
        temp_df = pd.DataFrame({
            'factor': df[factor_col],
            'ret': future_return
        }).dropna()
        # 計算 Spearman 相關係數
        ic, p_value = spearmanr(temp_df['factor'], temp_df['ret'])
        print(f"--- Factor Analysis: {factor_col} ---")
        print(f"IC Score (Rank Correlation): {ic:.4f}")
        print(f"P-Value: {p_value:.4e}")
        if abs(ic) > 0.02:
            print(">> 評價: 有效因子 (具有預測力)")
        else:
            print(">> 評價: 弱因子 (可能是雜訊)")
        print("-" * 30)
        return ic

class LabelingEngine:
    """
    實作標註法: Triple Barrier Method (TBM)
    考慮路徑依賴 (Path Dependency)
    """
    @staticmethod
    def get_daily_vol(close, span0=100):
        """計算動態波動率 (用於設定動態止盈止損)"""
        # 使用指數加權移動標準差
        # 這裡過濾 index 確保單調性，避免報錯
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        # 計算回報率的 EWM Std
        return close.pct_change().ewm(span=span0).std()

    @staticmethod
    def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
        """
        核心邏輯：檢查 High/Low 是否觸及 Stop Loss (SL) 或 Profit Taking (PT)
        """
        out = pd.DataFrame(index=molecule)
        # 轉換為 numpy array 以加速運算並避免索引問題
        out['stop_time'] = pd.NaT
        out['side'] = 0
        for dt, target in events.loc[molecule].iterrows():
            # [Fix] 如果目標波動率是 NaN，直接跳過（無法設定邊界）
            if pd.isna(target['trgt']):
                continue
            df0 = close[dt:target['t1']] # 路徑區間
            # [Fix] 如果區間內沒有數據（例如剛好是最後一天），跳過
            if df0.empty:
                continue
            # 計算路徑上的收益率
            path_returns = (df0 / close[dt]) - 1
            # 上邊界 (Profit Taking)
            thresh_upper = float(target['trgt'] * pt_sl[0])
            # 下邊界 (Stop Loss)
            thresh_lower = float(-target['trgt'] * pt_sl[1])
            # 尋找觸發時間
            # 使用 values 轉為 numpy array 進行比較，避免 Index 導致的 TypeError
            upper_hits = path_returns[path_returns > thresh_upper].index
            lower_hits = path_returns[path_returns < thresh_lower].index
            first_touch_pt = upper_hits.min() if not upper_hits.empty else pd.NaT
            first_touch_sl = lower_hits.min() if not lower_hits.empty else pd.NaT
            out.loc[dt, 'stop_time'] = target['t1'] # 預設為時間到期
            if pd.isnull(first_touch_pt) and pd.isnull(first_touch_sl):
                # 既沒止盈也沒止損
                out.loc[dt, 'side'] = 0
            elif pd.isnull(first_touch_sl):
                out.loc[dt, 'stop_time'] = first_touch_pt
                out.loc[dt, 'side'] = 1 # 止盈
            elif pd.isnull(first_touch_pt):
                out.loc[dt, 'stop_time'] = first_touch_sl
                out.loc[dt, 'side'] = -1 # 止損
            else:
                # 兩者都觸發，看誰先發生
                if first_touch_pt < first_touch_sl:
                    out.loc[dt, 'stop_time'] = first_touch_pt
                    out.loc[dt, 'side'] = 1
                else:
                    out.loc[dt, 'stop_time'] = first_touch_sl
                    out.loc[dt, 'side'] = -1
        return out

    def create_labels(self, df, horizon_days=5, pt_sl=[1, 1]):
        """
        主函數
        """
        print("Calculating Dynamic Volatility...")
        # 1. 計算波動率閾值
        df['volatility'] = self.get_daily_vol(df['close'])
        # 2. 定義垂直邊界 (時間到期)
        t1 = df.index + pd.Timedelta(days=horizon_days)
        # 建立 events DataFrame
        events = pd.DataFrame(index=df.index)
        events['t1'] = t1
        events['trgt'] = df['volatility']
        # [Fix] ：移除所有波動率為 NaN 的行 (通常是前 20-100 筆數據)
        # 這避免了後續比較時出現 NaTType/Float 錯誤
        events = events.dropna(subset=['trgt'])
        # 過濾掉那些 t1 超出數據範圍的 (最後幾天)
        events = events[events['t1'] <= df.index[-1]]
        print(f"Applying Triple Barrier Method on {len(events)} events...")
        # 3. 執行路徑依賴檢查
        if events.empty:
            print("Warning: No valid events found. Check your data length.")
            return df
        labels = self.apply_pt_sl_on_t1(
            df['close'], 
            events, 
            pt_sl=pt_sl, 
            molecule=events.index
        )
        # 4. 生成最終 Label
        df['ret'] = labels['side']
        df['target'] = (df['ret'] == 1).astype(int)
        # 統計
        print(f"Label Distribution:\n{df['target'].value_counts(normalize=True)}")
        # 同樣需要移除掉那些算不出 Label 的行
        return df.dropna(subset=['target', 'volatility'])

class ModelTrainer:
    """
    包含 Purged CV 和 Optuna 調參的訓練器
    """
    def __init__(self, df, feature_cols, target_col='target'):
        self.df = df
        self.features = feature_cols
        self.target = target_col
        self.X = df[feature_cols]
        self.y = df[target_col]

    def objective(self, trial):
        """Optuna 的優化目標函數"""
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'is_unbalance': True
        }
        # 使用 TimeSeriesSplit 進行驗證 (避免 Look-ahead bias)
        # 注意：嚴格來說還需要加 Embargo (間隔)，這裡用標準 TSS 簡化
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_index, val_index in tscv.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            gbm = lgb.train(
                param, 
                dtrain, 
                valid_sets=[dval], 
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            preds = gbm.predict(X_val)
            auc = roc_auc_score(y_val, preds)
            scores.append(auc)
        return np.mean(scores)

    def run_optimization(self, n_trials=20):
        print("Starting Hyperparameter Optimization with Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        return trial.params

    def train_final_model(self, best_params):
        print("\nTraining Final Model with Best Parameters...")
        # 最後使用前 80% 訓練, 後 20% 測試 (OOS Testing)
        split_point = int(len(self.df) * 0.8)
        X_train, X_test = self.X.iloc[:split_point], self.X.iloc[split_point:]
        y_train, y_test = self.y.iloc[:split_point], self.y.iloc[split_point:]
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
        best_params['is_unbalance'] = True
        model = lgb.train(
            best_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtest],
            callbacks=[
                lgb.log_evaluation(50),
                lgb.early_stopping(50)
            ]
        )
        # 評估
        y_prob = model.predict(X_test)
        y_pred = [1 if x > 0.5 else 0 for x in y_prob]
        print("\n--- Final Out-of-Sample Report ---")
        print(classification_report(y_test, y_pred))
        print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        # 特徵重要性 (Gain)
        lgb.plot_importance(model, importance_type='gain', max_num_features=10, title='Feature Importance (Gain)')
        plt.show()
        # 保存模型
        joblib.dump(model, 'alphabase_lgbm.pkl')
        print("Model saved to alphabase_lgbm.pkl")
        return model, X_test, y_test

# Main Execution Pipeline
if __name__ == "__main__":
    ticker = 'AAPL'
    # 1. 初始化
    dm = QuantDataManager(engine)
    df = dm.load_data(ticker)
    # 2. 高級標註 (Path Dependent Triple Barrier)
    # 設定：如果漲幅超過 1倍波動率 (止盈) 或 跌幅超過 1倍波動率 (止損)
    labeler = LabelingEngine()
    df_labeled = labeler.create_labels(df, horizon_days=5, pt_sl=[1.5, 1.0])
    # 3. 因子IC檢驗 (可選)
    print("\nRunning Factor Analysis...")
    analyzer = FactorAnalyzer()
    analyzer.analyze_ic(df_labeled, 'rsi_14') 
    analyzer.analyze_ic(df_labeled, 'ma_20')
    # 4. 定義特徵
    features = ['rsi_14', 'bollinger_upper', 'bollinger_lower', 'log_return', 'ma_20']
    # 5. 模型訓練 (Optuna -> Train)
    trainer = ModelTrainer(df_labeled, features)
    # 執行貝葉斯優化
    best_params = trainer.run_optimization(n_trials=20) # 實際專案建議設為 50+
    # 訓練最終模型
    model, X_test, y_test = trainer.train_final_model(best_params)