import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# --- 設定 (ここを編集してください) ---
CSV_FILE_PATH = 'hidden_training1.csv'  # お手元のCSVファイルへのパス
DATE_COLUMN = '年月'             # 日付が格納されている列名
VALUE_COLUMN = '当日の健診数'           # 分析対象の数値が格納されている列名

# Matplotlibで日本語を表示するための設定（ご自身の環境に合わせてフォント名を変更してください）
# plt.rcParams['font.family'] = 'IPAexGothic'
# --- 設定ここまで ---

# --- 1. データの読み込みと前処理 ---
print("--- 1. データの読み込みと前処理 ---")
df = pd.read_csv(CSV_FILE_PATH)

# 日付のカラムをdatetime型に変換し、インデックスに設定
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.set_index(DATE_COLUMN)

# 欠損値(NaN)を0で補完
df[VALUE_COLUMN] = df[VALUE_COLUMN].fillna(0)

print("\nデータの基本情報を表示します。")
df.info()
print("\n先頭5行のデータ:")
print(df.head())

# --- 2. データを訓練データとテストデータに分割 ---
print("\n--- 2. データを訓練データとテストデータに分割します ---")
split_point = 1098 # 指定された分割点 (約3年分のデータ)

if len(df) <= split_point:
    print(f"エラー: データ件数が{len(df)}件しかなく、{split_point}件で分割できません。")
    print("テストデータを作成するには、より多くのデータが必要です。")
else:
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    print(f"訓練データ: {train_df.index.min().date()} から {train_df.index.max().date()} まで ({len(train_df)}件)")
    print(f"テストデータ: {test_df.index.min().date()} から {test_df.index.max().date()} まで ({len(test_df)}件)")

    # --- 3. ARIMAモデルによる予測 ---
    print("\n--- 3. ARIMAモデルによる予測 ---")
    # ARIMAモデルの次数 (p, d, q) を設定
    # p: 自己回帰 (AR) の次数
    # d: 差分 (I) の次数 (データの非定常性を除去するために必要な差分の回数)
    # q: 移動平均 (MA) の次数
    # これらの次数は、ACF/PACFプロットや情報量基準（AIC/BIC）を用いて決定するのが一般的です。
    # ここでは一例として (5, 1, 0) を使用します。
    arima_order = (5, 1, 0)
    print(f"ARIMAモデルの次数: {arima_order}")

    try:
        # ARIMAモデルの学習
        arima_model = ARIMA(train_df[VALUE_COLUMN], order=arima_order)
        arima_model_fit = arima_model.fit()
        print("ARIMAモデルの学習が完了しました。")
        # print(arima_model_fit.summary()) # モデルのサマリーを表示したい場合

        # テストデータ期間の予測
        arima_predictions = arima_model_fit.predict(
            start=len(train_df),
            end=len(df) - 1
        )

        # 予測結果の評価
        mae_arima = mean_absolute_error(test_df[VALUE_COLUMN], arima_predictions)
        rmse_arima = np.sqrt(mean_squared_error(test_df[VALUE_COLUMN], arima_predictions))
        mape_arima = mean_absolute_percentage_error(test_df[VALUE_COLUMN], arima_predictions)

        print("\n【ARIMAモデルの予測精度評価】")
        print(f"  - MAE: {mae_arima:.2f}")
        print(f"  - RMSE: {rmse_arima:.2f}")
        print(f"  - MAPE: {mape_arima:.2%}")

        # 予測結果の可視化
        plt.figure(figsize=(15, 6))
        plt.plot(train_df.index, train_df[VALUE_COLUMN], label='訓練データ (実績)', color='blue')
        plt.plot(test_df.index, test_df[VALUE_COLUMN], label='テストデータ (実績)', color='green')
        plt.plot(test_df.index, arima_predictions, label='ARIMA予測', color='red', linestyle='--')
        plt.title(f'ARIMAモデルによる{VALUE_COLUMN}の予測', fontsize=16)
        plt.xlabel('日付', fontsize=12)
        plt.ylabel(VALUE_COLUMN, fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"ARIMAモデルの学習または予測中にエラーが発生しました: {e}")
        print("ARIMAモデルの次数 (p, d, q) を調整してみてください。")

    # --- 4. SARIMAモデルによる予測 ---
    print("\n--- 4. SARIMAモデルによる予測 ---")
    # SARIMAモデルの次数 (p, d, q) と季節次数 (P, D, Q, S) を設定
    # S: 季節周期 (例: 日次データで週次パターンなら7, 年次パターンなら365)
    # ここでは週次パターンを考慮し、S=7 とします。
    sarima_order = (1, 1, 1)
    sarima_seasonal_order = (1, 1, 0, 7) # (P, D, Q, S)
    print(f"SARIMAモデルの次数: {sarima_order}")
    print(f"SARIMAモデルの季節次数: {sarima_seasonal_order}")

    try:
        # SARIMAモデルの学習
        sarima_model = SARIMAX(
            train_df[VALUE_COLUMN],
            order=sarima_order,
            seasonal_order=sarima_seasonal_order,
            enforce_stationarity=False, # 定常性を強制しない
            enforce_invertibility=False # 可逆性を強制しない
        )
        sarima_model_fit = sarima_model.fit(disp=False) # disp=Falseで学習中の詳細出力を抑制
        print("SARIMAモデルの学習が完了しました。")
        # print(sarima_model_fit.summary()) # モデルのサマリーを表示したい場合

        # テストデータ期間の予測
        sarima_predictions = sarima_model_fit.predict(
            start=len(train_df),
            end=len(df) - 1
        )

        # 予測結果の評価
        mae_sarima = mean_absolute_error(test_df[VALUE_COLUMN], sarima_predictions)
        rmse_sarima = np.sqrt(mean_squared_error(test_df[VALUE_COLUMN], sarima_predictions))
        mape_sarima = mean_absolute_percentage_error(test_df[VALUE_COLUMN], sarima_predictions)

        print("\n【SARIMAモデルの予測精度評価】")
        print(f"  - MAE: {mae_sarima:.2f}")
        print(f"  - RMSE: {rmse_sarima:.2f}")
        print(f"  - MAPE: {mape_sarima:.2%}")

        # 予測結果の可視化
        plt.figure(figsize=(15, 6))
        plt.plot(train_df.index, train_df[VALUE_COLUMN], label='訓練データ (実績)', color='blue')
        plt.plot(test_df.index, test_df[VALUE_COLUMN], label='テストデータ (実績)', color='green')
        plt.plot(test_df.index, sarima_predictions, label='SARIMA予測', color='purple', linestyle='--')
        plt.title(f'SARIMAモデルによる{VALUE_COLUMN}の予測', fontsize=16)
        plt.xlabel('日付', fontsize=12)
        plt.ylabel(VALUE_COLUMN, fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"SARIMAモデルの学習または予測中にエラーが発生しました: {e}")
        print("SARIMAモデルの次数 (p, d, q) や季節次数 (P, D, Q, S) を調整してみてください。")