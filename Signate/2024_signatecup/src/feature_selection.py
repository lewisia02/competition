import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

def feature_selection_corr(df, threshold=0.9):
    """
    高い相関を持つ特徴量を除外する関数。

    Parameters:
    ----------
    df : pandas.DataFrame
        特徴量を含むデータフレーム。
    threshold : float, optional
        除外する相関のしきい値（デフォルトは0.9）。

    Returns:
    -------
    pandas.DataFrame
        相関が高い特徴量を除外した新しいデータフレーム。
    """
    # 数値型のカラムのみを選択
    numerical_df = df.select_dtypes(include=[np.number])
    print(f"Numerical features shape: {numerical_df.shape}")  # 数値型特徴量の数を表示
    
    # 相関行列の計算
    corr_matrix = numerical_df.corr()
    
    # 相関行列から高い相関を持つ特徴量を除外
    columns_to_drop = set()  # 除外対象の特徴量を格納するセット

    # 相関行列をループして、高い相関の特徴量を見つける
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # しきい値を超える相関係数を持つ特徴量を除外リストに追加
                columns_to_drop.add(corr_matrix.columns[j])
    
    # 高い相関を持つカラムを除外したデータフレームを作成
    reduced_df = numerical_df.drop(columns=columns_to_drop)
    
    return reduced_df  # 除外後のデータフレームを返す

def permutation_importance(model, X_train, y_train):
    """
    モデルの特徴量の重要度を計算するためのパーミュテーション重要度を使用する関数。

    Parameters:
    ----------
    model : scikit-learn estimator
        学習済みのモデル。
    X_train : pandas.DataFrame
        学習用の特徴量データ。
    y_train : pandas.Series
        学習用のターゲットデータ。

    Returns:
    -------
    pandas.DataFrame
        特徴量のパーミュテーション重要度の結果を含むデータフレーム。
    """
    # AUCスコアを使用したカスタムスコア関数を定義
    def auc_score(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
    
    # パーミュテーション重要度のためのスコア関数を作成
    scorer = make_scorer(auc_score)

    # パーミュテーション重要度を計算
    result = permutation_importance(model, X_train, y_train, scoring=scorer, n_repeats=10, n_jobs=-1, random_state=71)
    
    # 結果をデータフレームに変換
    perm_imp_df = pd.DataFrame({"importances_mean": result["importances_mean"], "importances_std": result["importances_std"]}, index=X_train.columns)
    
    # 重要度のプロットを作成
    perm_imp_df.sort_values("importances_mean", ascending=False).importances_mean.plot.barh()
    plt.xlabel('Importance Mean')
    plt.title('Permutation Importance')
    plt.show()

    return perm_imp_df  # パーミュテーション重要度のデータフレームを返す
