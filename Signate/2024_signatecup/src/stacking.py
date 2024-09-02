import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle
import src.util as utils  # utilsモジュールをインポート（必要に応じて使用）

np.random.seed(42)  # 再現性のためのランダムシード設定

class Stacking:
    """
    モデルのスタッキングを行うためのクラス。
    
    Attributes:
    ----------
    train_func : function
        モデルの訓練を行う関数。
    pred_func : function
        モデルの予測を行う関数。
    N_SPLITS : int
        交差検証の分割数。
    NUM_CLASSES : int
        クラスの数（分類タスクに使用）。
    random_seed : int
        乱数シード値（再現性のため）。
    isSave : bool
        学習したモデルを保存するかどうかのフラグ。
    save_dir : str
        学習モデルの保存先ディレクトリ。
    model_name : str
        保存するモデルの名前。
    """

    def __init__(self,
                 train_func,
                 pred_func,
                 N_SPLITS=5,
                 NUM_CLASSES=2,
                 isSave=True,
                 save_dir="model",
                 model_name="",
                 random_seed=0):
        """
        初期化メソッド。スタッキングに必要な設定を行う。

        Parameters:
        ----------
        train_func : function
            モデルの訓練を行う関数。
        pred_func : function
            モデルの予測を行う関数。
        N_SPLITS : int, optional
            交差検証の分割数（デフォルトは5）。
        NUM_CLASSES : int, optional
            クラスの数（デフォルトは2）。
        isSave : bool, optional
            学習したモデルを保存するかどうかのフラグ（デフォルトはTrue）。
        save_dir : str, optional
            学習モデルの保存先ディレクトリ（デフォルトは"model"）。
        model_name : str, optional
            保存するモデルの名前（デフォルトは空文字列）。
        random_seed : int, optional
            乱数シード値（デフォルトは0）。
        """
        self.train_func = train_func
        self.pred_func = pred_func
        self.N_SPLITS = N_SPLITS
        self.NUM_CLASSES = NUM_CLASSES
        self.random_seed = random_seed
        self.isSave = isSave
        
        if self.isSave:
            self.save_dir = save_dir
            self.model_name = model_name
            model_path = os.path.join(self.save_dir, self.model_name)
            print(model_path)
            self.create_directory_if_not_exists(model_path)

    def create_directory_if_not_exists(self, path):
        """
        指定したパスにディレクトリが存在しない場合、新たに作成する。

        Parameters:
        ----------
        path : str
            作成するディレクトリのパス。
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def classification(self, X_train, y_train, X_test):
        """
        分類タスクのためのスタッキングを実行する。

        Parameters:
        ----------
        X_train : pandas.DataFrame
            学習用の特徴量データ。
        y_train : pandas.Series
            学習用のターゲットデータ。
        X_test : pandas.DataFrame
            テスト用の特徴量データ。

        Returns:
        -------
        tuple
            検証データに対するアウトオブフォールドの予測結果とテストデータの平均予測結果。
        """
        columns = X_train.columns.tolist()  # 特徴量の列名リストを取得

        # 結果を格納するための配列を初期化
        oof_valid = np.zeros(X_train.shape[0])  # アウトオブフォールド用の配列
        oof_test = np.zeros(X_test.shape[0])  # テストデータ用の配列
        oof_test_skf = np.zeros((self.N_SPLITS, X_test.shape[0]))  # 各分割のテストデータ予測結果を格納する配列

        # Stratified K-Foldを使用してデータセットを分割
        skf = StratifiedKFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.random_seed)
        for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
            print(f'[CV] {i+1}/{self.N_SPLITS}')  # 現在の交差検証の分割数を表示

            # 学習用データと検証用データに分割
            X_train_, X_valid_ = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_, y_valid_ = y_train.iloc[train_index], y_train.iloc[valid_index]

            # モデルの訓練
            model = self.train_func(X_train_, X_valid_, y_train_, y_valid_)
            
            # モデルを保存
            if self.isSave:
                model_path = os.path.join(self.save_dir, self.model_name, f"{i}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            # 検証データに対する予測
            oof_valid[valid_index] = self.pred_func(X_valid_)
            # テストデータに対する予測
            oof_test_skf[i, :] = self.pred_func(X_test)

            # 現在の分割のAUCスコアを計算
            score = roc_auc_score(y_valid_, oof_valid[valid_index])
            print(f'Fold {i+1} AUC: {score}')

        # テストデータに対する最終的な平均予測を計算
        oof_test[:] = oof_test_skf.mean(axis=0)
        # 全体のAUCスコアを表示
        print(roc_auc_score(y_train, oof_valid))
        return oof_valid, oof_test
