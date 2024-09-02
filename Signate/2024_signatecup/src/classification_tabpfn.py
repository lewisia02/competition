import numpy as np
from tabpfn import TabPFNClassifier
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

class ClassificationModels:
    """
    タブPFN分類器を用いた分類モデルの学習を行うクラス。

    Attributes:
    ----------
    device : str
        使用するデバイス（CPUまたはCUDA）。
    N_ensemble_configurations : int
        アンサンブル設定の数。
    """

    def __init__(self, N_ensemble_configurations=32):
        """
        初期化メソッド。

        Parameters:
        ----------
        N_ensemble_configurations : int, optional
            アンサンブル設定の数（デフォルトは32）。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.N_ensemble_configurations = N_ensemble_configurations
        print("TabPFN model loaded.")

    def tabPFN_train(self, X_train, X_valid, y_train, y_valid):
        """
        TabPFNモデルを用いて訓練および評価を行う。

        Parameters:
        ----------
        X_train : numpy.ndarray
            訓練データの特徴量。
        X_valid : numpy.ndarray
            検証データの特徴量。
        y_train : numpy.ndarray
            訓練データのラベル。
        y_valid : numpy.ndarray
            検証データのラベル。

        Returns:
        -------
        TabPFNClassifier
            訓練済みのTabPFNモデル。
        """
        model = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.N_ensemble_configurations)
        model.fit(X_train, y_train, overwrite_warning=True)

        # モデルの予測
        y_eval, p_eval = model.predict(X_valid, return_winning_probability=True)
        y_pred = model.predict_proba(X_valid)
        
        # 精度とAUCスコアの計算
        score = roc_auc_score(y_valid, y_pred[:, 1])
        print('Accuracy:', accuracy_score(y_valid, y_eval))
        print('AUC:', score)
    
        return model


class CustomStacking:
    """
    カスタムスタッキングを用いて分類モデルの学習を行うクラス。

    Attributes:
    ----------
    N_SPLITS : int
        交差検証の分割数。
    random_seed : int
        ランダムシード値。
    tab_class : ClassificationModels
        タブPFN分類モデルのインスタンス。
    """

    def __init__(self, N_SPLITS, N_ensemble_configurations=32, random_seed=0):
        """
        初期化メソッド。

        Parameters:
        ----------
        N_SPLITS : int
            交差検証の分割数。
        N_ensemble_configurations : int, optional
            アンサンブル設定の数（デフォルトは32）。
        random_seed : int, optional
            ランダムシード値（デフォルトは0）。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.N_SPLITS = N_SPLITS
        self.random_seed = random_seed
        self.tab_class = ClassificationModels(N_ensemble_configurations)

    def classification(self, X_train, y_train, X_test):
        """
        カスタムスタッキングを用いて分類を行う。

        Parameters:
        ----------
        X_train : pandas.DataFrame
            訓練データの特徴量。
        y_train : pandas.Series
            訓練データのラベル。
        X_test : pandas.DataFrame
            テストデータの特徴量。

        Returns:
        -------
        tuple
            訓練データに対するアウトオブフォールドの予測結果とテストデータの平均予測結果。
        """
        # 入力データをNumPy配列に変換
        X_train = X_train.values
        y_train = y_train.values
        X_test = X_test.values
        
        # 結果を格納するための配列を初期化
        oof_valid = np.zeros((self.N_SPLITS, X_train.shape[0]))
        oof_test = np.zeros(X_test.shape[0])
        oof_test_skf = np.zeros((self.N_SPLITS, X_test.shape[0]))
    
        # Stratified K-Foldでデータセットを分割
        skf = StratifiedKFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.random_seed)
        for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
            print(f'[CV] {i+1}/{self.N_SPLITS}')

            # 訓練用データと検証用データに分割
            X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
            y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    
            # モデルの訓練
            model = self.tab_class.tabPFN_train(X_train_, X_valid_, y_train_, y_valid_)

            # アウトオブフォールドおよびテストデータに対する予測
            oof_valid[i, valid_index] = model.predict_proba(X_valid_)[:, 1]
            oof_test_skf[i, :] = model.predict_proba(X_test)[:, 1]
        
        # アウトオブフォールド予測の平均を計算
        oof_valid = np.sum(oof_valid, axis=0) / (self.N_SPLITS - 1)
        oof_test[:] = oof_test_skf.mean(axis=0)
        
        return oof_valid, oof_test
