import lightgbm as lgb
from sklearn.model_selection import train_test_split

class MissingImputation:
    """
    欠損値をLightGBMを用いた回帰モデルで補完するためのクラス。
    
    Attributes:
    ----------
    target : str
        欠損補完の対象となる列名（ターゲット変数）。
    y : pandas.Series
        ターゲット変数のデータ。
    X : pandas.DataFrame
        ターゲット変数以外の特徴量データ。
    model : lightgbm.Booster
        学習済みのLightGBMモデル。
    """
    
    def __init__(self, target, train):
        """
        初期化メソッド。ターゲット変数と訓練データを設定する。

        Parameters:
        ----------
        target : str
            欠損補完の対象となる列名（ターゲット変数）。
        train : pandas.DataFrame
            訓練データセット。
        """
        self.target = target
        self.y = train[target]  # ターゲット変数を取得
        self.X = train.drop(columns=[target], axis=1)  # ターゲット変数を除いた特徴量データ

    def lgb_regression(self):
        """
        LightGBMを用いた回帰モデルを訓練するメソッド。
        """
        # 訓練データと検証データに分割
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, random_state=42)
        lgb_train = lgb.Dataset(X_train, y_train)  # 訓練データセットの作成
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)  # 検証データセットの作成
    
        # コールバック関数を設定
        callbacks = [
            lgb.log_evaluation(500),  # 500回ごとにログを出力
            lgb.early_stopping(300),  # 300回改善がなければ早期停止
        ]
    
        # LightGBMのハイパーパラメータ設定
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 0,
            'learning_rate': 0.01,
            'feature_pre_filter': False,
            'lambda_l1': 2.3078927924015246e-05,
            'lambda_l2': 0.027571669518052705,
            'num_leaves': 64,
            'feature_fraction': 0.6,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 20,
            'num_iterations': 10000,
            'early_stopping_round': None
        }
        
        # LightGBMモデルの訓練
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=lgb_eval,
            callbacks=callbacks,
        )

    def pred_lgb(self, data):
        """
        訓練済みのモデルを使用してデータの予測を行うメソッド。

        Parameters:
        ----------
        data : pandas.DataFrame
            予測を行うための特徴量データ。

        Returns:
        -------
        numpy.ndarray
            予測結果。
        """
        return self.model.predict(data)
