import numpy as np

import json
import optuna

from imblearn.over_sampling import RandomOverSampler,ADASYN
from imblearn.under_sampling import RandomUnderSampler,EditedNearestNeighbours
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

import logging
logging.getLogger("lightgbm").setLevel(logging.WARNING)

np.random.seed(42)

class KFOLD_CV:
    def __init__(self, N_SPLITS=5, random_seed = 0, train_func = [], pred_func = []):
        self.N_SPLITS=N_SPLITS
        self.random_seed = random_seed
        self.train_func = train_func
        self.pred_func = pred_func


    def KFOLD(self, X_train, y_train, params):
        columns = X_train.columns.tolist()
    
        # 結果を入れるための箱の準備
        oof_valid = np.zeros(X_train.shape[0])
    
        # 交差検定をインスタンス化
        skf = StratifiedKFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.random_seed)
        for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
            print('[CV] {}/{}'.format(i+1, self.N_SPLITS))
    
            # 学習データと答えを分割
            X_train_, X_valid_ = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_, y_valid_ = y_train.iloc[train_index], y_train.iloc[valid_index]
    
            # 学習
            model = self.train_func(X_train_, X_valid_, y_train_, y_valid_, params)
    
            oof_valid[valid_index] = self.pred_func(model, X_valid_)
            score = roc_auc_score(y_valid_, oof_valid[valid_index])
            print(f'Fold {i+1} AUC: {score}')
    
        score = roc_auc_score(y_train, oof_valid)
        return score


class Classification_models:
    def __init__(self,
                 NUM_CLASS,
                 random_seed = 0,
                 isOversampling = False,
                 categorical_columns = []):
        self.NUM_CLASS = NUM_CLASS
        self.random_seed = random_seed
        self.isOversampling = isOversampling
        self.categorical_columns = categorical_columns
        
        if self.isOversampling:
            #オーバーサンプリング
            self.ros = ADASYN(random_state=42)
            #必要なものだけを使うようにする
            self.enn = EditedNearestNeighbours()


    def lgb_train(self, X_train, X_valid, y_train, y_valid):
        if self.isOversampling:
            X_train, y_train = self.ros.fit_resample(X_train, y_train)
            #X_train, y_train = self.enn.fit_resample(X_train, y_train)
        
        # カテゴリ変数をカテゴリ型に変換
        X_train.loc[:, self.categorical_columns] = X_train[self.categorical_columns].astype('category')
        X_valid.loc[:, self.categorical_columns] = X_valid[self.categorical_columns].astype('category')

        # データセットを生成する
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = self.categorical_columns) 
        lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature = self.categorical_columns, reference=lgb_train) 
    
        callbacks = [ 
                lgb.log_evaluation(500), 
                lgb.early_stopping(300, verbose=False), 
            ]
    
        dir = "config/params_lgb.json"
        encoding = "utf-8"
        with open(dir, mode="rt", encoding="utf-8") as f:
        	params = json.load(f)

        params.update(task = 'train',
                      boosting_type ='gbdt',
                      objective = 'binary',
                      metric = 'auc',
                      is_unbalance = True,
                      random_state = 0,
                      n_estimaters = 10000,

                      #random_state = self.random_seed,
                      verbosity = -1,  # ログレベルを抑制
                     )

        self.model = lgb.train(params, 
                    lgb_train, 
                    valid_sets=lgb_eval,
                       )

        return self.model
    def lgb_pred(self, data):
        data.loc[:, self.categorical_columns] = data[self.categorical_columns].astype('category')
        return self.model.predict(data)


    def xgb_train(self, X_train, X_valid, y_train, y_valid):
        if self.isOversampling:
            X_train, y_train = self.ros.fit_resample(X_train, y_train)
            X_train, y_train = self.enn.fit_resample(X_train, y_train)

        # カテゴリ変数をカテゴリ型に変換
        X_train.loc[:, self.categorical_columns] = X_train[self.categorical_columns].astype('category')
        X_valid.loc[:, self.categorical_columns] = X_valid[self.categorical_columns].astype('category')
        # データセットを生成する
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        # クラスの比率に基づいてscale_pos_weightを計算
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos

        dir = "config/params_xgb.json"
        encoding = "utf-8"
        with open(dir, mode="rt", encoding="utf-8") as f:
        	params = json.load(f)
        params.update(objective = "multi:softprob" if self.NUM_CLASS > 2 else "binary:logistic",
                      num_class = self.NUM_CLASS if self.NUM_CLASS > 2 else 1,
                      tree_method = 'hist',  # 'gpu_hist' の代わりに 'hist' を使用
                      device = 'cuda',       # GPUでのトレーニングを指定
                      eval_metric = 'auc',
                      scale_pos_weight = scale_pos_weight,  # クラスの不均衡を調整
                      random_state = self.random_seed,
        )



        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=0,
            evals=[(dtrain, "train"), (dvalid, "val")],
        )

        return self.model

    def xgb_pred(self, data):
        data.loc[:, self.categorical_columns] = data[self.categorical_columns].astype('category')
        ddata = xgb.DMatrix(data, enable_categorical=True)
        return self.model.predict(ddata)


    def cat_train(self, X_train, X_valid, y_train, y_valid):
        if self.isOversampling:
            X_train, y_train = self.ros.fit_resample(X_train, y_train)
            X_train, y_train = self.enn.fit_resample(X_train, y_train)
            
        X_train.loc[:, self.categorical_columns] = X_train[self.categorical_columns].astype('category')
        X_valid.loc[:, self.categorical_columns] = X_valid[self.categorical_columns].astype('category')
        
        c_train = Pool(X_train, label=y_train, cat_features = self.categorical_columns)  
        c_valid = Pool(X_valid, label=y_valid, cat_features = self.categorical_columns)
        
        dir = "config/params_cat.json"
        encoding = "utf-8"
        with open(dir, mode="rt", encoding="utf-8") as f:
        	params = json.load(f)
            
        params.update(loss_function = "MultiClass" if self.NUM_CLASS > 2 else "Logloss",
                     iterations = 10000,
                     learning_rate = 0.01,
                     eval_metric='AUC',
                     od_type = 'Iter',
                     od_wait = 100,
                     verbose = 0,
                     random_seed = self.random_seed,
                     task_type = 'GPU',
        )
        
        self.model = CatBoostClassifier(**params )
        self.model.fit(c_train,
                  eval_set=c_valid,
                  verbose_eval=1000,
                  plot=False,
                 )
        return self.model
    
    def cat_pred(self, data):
        data.loc[:, self.categorical_columns] = data[self.categorical_columns].astype('category')
        c_data = Pool(data, cat_features = self.categorical_columns)
        return self.model.predict_proba(c_data)[:,1]









    
    def lgb_tune(self, X, y):
        def _lgb_train(X_train, X_valid, y_train, y_valid, params):
            # データセットを生成する
            lgb_train = lgb.Dataset(X_train, y_train) 
            lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train) 
        
            self.model = lgb.train(params, 
                                    lgb_train, 
                                    valid_sets=lgb_eval,
                                       )
    
            return self.model


        def _lgb_pred(model, data):
            return model.predict(data)
            
        def objective(trial, X, y):       
            params = {
                "verbosity": -1,
                "max_bin": trial.suggest_int("max_bin", 10, 500),
                "num_leaves": trial.suggest_int("num_leaves", 2, 500),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 50),
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-8, 10.0, log=True),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "path_smooth": trial.suggest_int("path_smooth", 0, 10),
                
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                "is_unbalance": True,
                "random_state" :0,
                'metric': 'auc',
                'n_estimaters': 10000,
            }

            CV = KFOLD_CV(N_SPLITS=5, random_seed = 0, train_func = _lgb_train, pred_func = _lgb_pred)
            score = CV.KFOLD(X, y, params)

            return score 
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=1000)
        
        #スコアを見る
        print(study.best_params)    
        print(study.best_value)
        
        
        dir = "config/params_lgb.json"
        data = study.best_params
        with open(dir, mode="wt", encoding="utf-8") as f:
        	json.dump(data, f, ensure_ascii=False, indent=2)



    def xgb_tune(self, X, y):
        def _xgb_train(X_train, X_valid, y_train, y_valid, params):
            if self.isOversampling:
                X_train, y_train = self.ros.fit_resample(X_train, y_train)
                X_train, y_train = self.enn.fit_resample(X_train, y_train)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=100000,
                early_stopping_rounds=100,
                verbose_eval=0,
                evals=[(dtrain, "train"), (dvalid, "val")],
            )
    
            return self.model
    
        def _xgb_pred(model, data):
            ddata = xgb.DMatrix(data)
            return model.predict(ddata)
        
        def objective(trial, X, y):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            # ハイパーパラメータの提案
            params = {
                "objective": "multi:softprob" if self.NUM_CLASS > 2 else "binary:logistic",
                'num_class': self.NUM_CLASS if self.NUM_CLASS > 2 else 1,
                "random_state": 42,
                "eval_metric":'auc',
                "is_unbalance":  True,

                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }

            CV = KFOLD_CV(N_SPLITS=5, random_seed = 0, train_func = _xgb_train, pred_func = _xgb_pred)
            score = CV.KFOLD(X, y, params)

            print(score)
            return score 
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=1000)
        
        #スコアを見る
        print(study.best_params)    
        print(study.best_value)
        
        
        dir = "config/params_xgb.json"
        data = study.best_params
        with open(dir, mode="wt", encoding="utf-8") as f:
        	json.dump(data, f, ensure_ascii=False, indent=2)



    def cat_tune(self, X, y):
        def _cat_train(X_train, X_test, y_train, y_test, params):
                
            c_train = Pool(X_train, label=y_train)#,cat_features=categorical_features)  
            c_valid = Pool(X_test, label=y_test)#,cat_features=categorical_features)
            
            self.model = CatBoostClassifier(**params )
            self.model.fit(c_train,
                      eval_set=c_valid,
                      verbose_eval=1000,
                      plot=False,
                     )
            return self.model
        
        def _cat_pred(model, data):
            c_data = Pool(data)
            return model.predict_proba(c_data)[:,1]

        def objective(trial, X, y):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)
            c_train = Pool(X_train, label=y_train)#,cat_features=categorical_features)  
            c_valid = Pool(X_valid, label=y_valid)#,cat_features=categorical_features)
        
            params = {"loss_function" : "MultiClass" if self.NUM_CLASS > 2 else "Logloss",
                     "iterations":10000,
                     "eval_metric":'AUC',
                     "learning_rate" : 0.01,
                     "od_type": 'Iter',
                     "od_wait": 100,
                     "verbose":0,
                     "random_seed":42,
                     "task_type":'GPU',
                     "depth": trial.suggest_int("depth", 4, 10),
                     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0),
                     "border_count": trial.suggest_int("border_count", 32, 255),
                     "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                     "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0),
                    }

            CV = KFOLD_CV(N_SPLITS=5, random_seed = 0, train_func = _cat_train, pred_func = _cat_pred)
            score = CV.KFOLD(X, y, params)

            print(score)
            return score 
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
        
        #スコアを見る
        print(study.best_params)    
        print(study.best_value)
        
        
        dir = "config/params_cat.json"
        data = study.best_params
        with open(dir, mode="wt", encoding="utf-8") as f:
        	json.dump(data, f, ensure_ascii=False, indent=2)

