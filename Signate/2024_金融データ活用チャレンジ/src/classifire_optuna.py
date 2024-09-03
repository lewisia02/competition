import optuna
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import Pool
from catboost import CatBoostClassifier
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def objective_lgb(trial, X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    dtrain = lgb.Dataset(X_train, label=y_train)

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
        'metric': 'auc',
        "learning_rate": 0.1,
        'num_iterations': 10000,
    }

    model = lgb.train(params,
                    dtrain,
                   )

    pred = model.predict(X_valid)
    label = y_valid.values
    
    # 陽性の確率だけが必要なので[:, 1]をして陰性の確率を落とす
    fpr, tpr, thresholds = roc_curve(label, pred)
    print(auc(fpr, tpr))
    
    return auc(fpr, tpr)

def objective_xgb(trial, X, y):
    
    param = {
        'objective': 'binary:logistic',
        'learning_rate': 0.01,
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000, 100),
        'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
        'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }
    model = XGBClassifier(random_state=42, 
                             tree_method='gpu_hist', 
                             gpu_id=0, 
                             predictor="gpu_predictor"
                             ,**param )
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train,eval_set=[(X_valid,y_valid)], early_stopping_rounds=150,verbose=False)
    pred = model.predict_proba(X_valid)[:,1]
    label = y_valid.values
    
    # 陽性の確率だけが必要なので[:, 1]をして陰性の確率を落とす
    fpr, tpr, thresholds = roc_curve(label, pred)
    print(auc(fpr, tpr))
    
    return auc(fpr, tpr)


def objective_cat(trial, X, y):
    
    param = {
            'iterations' : 1000,
            'depth' : trial.suggest_int('depth', 4, 10),
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            'random_strength' :trial.suggest_int('random_strength', 0, 100),
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
            }

    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)
    
    model = CatBoostClassifier(**param,
                              #task_type= 'GPU',
                              silent=True)
    model.fit(train_pool,
             eval_set=valid_pool,# 検証用データ
             early_stopping_rounds=10, # 10回以上精度が改善しなければ中止
             use_best_model=True,# 最も精度が高かったモデルを使用するかの設定
             )
    
        
    pred = model.predict_proba(X_valid)[:,1]
    label = y_valid.values
    
    # 陽性の確率だけが必要なので[:, 1]をして陰性の確率を落とす
    fpr, tpr, thresholds = roc_curve(label, pred)
    print(auc(fpr, tpr))
    
    return auc(fpr, tpr)

