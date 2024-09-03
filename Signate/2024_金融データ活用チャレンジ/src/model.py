def apply_all_classifier(X, y, test,cv=3):
    """
    apply all classifiers in sklearn with k-fold cross validation
    cv should be greater than or equal to 2

    Args:
        x (_type_): independent variable
        y (_type_): target variable
        cv (int, optional): how many folds. Defaults to 3.
    """
    
    import time
    import pandas as pd
    import numpy as np
    from sklearn.utils import all_estimators
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    
    results_df = pd.DataFrame(columns=["model_name","accuracy","auc","timeElapsed","cv"])
    
    from sklearn.model_selection import StratifiedKFold
    
    #交差検定数
    N_SPLITS = cv
    
    #比率をたもって交差検定をインスタンス化
    skf = StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state = 42)
    
    valid_result = pd.DataFrame()
    test_result = pd.DataFrame()
    
    for model in all_estimators(type_filter="classifier"):
        #結果を入れるための箱を用意しておく
        oof_valid = np.zeros((X.shape[0]))
        oof_test = np.zeros((test.shape[0]))
        oof_test_skf = np.zeros((N_SPLITS, test.shape[0]))
    
        try:
            for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]


                results_df.to_csv("results.csv",index=False)
                time_start = time.time()
            

                try:
                    thismodel = model[1]()
                    print(model[0])
                    thismodel.fit(X_train,y_train)
                    y_pred = thismodel.predict(X_valid)
                    proba_val = thismodel.predict_proba(X_valid)[:,1]
                    proba_test = thismodel.predict_proba(test)[:,1]

                    acc = accuracy_score(y_true=y_valid,
                                            y_pred=y_pred)
                    fpr, tpr, thresholds = roc_curve(y_valid, proba_val)
                    auc_score = auc(fpr, tpr)
                    print("accuracy",acc)
                    print("auc",auc_score)
                    oof_valid[valid_index] = proba_val
                    oof_test_skf[i, :] = proba_test
                    
                    time_spent = time.time()-time_start
                    results_df = pd.concat([results_df,
                                            pd.DataFrame({"model_name":model[0],
                                                        "accuracy":[acc],
                                                        "auc":[auc_score],
                                                        "timeElapsed":[time_spent],
                                                        "cv":[i]})],
                                        axis=0)
                except  Exception as e:
                    print(e)
                    print("fitting error")
                    results_df = pd.concat([results_df,
                                            pd.DataFrame({"model_name":model[0],
                                                        "score":["fittingError"],
                                                        "timeElapsed":[0],
                                                        "cv":[i]})],
                                        axis=0)
        except:
            print("instanciating error")
            results_df=pd.concat([results_df,
                                        pd.DataFrame({"model_name":model[0],
                                                    "score":["instanciatingError"],
                                                    "timeElapsed":[0],
                                                    "cv":[i]})],
                                    axis=0)
        valid_result[model[0]] = oof_valid
        test_result[model[0]] = oof_test_skf.mean(axis=0)

        valid_result.to_csv("all_valid.csv",index=False)
        test_result.to_csv("all_test.csv",index=False)
        
    temp = results_df.pivot(index="model_name",columns="cv")
    temp = temp.reset_index()
    temp.columns = temp.columns.values
    temp.columns = ["model_name"]+temp.columns[1:].to_list()
    results_df["mean_accuracy"] = pd.to_numeric(results_df["accuracy"],errors="coerce")
    results_df["mean_auc"] = pd.to_numeric(results_df["auc"],errors="coerce")
    results_df["mean_time"] = pd.to_numeric(results_df["timeElapsed"],errors="coerce")
    results_df = (results_df.groupby("model_name"))[["mean_accuracy","mean_auc","mean_time"]].agg("mean")
    results_df = results_df.sort_values(by="mean_auc",ascending=False)
    results_df = results_df.reset_index()
    results_df = results_df.merge(temp,on=["model_name"])
    results_df.to_csv("results_classifier.csv",index=False)
    print("Done!")