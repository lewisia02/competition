import numpy as np
import pandas as pd


def sepalate_CatNum(df):
    # データフレームの各列の型を判別し、数値列とカテゴリ列に分ける
    numerical_columns = []
    categorical_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)
    # 結果を表示
    print("数値列:", numerical_columns)
    print("カテゴリ列:", categorical_columns)

    return numerical_columns, categorical_columns

def count_element(df, column):
    from collections import Counter
    element_count = Counter(df[column])

    for element, count in element_count.items():
        print(f"要素: {element}, 数: {count}")


def edit_Date(df, column):
    retval = df.copy()
    retval[column] = pd.to_datetime(retval[column], format='%d-%b-%y')
    retval[column[:-4] + "Year"] = retval[column].dt.year
    retval[column[:-4] + "Month"] = retval[column].dt.month
    #retval["ApprovalDay"] = retval["ApprovalDate"].dt.day
    return retval.drop([column],axis=1)

def yes_no(i):
    if i > 0:
        return 1
    return 0

def preprocess(train, test, ce_dict=[]):
    
    data_dic = {"train":train,"test":test}
    for tt in ["train","test"]:
        
        # 欠損数を数える
        data_dic[tt]["NullCount"] = data_dic[tt].isnull().sum(axis=1)        
        
        #欠測補完
        data_dic[tt]["RevLineCr"] = data_dic[tt]["RevLineCr"].fillna("N")
        data_dic[tt]["LowDoc"] = data_dic[tt]["LowDoc"].fillna("N")
        
        # 借り手の会社に関する変数（Sector, FranchiseCode）
        # 31-33, 44-45, 48-49 は同じらしい => 32,33を31に, 45を44に, 49を48に変換
        def Sector2label(df):
            for i,sec in enumerate([[11],[21],[22],[23],[31,32,33],[42],[44,45],[48,49],[51],[52],[53],[54],[55],[56],[61],[62],[71],[72],[81],[92]]):
                df["Sector"] = df["Sector"].map(lambda x:i if x in sec else x)
            return df
        data_dic[tt] = Sector2label(data_dic[tt])
        code_dict = {
            1: 0
        }
        data_dic[tt]["FranchiseCode"] = data_dic[tt]["FranchiseCode"].replace(code_dict)
        data_dic[tt]["IsFranchise"] = data_dic[tt]["FranchiseCode"].map(lambda x:1 if x == 0 else 0)
        
    
        # 今回の借り入れに関する変数（RevLineCr, LowDoc）
        # 公式ページには値の候補が2つ（YesとNoのYN）と記載があるが、実際の値の種類は2より多い。YN以外はNaNへ置換
        #revline_dict = {'0': np.nan, 'T': np.nan}
        revline_dict = {'0': np.nan}
        data_dic[tt]["RevLineCr"] = data_dic[tt]["RevLineCr"].replace(revline_dict)
    
        #lowdoc_dict = {'C': np.nan, '0': np.nan, 'S': np.nan, 'A': np.nan}
        lowdoc_dict = {'0': np.nan}
        data_dic[tt]["LowDoc"] = data_dic[tt]["LowDoc"].replace(lowdoc_dict)
    
    
        # 日付系の変数（DisbursementDate, ApprovalDate）
        data_dic[tt] = edit_Date(data_dic[tt], "DisbursementDate")
        data_dic[tt] = edit_Date(data_dic[tt], "ApprovalDate")
    
    
        # 本来数値型のものを変換する
        cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
        data_dic[tt][cols] = data_dic[tt][cols].applymap(lambda x: x.strip().replace('$', '').replace(',', '')).astype(float).astype(int)

        ##  ここまでに基礎変換記述
        
        # 特徴量エンジニアリング
        data_dic[tt]["DisbursementYear"] = data_dic[tt]["DisbursementYear"].fillna(data_dic[tt]["ApprovalFY"])
        
        #　州情報に関する
        data_dic[tt]["State_is_BankState"] = (data_dic[tt]["State"] == data_dic[tt]["BankState"])
        data_dic[tt]["State_is_BankState"] = data_dic[tt]["State_is_BankState"].replace({True: 1, False: 0})
        
        ## 金額に関する特徴量
        #対数変換する
        for c in ["DisbursementGross", "GrAppv", "SBA_Appv"]:
            data_dic[tt][c + "_log"] = data_dic[tt][[c]].apply(np.log).values
            
        data_dic[tt]['SBA_Portion'] = data_dic[tt]['SBA_Appv'] / data_dic[tt]['GrAppv']
        data_dic[tt]["DisbursementGrossRatio"] = data_dic[tt]["DisbursementGross"] / data_dic[tt]["GrAppv"]
        data_dic[tt]["DisbursementGrossSBARatio"] = data_dic[tt]["DisbursementGross"] / data_dic[tt]["SBA_Appv"]
        data_dic[tt]["MonthlyRepayment"] = data_dic[tt]["GrAppv"] / (data_dic[tt]["Term"]+1)
        
        data_dic[tt]["sub_GS"] = data_dic[tt]["GrAppv"] - data_dic[tt]["SBA_Appv"]
        data_dic[tt]["sub_DS"] = data_dic[tt]["DisbursementGross"] -  data_dic[tt]["SBA_Appv"]
        data_dic[tt]["sub_GD"] = data_dic[tt]["GrAppv"] - data_dic[tt]["DisbursementGross"]
        
        data_dic[tt]['SBA_Portion_log'] = data_dic[tt]['SBA_Appv_log'] / data_dic[tt]['GrAppv_log']
        data_dic[tt]["DisbursementGrossRatio_log"] = data_dic[tt]["DisbursementGross_log"] / data_dic[tt]["GrAppv_log"]
        data_dic[tt]["DisbursementGrossSBARatio_log"] = data_dic[tt]["DisbursementGross_log"] / data_dic[tt]["SBA_Appv_log"]
        data_dic[tt]["MonthlyRepayment_log"] = data_dic[tt]["GrAppv_log"] / (data_dic[tt]["Term"]+1)
        
        data_dic[tt]["sub_GS_log"] = data_dic[tt]["GrAppv_log"] - data_dic[tt]["SBA_Appv_log"]
        data_dic[tt]["sub_DS_log"] = data_dic[tt]["DisbursementGross_log"] -  data_dic[tt]["SBA_Appv_log"]
        data_dic[tt]["sub_GD_log"] = data_dic[tt]["GrAppv_log"] - data_dic[tt]["DisbursementGross_log"]
        
        
        
        #　従業員数に関する特徴量
        data_dic[tt]['IscreateJob'] = data_dic[tt].CreateJob.apply(yes_no)
        data_dic[tt]['IsRetained'] = data_dic[tt].RetainedJob.apply(yes_no)
        data_dic[tt]['SubRetained'] = data_dic[tt].RetainedJob - data_dic[tt].NoEmp
        data_dic[tt]['SubcreateJob'] = data_dic[tt].CreateJob- data_dic[tt].NoEmp
        
        ## 期間に関する特徴量
        data_dic[tt]["FY_Diff"] = data_dic[tt]["ApprovalFY"] - data_dic[tt]["DisbursementYear"]
        data_dic[tt]["ScheduledPaidoff"] = data_dic[tt]["DisbursementYear"] + (data_dic[tt]["Term"]//12)


        # 不動産担保ローン（融資期間20年以上）の分野
        data_dic[tt]['RealEstate'] = np.where(data_dic[tt]['Term'] >= 240, 1, 0)
        data_dic[tt]["shortTerm"] = data_dic[tt]["Term"].map(lambda x:1 if x <= 140 else 0)#shot Term
        data_dic[tt]["MiddleTerm"]= data_dic[tt]["Term"].map(lambda x:1 if 140 < x <= 220 else 0)# middle Term
        data_dic[tt]["LongTerm"]= data_dic[tt]["Term"].map(lambda x:1 if 220 < x <= 260 else 0)# long Term
        data_dic[tt]["superLongTerm"] = data_dic[tt]["Term"].map(lambda x:1 if 260 < x else 0)# big long Term


        # 大不況 (2007 ～ 2009 年) 中に活発な融資の分野
        data_dic[tt]['GreatRecession'] = np.where(((2007 <= data_dic[tt]['DisbursementYear']) & (data_dic[tt]['DisbursementYear'] <= 2009)) | 
                                             ((data_dic[tt]['DisbursementYear'] < 2007) & (data_dic[tt]['DisbursementYear'] + (data_dic[tt]['Term']/12) >= 2007)), 1, 0)
    
    #Sector に関する調整
    """
    for dl in ["DisbursementGross_log"]: 
        tmp = data_dic["train"][["Sector",dl]].groupby(["Sector"]).describe().reset_index()
        tmp.columns = ["Sector","count",dl + "_mean",dl + "_std",dl + "_min",dl + "_25%",dl + "_50%",dl + "_75%",dl + "_max"]
        tmp = tmp[["Sector", dl + "_mean",dl + "_std",dl + "_min",dl + "_25%",dl + "_75%",dl + "_max"]]
        for tt in ["train","test"]:
            data_dic[tt] = pd.merge(data_dic[tt], tmp, on = "Sector",how = "inner")
            #for c in [dl + "_mean",dl + "_std",dl + "_min",dl + "_25%",dl + "_50%",dl + "_75%",dl + "_max"]:
            #    data_dic[tt][c] = data_dic[tt][c] - data_dic[tt][dl]
    """
    # カテゴリカル変数の設定
    import category_encoders as ce
    from sklearn.preprocessing import LabelEncoder
    
    if len(ce_dict)>0:
        for c in ce_dict:
            print(c,data_dic["train"][c].unique())
            le = LabelEncoder()
            tmp = pd.concat([data_dic["train"],data_dic["test"]])
            #le.fit(data_dic["train"][c])
            le.fit(tmp[c])
            data_dic["train"][c + "_label"] = le.transform(data_dic["train"][c])
            data_dic["test"][c + "_label"] = le.transform(data_dic["test"][c])

            count_encoder = ce.CountEncoder(cols=[c])
            count_encoder.fit(data_dic["train"][c])
            data_dic["train"][c + "_count"] = count_encoder.transform(data_dic["train"][c])
            data_dic["test"][c + "_count"] = count_encoder.transform(data_dic["test"][c])
        
        data_dic["train"] = data_dic["train"].drop(ce_dict,axis=1)
        data_dic["test"] = data_dic["test"].drop(ce_dict,axis=1)
        
    data_dic["train"] = data_dic["train"].fillna(-1)
    data_dic["test"] = data_dic["test"].fillna(-1)
    return data_dic["train"], data_dic["test"]
    
def category_rate(train, test, column):
    print(len(train),len(test))
    df_copy = train.copy()
    index_Term = pd.DataFrame([_ for _ in range(400)],columns=["Term"])
    
    tmp1 = df_copy.groupby([column,"MIS_Status"]).count()["NoEmp"].reset_index()
    tmp2 = df_copy.groupby([column]).count()["NoEmp"].reset_index()
    tmp1.columns = [column,"MIS_Status","NoEmp_MIS_Status_count"]
    tmp2.columns = [column,"NoEmp_count"]
    tmp = pd.merge(tmp1,tmp2)

    tmp["NoEmp_rate"] = tmp["NoEmp_MIS_Status_count"]/tmp["NoEmp_count"]

    tmp1 = tmp[tmp["MIS_Status"] == 0].reset_index(drop=True).drop(["MIS_Status"],axis=1)
    tmp2 = tmp[tmp["MIS_Status"] == 1].reset_index(drop=True).drop(["MIS_Status"],axis=1)
    

    tmp1.columns = ["Term","NoEmp_MIS_Status_count_st0","NoEmp_count_st0","NoEmp_rate_st0"]
    tmp2.columns = ["Term","NoEmp_MIS_Status_count_st1","NoEmp_count_st1","NoEmp_rate_st1"]

    tmp = pd.merge(tmp1,tmp2, how = "outer")
    tmp = pd.merge(index_Term,tmp,how = "left")
    tmp = tmp.fillna(0)
    
    #for c in ["NoEmp_MIS_Status_count_st0","NoEmp_count_st0","NoEmp_rate_st0","NoEmp_MIS_Status_count_st1","NoEmp_count_st1","NoEmp_rate_st1"]:
    #    tmp[c] = tmp[c].rolling(window=3,min_periods=1).mean()
    
    train = pd.merge(train,tmp, on = ["Term"])
    test = pd.merge(test,tmp, on = ["Term"])
    print(len(train),len(test))
    return train, test