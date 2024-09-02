import pandas as pd
import numpy as np

import re
import unicodedata
from kanjize import kanji2number

#年齢についての変換(20代とかは確率的に再配置)
def age_distribution(df, age):
    age = abs(age)
    age_counts =df[df.Age.between(age,age+9)].Age.value_counts()
    age_distribution = age_counts / age_counts.sum()
    age_distribution = age_distribution.to_dict()
    return np.random.choice(list(age_distribution.keys()), p=list(age_distribution.values()))
    
def Income_distribution(df, Income):

    tmp = df[df["MonthlyIncome"] % 1000 != 0].copy()
    Income_counts =tmp[tmp.MonthlyIncome.between(Income,Income + 100000)].MonthlyIncome.value_counts()
    Income_distribution = Income_counts / Income_counts.sum()
    Income_distribution = Income_distribution.to_dict()
    try:
        return np.random.choice(list(Income_distribution.keys()), p=list(Income_distribution.values()))
    except:
        return Income
    
def Age(txt):
    txt = str(txt)
    try:
        num = kanji2number(txt)
    except:
        if txt == "nan":
            return np.nan
        else:
            if txt[-1] == "代":
                txt = txt[:-1]
                age = kanji2number(txt)
                age = -age
            else:
                txt = txt[:-1]
                age = kanji2number(txt)
            return age

#売り込み時間についての変換
#分数に統一
def DurationOfPitch(time_txt):
    if type(time_txt) == float:
        return time_txt
    if "秒" in time_txt:
        time = time_txt.replace("秒","")
        time = int(time)
    if "分" in time_txt:
        time = time_txt.replace("分","")
        time = int(time)
        time = time * 60
    return time/60

##性別の表記統一
def Gender(txt):
    txt = txt.lower()
    txt = unicodedata.normalize('NFKC', txt)
    txt = txt.replace(' ', '')
    return txt

#フォローアップの数を1/100に変換
def NumberOfFollowups(x):
    if x >= 100:
        x = x/100
    return x


# Unicode正規化を行う関数
def normalize_unicode(text):
    return unicodedata.normalize('NFKD', text)
# キリル文字をローマ字に変換するマッピング辞書
cyrillic_to_latin_map = {
    'А': 'A', 'а': 'a',
    'Б': 'B', 'б': 'b',
    'Г': 'G', 'г': 'g',
    'Д': 'D', 'д': 'd',
    'Е': 'E', 'е': 'e',
    'Ё': 'Yo', 'ё': 'yo',
    'Ж': 'Zh', 'ж': 'zh',
    'З': 'Z', 'з': 'z',
    'И': 'I', 'и': 'i',
    'Й': 'Y', 'й': 'y',
    'К': 'K', 'к': 'k',
    'Л': 'L', 'л': 'l',
    'М': 'M', 'м': 'm',
    'Н': 'N', 'н': 'n',
    'О': 'O', 'о': 'o',
    'П': 'P', 'п': 'p',
    'Р': 'R', 'р': 'r',
    'С': 'C', 'с': 'c',
    'Т': 'T', 'т': 't',
    'У': 'U', 'у': 'u',
    'Ф': 'F', 'ф': 'f',
    'Х': 'Kh', 'х': 'kh',
    'Ц': 'Ts', 'ц': 'ts',
    'Ч': 'Ch', 'ч': 'ch',
    'Ш': 'Sh', 'ш': 'sh',
    'Щ': 'Shch', 'щ': 'shch',
    'Ъ': '', 'ъ': '',
    'Ы': 'Y', 'ы': 'y',
    'Ь': 'b', 'ь': 'b',
    'Э': 'E', 'э': 'e',
    'Ю': 'Yu', 'ю': 'yu',
    'Я': 'Ya', 'я': 'ya',
    'ѵ':'v'
}

# キリル文字をローマ字に変換する関数
def transliterate(text, char_map):
    return ''.join([char_map.get(char, char) for char in text])

# 文字列を正規化し置換する関数
def ProductPitched(text, char_map, pattern_map):
    # Unicode正規化
    text = normalize_unicode(text)
    text = text.lower()

    # キリル文字をローマ字に変換
    text = transliterate(text, cyrillic_to_latin_map)

    # 似た文字の置換
    for original, replacement in char_map.items():
        text = text.replace(original, replacement)
    
    # 特定のパターンの置換
    for pattern, replacement in pattern_map.items():
        text = re.sub(pattern, replacement, text)
    
    return text.lower()


#
def extract_numbers(text):
    # 正規表現を使用して数値を抽出
    numbers = re.findall(r'\d+', text)
    return numbers

#１年間の旅行数
def NumberOfTrips(txt):
    if type(txt) == float:
        return txt
    times = 1
    if "半年" in txt:
        times = 2
    elif "四半期" in txt:
        times = 4
        txt = txt.replace("四半期","")
    num = int(extract_numbers(txt)[0])
    num = num * times
    return num

# 月給を表記統一
def MonthlyIncome(x):
    if type(x) == float:
        return x
    
    if "万円" in x:
        x = x.replace("万円","").replace("月収","")
        x = float(x) * 10000
    return x

#customer_infoから情報を取り出す
def customer_info(df):
    df_copy = df["customer_info"]

    for replace_token in [",", "、", " ", "　", "／","\t","\n"]:
        df_copy = df_copy.str.replace(replace_token,"/")

    tmp = []
    for i, t in enumerate(df_copy.values):
        tmp.append(t.split("/")[0])
    df["marriage"] = pd.DataFrame(tmp)
    
    tmp = []
    for i, t in enumerate(df_copy.values):
        tmp.append(t.split("/")[1])
    _tmp = pd.DataFrame(tmp)[0].str.replace("自動車","車")
    _tmp = _tmp.str.replace("乗用車","車")
    _tmp = _tmp.str.replace("自家用車","車")
    _tmp = _tmp.str.replace("保有なし","未所有")
    _tmp = _tmp.str.replace("保有","所有")
    _tmp = _tmp.str.replace("所持","所有")
    _tmp = _tmp.str.replace("あり","所有")
    _tmp = _tmp.str.replace("なし","未所有")
    df["car"] = pd.DataFrame(_tmp)

    tmp = []
    for i, t in enumerate(df_copy.values):
        t = "".join(t.split("/")[2:])
        t = extract_numbers(t)
        tmp.append(t)
    df["child"] = pd.DataFrame(tmp).astype(float).fillna(0)
    return df