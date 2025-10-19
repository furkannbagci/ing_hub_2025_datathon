import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

df_customer_history = pd.read_csv("customer_history.csv")
df_customer = pd.read_csv("customers.csv")
df_referance = pd.read_csv("referance_data.csv")
df_referance_test = pd.read_csv("referance_data_test.csv")


#****************************************************************************************
#FONKSİYONLAR

def df_onizleme(df):
    print(f"describe :  \n {df.describe()}")
    print("########################################################")
    print(f"nunique :  \n {df.nunique()}")
    print("########################################################")
    print(f"null counts : \n {df.isnull().sum()}")
    print("########################################################")
    print(f"info : \n {df.info()}")


def ikili_karsilastirma(df,col1,col2,tru=0):

    d=df.groupby([col1,col2]).size().reset_index()
    print(d)
    if(tru == 1):
        plt.title(col1 + " -- "+ col2)
        plt.xlabel(col1+"-"+col2)
        d.plot(kind='bar',figsize=(20,10))
        plt.show()

def plot_category_count(df):
    cols = [col for col in df.columns if df[col].dtype == "object"]
    for col in cols:
        col_plot = df[col].value_counts()
        col_plot.plot(kind='bar',figsize=(12,8))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.show()

def categorical_col_churn_analysis(df):
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    for col in cat_cols:
        plt_cols = df.groupby(col)["churn"].mean()
        plt_cols.plot(kind='bar')
        plt.title(f"churn - {col}")
        plt.show()

def numerical_col_churn_analysis(df , max_unique = 20):
    not_in = ["mobile_eft_all_amt","cc_transaction_all_amt","cust_id","churn"]
    num_cols = [col for col in df.columns if df[col].dtype != "object" and col not in not_in]
    for col in num_cols:
        num_unique = df[col].nunique()
        if num_unique > max_unique:
            new_col = df.groupby(pd.cut(df[col],bins=60))["churn"].mean()
            new_col.plot(kind='bar')
            plt.title(f"churn - {col}")
        else:
            plt_cols = df.groupby(col)["churn"].mean()
            plt_cols.plot(kind='bar')
            plt.title(f"churn - {col}")
        plt.show()

def transaction_aim_for_churn_0(df,churn_value = 0):
    df_copy = df.copy()
    #churn değeri 0 olanlar için yıl bazlı harcama tablosu
    df_copy = df_copy[df_copy["churn"] == churn_value]

    plt.figure(figsize=(20, 15))
    # ay bazlı işlem ortalamaları
    monthly_agg = df_copy.groupby(pd.Grouper(key='date', freq='ME')).agg({
        'mobile_eft_all_cnt': 'mean',
        'mobile_eft_all_amt': 'mean',
        'cc_transaction_all_amt': 'mean',
        'cc_transaction_all_cnt': 'mean',
        'active_product_category_nbr': 'mean'
    }).reset_index()

    plt.title(f"Transaction Aim for Churn {churn_value}")

    plt.subplot(3, 2, 1)
    plt.plot(monthly_agg['date'], monthly_agg['mobile_eft_all_cnt'], marker='o')
    plt.title('Average Mobile EFT Count Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 2)
    plt.plot(monthly_agg['date'], monthly_agg['mobile_eft_all_amt'], marker='o', color='orange')
    plt.title('Average Mobile EFT Amount Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 3)
    plt.plot(monthly_agg['date'], monthly_agg['cc_transaction_all_amt'], marker='o', color='green')
    plt.title('Average Credit Card Transaction Amount Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 4)
    plt.plot(monthly_agg['date'], monthly_agg['cc_transaction_all_cnt'], marker='o', color='red')
    plt.title('Average Credit Card Transaction Count Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 5)
    plt.plot(monthly_agg['date'], monthly_agg['active_product_category_nbr'], marker='o', color='purple')
    plt.title('Average Active Product Categories Over Time')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def harcama_tenure_oranı(df):
    df_copy = df.copy()
    sum_val = df_copy.groupby("cust_id").agg({
        "mobile_eft_all_amt" : "sum",
        "cc_transaction_all_amt" : "sum",
        "mobile_eft_all_cnt" : "sum",
        "cc_transaction_all_cnt" : "sum",
        "tenure" : "min"
    })

    cols = ["mobile_eft_all_amt","cc_transaction_all_amt","mobile_eft_all_cnt","cc_transaction_all_cnt"]
    for col in cols:
        sum_val[f"{col}_tenure_ratio"] = sum_val[col]/sum_val["tenure"]

    """
    #rationun churn üzerindeki etkisine bakmak için
    
    sum_val["eft_ratio_bin"] = pd.qcut(sum_val["mobile_eft_all_amt_tenure_ratio"], 30, duplicates='drop')
    sum_val["cc_ratio_bin"] = pd.qcut(sum_val["cc_transaction_all_amt_tenure_ratio"], 30, duplicates='drop')
    sum_val["mobile_cnt"] = pd.qcut(sum_val["mobile_eft_all_cnt_tenure_ratio"], 30, duplicates='drop')
    sum_val["cc_cnt"] = pd.qcut(sum_val["cc_transaction_all_cnt_tenure_ratio"], 30, duplicates='drop')
    
    
    
    # Her oran aralığına göre ortalama churn oranını hesapla
    eft_churn = sum_val.groupby("eft_ratio_bin")["churn"].mean()
    cc_churn = sum_val.groupby("cc_ratio_bin")["churn"].mean()
    mobile_cnt = sum_val.groupby(["mobile_cnt"])["churn"].mean()
    cc_tr_cnt = sum_val.groupby(["cc_cnt"])["churn"].mean()

    plt.figure(figsize=(12, 5))

    plt.subplot(2, 2, 1)
    eft_churn.plot(kind="bar", rot=45)
    plt.title("Mobile EFT / Tenure Oranı'na Göre Ortalama Churn")
    plt.ylabel("Ortalama Churn Oranı")
    plt.xlabel("Mobile EFT / Tenure Oranı Aralığı")

    plt.subplot(2, 2, 2)
    cc_churn.plot(kind="bar", rot=45)
    plt.title("Kredi Kartı / Tenure Oranı'na Göre Ortalama Churn")
    plt.ylabel("Ortalama Churn Oranı")
    plt.xlabel("CC / Tenure Oranı Aralığı")
    
    plt.tight_layout()
    plt.show()

    plt.subplot(2, 2, 3)
    mobile_cnt.plot(kind="bar", rot=45)
    plt.title("mobile count / Tenure Oranı'na Göre Ortalama Churn")
    plt.ylabel("Ortalama Churn Oranı")
    plt.xlabel("mobile_count / Tenure Oranı Aralığı")

    plt.subplot(2, 2, 4)
    cc_tr_cnt.plot(kind="bar", rot=45)
    plt.title("cc count / Tenure Oranı'na Göre Ortalama Churn")
    plt.ylabel("Ortalama Churn Oranı")
    plt.xlabel("CC / Tenure Oranı Aralığı")

    plt.tight_layout()
    plt.show()
    """

    print(sum_val)


def calculate_slope(series):
    """
    lineer regrasyon eğimini hesaplama
    harcama tutarlarının zaman aralığında lineer olarak eğilimin hesaplamak için
    """

    if len(series) < 2:  # eğer 2'den az veri varsa hesaplama olamaz
        return 0

    # y=harcama x=zaman
    y = series.values
    x = np.arange(len(series))

    # 1.dereceden polinom
    # sonuç [slope,intercept]
    slope = np.polyfit(x, y, 1)[0]
    return slope



def monthly_numerical_analysis(df, month=3):
    """
    month değeri kadar bir zaman aralığında sayısal işlem değerlerinin ["mean","std","max","min","last"] değerlerini döndürür
    month değeri için işlem sayısının 0 olduğu ay sayısını döndürür
    """

    df_copy_train = df.copy()

    num_col = ["mobile_eft_all_cnt", "active_product_category_nbr","mobile_eft_all_amt",
               "cc_transaction_all_amt", "cc_transaction_all_cnt"]


    df_copy_train["month_diff"] = ((df_copy_train["ref_date"].dt.year-df_copy_train["date"].dt.year)*12 +
                                   (df_copy_train["ref_date"].dt.month-df_copy_train["date"].dt.month)).astype(int)
    final_month_data = df_copy_train[(df_copy_train["month_diff"] > 0) & (df_copy_train["month_diff"] <= month)]

    #verilen ay içerisinde işlem sayılarının 0 olduğu ay sayısı
    mobile_eft_zeros_count = final_month_data.groupby(["cust_id","ref_date"])["mobile_eft_all_cnt"].agg(lambda x: (x==0).sum())
    mobile_eft_zeros_count = mobile_eft_zeros_count.reset_index(name='mobile_eft_zeros_count')
    cc_transaction_zeros_count = final_month_data.groupby(["cust_id","ref_date"])["cc_transaction_all_cnt"].agg(lambda x: (x==0).sum())
    cc_transaction_zeros_count = cc_transaction_zeros_count.reset_index(name='cc_transaction_zeros_count')

    #lineer eğilim hesaplama kısmı
    slope_mobile = final_month_data.groupby("cust_id")["mobile_eft_all_amt"].apply(calculate_slope)
    slope_cc_transaction = final_month_data.groupby("cust_id")["cc_transaction_all_amt"].apply(calculate_slope)
    #veri eksikliği sonucu nan olabilecek değerlerin yerini 0 ile doldurma
    slope_mobile.fillna(0, inplace=True)
    slope_cc_transaction.fillna(0, inplace=True)
    #yeniden isimlendirme
    new_name = "trend_mobile"
    slope_mobile.name = new_name
    new_name = "trend_cc_transaction"
    slope_cc_transaction.name = new_name


    agg_features = final_month_data.groupby(["cust_id","ref_date"]).agg({**{col:["mean","std","max","min","last","sum"] for col in num_col}, #değerlere yapılacak işlemler
                                                                    "date":["count"]})#aktif ay sayısı

    agg_features.columns = ["_".join(col).strip() for col in agg_features.columns.values]
    agg_features.reset_index(inplace=True)

    #0 işlem sayılarını merge
    agg_features = agg_features.merge(mobile_eft_zeros_count, on=["cust_id","ref_date"])
    agg_features = agg_features.merge(cc_transaction_zeros_count, on=["cust_id","ref_date"])

    #lineer eğilim merge
    agg_features = agg_features.merge(slope_mobile, on=["cust_id"])
    agg_features = agg_features.merge(slope_cc_transaction, on=["cust_id"])

    #std nan olan değerleri 0 ile doldurulması (sadece 1 işlem yapılmış)
    std_cols = [col for col in agg_features.columns if 'std' in col]
    agg_features[std_cols] =  agg_features[std_cols].fillna(0)

    # sütun isimlerinde hangi ay bazında işlem yapıldığını da ekledim
    agg_features.columns = [f"{col}_last_{month}" if col != "cust_id" else col for col in agg_features.columns]

    print(agg_features)
    print(f"\n null count \n{agg_features.isnull().sum()}")
    print(f"\n unique count \n {agg_features.nunique()}")


def consecutive_zero_transection_count(df):
    """
    consecutive şekilde ref_Daet tarihinden geriye dönük olarak kaç ay işlem yapılmadığını döndürür
    """
    df_copy = df.copy()
    #verileri id asc ref_Date desc olacak şekilde sıraladık
    df_copy = df_copy.sort_values(["cust_id","date"],ascending=[True,False])

    df_copy["is_mobile_zero"] = (df_copy["mobile_eft_all_cnt"]==0)
    df_copy["is_cc_zero"] = (df_copy["cc_transaction_all_cnt"]==0)

    def döngü_hesaplama(col):
        count = 0
        for value in col:
            if value:
                count += 1
            else:
                break
        return count

    consecutive_counts = df_copy.groupby("cust_id").agg(
        consecutive_mobile_zeros = ("is_mobile_zero", döngü_hesaplama),
        consecutive_cc_zeros = ("is_cc_zero",döngü_hesaplama)
    )

    consecutive_counts.reset_index(inplace=True)

    print(consecutive_counts)
    #print(df_copy.head(20))




#****************************************************************************************

df_onizleme(df_customer_history)
df_onizleme(df_customer)

ikili_karsilastirma(df_customer,'gender','province',1)

#****************************************************************************************
#EKSİK VERİLERİN DOLDURULMASI

#date verisini object -> datetime yapma
df_customer_history["date"] = pd.to_datetime(df_customer_history["date"])
df_referance["ref_date"] = pd.to_datetime(df_referance["ref_date"])
df_referance_test["ref_date"] = pd.to_datetime(df_referance_test["ref_date"])

#df_customer_history eksik verilerin yerini 0 ile doldurma
df_customer_history["mobile_eft_all_cnt"] = df_customer_history["mobile_eft_all_cnt"].fillna(0)
df_customer_history["mobile_eft_all_amt"] = df_customer_history["mobile_eft_all_amt"].fillna(0)
df_customer_history["cc_transaction_all_amt"] = df_customer_history["cc_transaction_all_amt"].fillna(0)
df_customer_history["cc_transaction_all_cnt"] = df_customer_history["cc_transaction_all_cnt"].fillna(0)

#work_sector sütunundaki boş verileri doldurduk
df_customer["work_sector"]=df_customer["work_sector"].fillna("Unknown")


#****************************************************************************************
#MERGE İŞLEMLERİ
#df_customer,df_referance ve df_referance_test ile df_customer_history merge ettik
df_merge = df_customer.merge(df_customer_history,how = "inner",on = 'cust_id')
df_merge_train = df_merge.merge(df_referance,how = "inner",on = 'cust_id')
df_merge_test = df_merge.merge(df_referance_test,how = "inner",on = 'cust_id')


df_onizleme(df_merge_train)
df_onizleme(df_merge_test)
#****************************************************************************************
#VERİ ANALİZİ VE GÖRSELLEŞTİRME

#kategorik ve numeric değerlerin churn dağılımı
categorical_col_churn_analysis(df_merge_train)
numerical_col_churn_analysis(df_merge_train) #tenure değerini son 50-60 verisi churn değeri yüksek

transaction_aim_for_churn_0(df_merge_train,0)
transaction_aim_for_churn_0(df_merge_train,1)

harcama_tenure_oranı(df_merge_train)
harcama_tenure_oranı(df_merge_test)

#****************************************************************************************
#FETAURE ENGİNEERİNG

#3-6-12 aylık numeric verilerin değerleri
monthly_numerical_analysis(df_merge_train,3)
monthly_numerical_analysis(df_merge_train,6)
monthly_numerical_analysis(df_merge_train,12)

consecutive_zero_transection_count(df_merge_train)




#df_merge = pd.get_dummies(df_merge,columns=['gender',"province","religion","work_type","work_sector"])
