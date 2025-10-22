import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

def max_date_is_eq_ref_date(df):
    df_copy = df.copy()
    a = df_copy.groupby(["cust_id","ref_date"])["date"].max().reset_index()
    print(a[a["date"] != a["ref_date"]].count())


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


def tenure_in_data(df):
    """
    her biir customer için veri setindeki tarih aralığının ay bazında gösterir yani datadaki tenure sayısını
    column name = active_tenure
    """
    df_copy = df.copy()

    diff = df_copy.groupby("cust_id")["date"].agg(["max","min"]) #lambda ile tek saturda yapmak yerine daha hızlı çözüm
    diff["active_tenure"] = ((diff["max"] - diff["min"]).dt.days / 30).round().astype(int)
    diff = diff.reset_index()
    diff = diff.drop(columns=["max","min"])

    return diff


def harcama_tenure_oranı(df):
    """
    customerlerin günlük ortalama değerleri
    """

    df_copy = df.copy()
    active_tenure=tenure_in_data(df_copy)
    df_copy = df_copy.merge(active_tenure, how='left', on='cust_id')

    sum_val = df_copy.groupby("cust_id").agg({
        "mobile_eft_all_amt" : "sum",
        "cc_transaction_all_amt" : "sum",
        "mobile_eft_all_cnt" : "sum",
        "cc_transaction_all_cnt" : "sum",
        "tenure" : "min",
        "active_tenure" : "min"
    })



    cols = ["mobile_eft_all_amt","cc_transaction_all_amt","mobile_eft_all_cnt","cc_transaction_all_cnt"]
    for col in cols:
        sum_val[f"{col}_tenure_ratio"] = sum_val[col]/sum_val["active_tenure"] #aktif olduğu süre bazında

    sum_val = sum_val.drop(columns=["mobile_eft_all_amt","cc_transaction_all_amt","mobile_eft_all_cnt","cc_transaction_all_cnt"])

    #print(sum_val)
    return sum_val



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


    agg_features = final_month_data.groupby(["cust_id","ref_date"]).agg({**{col:["mean","std","max","min","last","sum"] for col in num_col}})

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

    #print(agg_features)
    #print(f"\n null count \n{agg_features.isnull().sum()}")
    #print(f"\n unique count \n {agg_features.nunique()}"

    return agg_features

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

    #print(consecutive_counts)
    return consecutive_counts




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


#****************************************************************************************
#FETAURE ENGİNEERİNG

tenure_ratio_train = harcama_tenure_oranı(df_merge_train)
tenure_ratio_test = harcama_tenure_oranı(df_merge_test)

#3-6-12 aylık numeric verilerin değerleri
month_numerical_analys_train_3 = monthly_numerical_analysis(df_merge_train,3)
month_numerical_analys_train_6 = monthly_numerical_analysis(df_merge_train,6)
month_numerical_analys_train_12 = monthly_numerical_analysis(df_merge_train,12)


month_numerical_analys_test_3 = monthly_numerical_analysis(df_merge_test,3)
month_numerical_analys_test_6 = monthly_numerical_analysis(df_merge_test,6)
month_numerical_analys_test_12 = monthly_numerical_analysis(df_merge_test,12)


#aralıksız harcama yapılmayan tarihler
cons_zero_tran_train = consecutive_zero_transection_count(df_merge_train)
cons_zero_tran_test = consecutive_zero_transection_count(df_merge_test)


#ref_date sütunlarının tek sütuna indirgenmesi
month_numerical_analys_train_3.rename(columns={'ref_date_last_3': 'ref_date'}, inplace=True)
month_numerical_analys_train_6.rename(columns={'ref_date_last_6': 'ref_date'}, inplace=True)
month_numerical_analys_train_12.rename(columns={'ref_date_last_12': 'ref_date'}, inplace=True)


month_numerical_analys_test_3.rename(columns={'ref_date_last_3': 'ref_date'}, inplace=True)
month_numerical_analys_test_6.rename(columns={'ref_date_last_6': 'ref_date'}, inplace=True)
month_numerical_analys_test_12.rename(columns={'ref_date_last_12': 'ref_date'}, inplace=True)


#get_dummies ile kategorik değişkenlerin true false ile ifadesi
cat_cols = ["gender","province","religion","work_type","work_sector"]

df_get_dummies = pd.get_dummies(df_customer,columns = cat_cols,drop_first = True)

#MERGE İŞLEMLERİ

df_train_base = pd.merge(df_referance, df_get_dummies, on='cust_id', how='left')
df_test_base = pd.merge(df_referance_test, df_get_dummies, on='cust_id', how='left')

final_data_train = df_train_base.merge(tenure_ratio_train, on='cust_id', how='left')
final_data_train = final_data_train.merge(cons_zero_tran_train,on = 'cust_id',how = 'left')
final_data_train = month_numerical_analys_train_3.merge(month_numerical_analys_train_6,on = ['cust_id',"ref_date"],how = 'left')
final_data_train = final_data_train.merge(month_numerical_analys_train_12,on = ['cust_id',"ref_date"],how = 'left')


final_data_test = df_test_base.merge(tenure_ratio_test, on='cust_id', how='left')
final_data_test = final_data_test.merge(cons_zero_tran_test,on = 'cust_id',how = 'left')
final_data_test = month_numerical_analys_test_3.merge(month_numerical_analys_test_6,on = ['cust_id',"ref_date"],how = 'left')
final_data_test = final_data_test.merge(month_numerical_analys_test_12,on = ['cust_id',"ref_date"],how = 'left')


final_data_train = final_data_train.merge(df_get_dummies,on = 'cust_id',how = 'left')
final_data_test = final_data_test.merge(df_get_dummies,on = 'cust_id',how = 'left')


#nan değerlerin 0 ile doldurulması
feature_cols_train = [col for col in final_data_train.columns if col not in df_train_base]
feature_cols_test = [col for col in final_data_test.columns if col not in df_test_base]

final_data_train[feature_cols_train] = final_data_train[feature_cols_train].fillna(0)
final_data_test[feature_cols_test] = final_data_test[feature_cols_test].fillna(0)

#çarpıklık belirleme

denek = [col for col in final_data_train if col not in df_get_dummies and final_data_train[col].dtype != "datetime64[ns]" ]

skewness = final_data_train[denek].skew()
print(denek)
print(skewness)

yüksek_çarpıklık = skewness[abs(skewness) > 1].index.tolist()
print(f"\nYüksek Çarpıklığa Sahip Sütun Sayısı (|skew| > 1): {len(yüksek_çarpıklık)}")
print("Yüksek Çarpık Sütunlar:")
print(yüksek_çarpıklık)

cols_to_log_transform = [col for col in yüksek_çarpıklık if skewness[col] > 1]
for col in cols_to_log_transform:
    final_data_train[col] = np.log1p(final_data_train[col])

    if col in final_data_test.columns:
        final_data_test[col] = np.log1p(final_data_test[col])


#veriyi train test olarak bölme

cols_to_drop = ["cust_id","ref_date"]

X_train = final_data_train.drop(cols_to_drop, axis = 1)
y_train = df_referance["churn"]

X_test = final_data_test.drop(cols_to_drop, axis = 1)
X_test = X_test[X_train.columns] #sütun hizalama


#korelasyon analizi

corr_matrix=X_train.corr()
esik_deger = 0.9
corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) #matrixi üst üçgenini alıyoruz tekrarlılardan kurtulmak için
high_corr = corr_matrix_upper.stack()[corr_matrix_upper.stack().abs() > esik_deger]
high_corr = high_corr.reset_index()
high_corr.columns = ['degisken_1', 'degisken_2', 'Korelasyon']

print(high_corr[high_corr["degisken_1"] != high_corr["degisken_2"]].sort_values(by="Korelasyon",ascending=False))




###################################################################################

def recall_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]

    tp_at_k = y_true[top].sum()
    P = y_true.sum()

    return float(tp_at_k / P) if P > 0 else 0.0


def lift_at_k(y_true, y_prob, k=0.1):
    """
    Tahmin edilen olasılıkların en üst k%'sını pozitif etiketleyerek lift (precision/prevalence) değerini hesaplar.

    Parametreler:
        y_true (list): Gerçek ikili etiketler.
        y_prob (list): Tahmin edilen olasılıklar.
        k (float): Pozitif etiketlenecek olasılıkların yüzdelik dilimi (varsayılan 0.1).

    Döndürür:
        float: En iyi k% tahminlerindeki lift değeri.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]

    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()

    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0


def convert_auc_to_gini(auc):
    """
    ROC AUC skorunu Gini katsayısına dönüştürür.

    Gini katsayısı, ROC AUC skorunun doğrusal bir dönüşümüdür.

    Parametreler:
        auc (float): ROC AUC skoru (0 ile 1 arasında).

    Döndürür:
        float: Gini katsayısı (-1 ile 1 arasında).
    """
    return 2 * auc - 1


def ing_hubs_datathon_metric(y_true, y_prob):
    """
    Gini, recall@10% ve lift@10% metriklerini birleştiren özel bir metrik hesaplar.

    Metrik, her bir skoru bir baseline modelin metrik değerlerine göre oranlar ve aşağıdaki ağırlıkları uygular:
    - Gini: %40
    - Recall@10%: %30
    - Lift@10%: %30

    Parametreler:
        y_true (list): Gerçek ikili etiketler.
        y_prob (list): Tahmin edilen olasılıklar.

    Döndürür:
        float: Ağırlıklandırılmış bileşik skor.
    """
    # final metrik için ağırlıklar
    score_weights = {
        "gini": 0.4,
        "recall_at_10perc": 0.3,
        "lift_at_10perc": 0.3,
    }

    # baseline modelin her bir metrik için değerleri
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }

    # y_prob tahminleri için metriklerin hesaplanması
    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)

    new_scores = {
        "roc_auc": roc_auc,
        "recall_at_10perc": recall_at_10perc,
        "lift_at_10perc": lift_at_10perc,
    }

    # roc auc değerlerinin gini değerine dönüştürülmesi
    baseline_scores["gini"] = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_scores["gini"] = convert_auc_to_gini(new_scores["roc_auc"])

    # baseline modeline oranlama
    final_gini_score = new_scores["gini"] / baseline_scores["gini"]
    final_recall_score = new_scores["recall_at_10perc"] / baseline_scores["recall_at_10perc"]
    final_lift_score = new_scores["lift_at_10perc"] / baseline_scores["lift_at_10perc"]

    # ağırlıklandırılmış metriğin hesaplanması
    final_score = (
        final_gini_score * score_weights["gini"] +
        final_recall_score * score_weights["recall_at_10perc"] +
        final_lift_score * score_weights["lift_at_10perc"]
    )
    return final_score



########################################################################################################
#MODELLEME VE OPTİMİZASYON
################################################################################################
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc  # Bellek yönetimi için
import optuna

try:
    df_referance_test = pd.read_csv("referance_data_test.csv")
except FileNotFoundError:
    print("HATA: referance_data_test.csv bulunamadı!")
    exit()

print(f"Modelleme için X_train shape: {X_train.shape}")
print(f"Modelleme için y_train shape: {y_train.shape}")
print(f"Modelleme için X_test shape: {X_test.shape}")

# ----- Optimizasyon Öncesi Skoru Kaydet (Opsiyonel)-----
oof_score_before_hpo = None  # Şimdilik None bırakalım


# ----- Optuna Objective Fonksiyonu (LGBM) -----
def objective_lgbm(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'random_state': 42, 'n_jobs': -1,
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500, step=100),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 10)
    }

    N_FOLDS_OPT = 5
    # DÜZELTME: random_state'i sabit tutalım
    skf_opt = StratifiedKFold(n_splits=N_FOLDS_OPT, shuffle=True, random_state=42)
    oof_preds_opt = np.zeros(X_train.shape[0])
    for fold, (train_idx, val_idx) in enumerate(skf_opt.split(X_train, y_train)):
        # ... (CV döngüsü öncekiyle aynı) ...
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        val_preds_proba = model.predict_proba(X_val_fold)[:, 1]
        if np.isnan(val_preds_proba).any(): return 0.0
        oof_preds_opt[val_idx] = val_preds_proba
        del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model
        gc.collect()
    if np.isnan(oof_preds_opt).any(): return 0.0
    score = ing_hubs_datathon_metric(y_train, oof_preds_opt)
    return score


# ----- Optimizasyon Çalışmasını Başlat (LGBM) -----
study_lgbm = optuna.create_study(direction='maximize', study_name='LGBM Optimization')
N_TRIALS_LGBM = 50  # Deneme sayısı
print(f"\n--- Optuna LightGBM Optimizasyonu Başlıyor ({N_TRIALS_LGBM} deneme) ---")
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS_LGBM)

# ----- Sonuçları Değerlendir (LGBM) -----
print("\n--- LightGBM Optimizasyon Tamamlandı ---")
best_score_lgbm = study_lgbm.best_value
best_params_lgbm = study_lgbm.best_params
print(f"En İyi Deneme Skoru (Özel Metrik): {best_score_lgbm:.5f}")
print("En İyi Parametreler (LGBM):")
print(best_params_lgbm)


# ... (Karşılaştırma ve baseline geçme kontrolü önceki kodda olduğu gibi) ...

# ----- XGBoost için Optuna Objective Fonksiyonu -----
def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'use_label_encoder': False, 'random_state': 42, 'nthread': -1,
        #'tree_method':'gpu_hist', 'gpu_id': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'early_stopping_rounds' : 50

    }
    N_FOLDS_OPT = 5
    # DÜZELTME: random_state'i sabit tutalım
    skf_opt = StratifiedKFold(n_splits=N_FOLDS_OPT, shuffle=True, random_state=42)
    oof_preds_opt = np.zeros(X_train.shape[0])
    for fold, (train_idx, val_idx) in enumerate(skf_opt.split(X_train, y_train)):
        # ... (CV döngüsü öncekiyle aynı) ...
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False)
        val_preds_proba = model.predict_proba(X_val_fold)[:, 1]
        if np.isnan(val_preds_proba).any(): return 0.0
        oof_preds_opt[val_idx] = val_preds_proba
        del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model
        gc.collect()
    if np.isnan(oof_preds_opt).any(): return 0.0
    score = ing_hubs_datathon_metric(y_train, oof_preds_opt)
    return score


# ----- XGBoost Optimizasyon Çalışmasını Başlat -----
study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost Optimization')
N_TRIALS_XGB = 50  # Deneme sayısı
print(f"\n--- Optuna XGBoost Optimizasyonu Başlıyor ({N_TRIALS_XGB} deneme) ---")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB)

# ----- Sonuçları Değerlendir (XGBoost) -----
print("\n--- XGBoost Optimizasyon Tamamlandı ---")
best_score_xgb = study_xgb.best_value
best_params_xgb = study_xgb.best_params
print(f"En İyi Deneme Skoru (Özel Metrik): {best_score_xgb:.5f}")
print("En İyi Parametreler (XGBoost):")
print(best_params_xgb)

# ----- MODEL SEÇİMİ -----
print(f"\n--- Model Karşılaştırma ve Seçim ---")
print(f"Optimize LGBM Skoru (Optuna OOF): {best_score_lgbm:.5f}")
print(f"Optimize XGBoost Skoru (Optuna OOF): {best_score_xgb:.5f}")

if best_score_xgb > best_score_lgbm:
    print("Optimize XGBoost daha iyi görünüyor. XGBoost parametreleri kullanılacak.")
    final_best_params = best_params_xgb
    final_model_type = 'XGBoost'
    # Final CV ve Tahminler için XGBoost kullanılacak
else:
    print("Optimize LightGBM daha iyi veya eşit görünüyor. LGBM parametreleri kullanılacak.")
    final_best_params = best_params_lgbm
    final_model_type = 'LightGBM'
    # Final CV ve Tahminler için LightGBM kullanılacak

# ----- EN İYİ MODEL İLE FİNAL ÇAPRAZ DOĞRULAMA -----

N_FOLDS_FINAL = 5
skf_final = StratifiedKFold(n_splits=N_FOLDS_FINAL, shuffle=True, random_state=123)  # Farklı RS

oof_predictions_final = np.zeros(X_train.shape[0])
cv_scores_final = []
feature_importances_final = pd.DataFrame(index=X_train.columns)  # Önemleri toplamak için
#test_predictions_final = np.zeros(X_test.shape[0])  # Test tahminlerini toplamak için (ortalama alacağız)
test_predictions_list = []

print(f"\n--- {final_model_type} ile {N_FOLDS_FINAL}-Kat Final Çapraz Doğrulama Başlıyor ---")
print("Kullanılacak Parametreler:")
print(final_best_params)

for fold, (train_index, val_index) in enumerate(skf_final.split(X_train, y_train)):
    print(f"\n===== Final Fold {fold + 1}/{N_FOLDS_FINAL} =====")
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Seçilen modele göre modeli başlat
    if final_model_type == 'LightGBM':
        model_final = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, n_jobs=-1,
                                         **final_best_params)
        model_final.fit(X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    elif final_model_type == 'XGBoost':
        model_final = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                        use_label_encoder=False, random_state=42, nthread=-1,
                                        # tree_method='gpu_hist', gpu_id=0, # GPU için
                                        **final_best_params)
        model_final.fit(X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        eval_metric='auc',
                        early_stopping_rounds=100,
                        verbose=False)
    else:
        raise ValueError("Geçersiz model tipi seçildi!")

    # Tahminler
    val_preds_proba = model_final.predict_proba(X_val_fold)[:, 1]
    oof_predictions_final[val_index] = val_preds_proba

    test_fold_preds = model_final.predict_proba(X_test)[:, 1]
    # DÜZELTME: Test tahminlerini doğrudan toplamak yerine listeye ekleyip sonra ortalama alacağız
    # test_predictions_final += test_fold_preds / N_FOLDS_FINAL
    test_predictions_list.append(test_fold_preds)  # Tekrar tanımlanan listeyi kullan

    # Skorlama
    fold_score = ing_hubs_datathon_metric(y_val_fold, val_preds_proba)
    cv_scores_final.append(fold_score)
    print(f"Final Fold {fold + 1} Özel Metrik Skoru: {fold_score:.5f}")

    # Özellik Önemleri
    feature_importances_final[f'Fold_{fold + 1}'] = model_final.feature_importances_

    del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model_final
    gc.collect()

# --- Final CV Sonu ---
mean_final_cv_score = np.mean(cv_scores_final)
print(f"\n--- Final Çapraz Doğrulama Tamamlandı ({final_model_type}) ---")
print(f"Kat Skorları: {[f'{s:.5f}' for s in cv_scores_final]}")
print(f"Ortalama Özel Metrik Skoru (Final CV): {mean_final_cv_score:.5f}")

overall_oof_final_score = ing_hubs_datathon_metric(y_train, oof_predictions_final)
print(f"Genel OOF Özel Metrik Skoru (Final CV): {overall_oof_final_score:.5f}")

# --- SKOR KARŞILAŞTIRMA (GÖNDERMEDEN ÖNCE) ---
print("\n--- Final Model Skorlarının Baseline ile Karşılaştırılması ---")

# Baseline değerleri
baseline_scores_comp = {
    "Gini": 0.38515,
    "Recall@10%": 0.18469,
    "Lift@10%": 1.84715
}
baseline_auc_comp = 0.69257  # Gini = 2*AUC - 1

# Final modelin OOF tahminleri üzerinden bireysel metrikleri hesapla
final_oof_auc = roc_auc_score(y_train, oof_predictions_final)
final_oof_gini = convert_auc_to_gini(final_oof_auc)
final_oof_recall10 = recall_at_k(y_train, oof_predictions_final, k=0.1)
final_oof_lift10 = lift_at_k(y_train, oof_predictions_final, k=0.1)
final_weighted_score = overall_oof_final_score  # Zaten hesaplamıştık

print(f"Metrik          | Baseline Değeri | Model Skoru (OOF) | Başarı Durumu")
print(f"----------------|-----------------|-------------------|--------------")
print(
    f"Gini            | {baseline_scores_comp['Gini']:.5f}       | {final_oof_gini:.5f}        | {'BAŞARILI' if final_oof_gini > baseline_scores_comp['Gini'] else 'BAŞARISIZ'}")
print(
    f"Recall@10%      | {baseline_scores_comp['Recall@10%']:.5f}       | {final_oof_recall10:.5f}        | {'BAŞARILI' if final_oof_recall10 > baseline_scores_comp['Recall@10%'] else 'BAŞARISIZ'}")
print(
    f"Lift@10%        | {baseline_scores_comp['Lift@10%']:.5f}       | {final_oof_lift10:.5f}        | {'BAŞARILI' if final_oof_lift10 > baseline_scores_comp['Lift@10%'] else 'BAŞARISIZ'}")
print(f"----------------|-----------------|-------------------|--------------")
print(
    f"Ağırlıklı Skor  | 1.00000         | {final_weighted_score:.5f}        | {'BAŞARILI' if final_weighted_score > 1.0 else 'BAŞARISIZ'}")
print(f"(Ayrıca AUC     | {baseline_auc_comp:.5f}       | {final_oof_auc:.5f}        | )")

# ----- ÖZELLİK ÖNEM SIRALAMASI (Final Model) -----
feature_importances_final['mean'] = feature_importances_final.mean(axis=1)
feature_importances_final.sort_values('mean', ascending=False, inplace=True)

N = 40  # Gösterilecek özellik sayısı
print(f"\n--- Ortalama En Önemli {N} Özellik (Final Model: {final_model_type}) ---")
print(feature_importances_final[['mean']].head(N))

# ----- TEST SETİ İÇİN FİNAL TAHMİN -----
# DÜZELTME: test_predictions_list'i kullanarak ortalamayı al
mean_test_predictions_final = np.mean(test_predictions_list, axis=0)
#################################################################################

cols_to_drop = ["cust_id","ref_date_x"]
xx_train = final_data_train.merge(df_referance,how="left",on="cust_id")
xx_train.drop(cols_to_drop, axis = 1)

#######################################################################################################

from autogluon.tabular import TabularPredictor

ag_label = "churn"  # Hedef değişkenimiz
ag_problem_type = 'binary'  # Problem tipi: İkili sınıflandırma

ag_eval_metric = 'roc_auc'
ag_model_path = "./autogluon_ing_churn_model"  # Modellerin kaydedileceği klasör
ag_time_limit = 1800   # Saniye cinsinden süre limiti (Örn: 2 saat). Zamanına göre ayarla.
# Preset'ler: 'medium_quality' (hızlı), 'high_quality' (dengeli), 'best_quality' (yavaş, en kapsamlı)
ag_presets = 'high_quality'  # İyi bir başlangıç noktası olabilir

# AutoGluon için Veri Hazırlığı
# AutoGluon kendi ön işlemesini yapsa da, senin hazırladığın temiz veriyle başlamak iyi olur.
# Ölçekleme AutoGluon tarafından yapılacağı için ölçeklenmemiş veriyi kullanabiliriz.
# Ancak log dönüşümü yapılmış veriyi vermek genellikle faydalıdır.
# Anahtar sütunları (cust_id, ref_date) çıkaralım.
col = ['cust_id', 'ref_date_x', 'ref_date_y']
xx_train = xx_train.drop(col, axis = 1)

ag_train_data = xx_train
ag_test_data = X_test



print(f"AutoGluon Eğitim Verisi Boyutu: {ag_train_data.shape}")
print(f"AutoGluon Test Verisi Boyutu: {ag_test_data.shape}")

# ----- AutoGluon Eğitimi -----
print(f"\n--- AutoGluon Eğitimi Başlıyor (Süre Limiti: {ag_time_limit} saniye) ---")
ag_predictor = TabularPredictor(
    label=ag_label,
    problem_type=ag_problem_type,
    eval_metric=ag_eval_metric,
    path=ag_model_path
).fit(
    train_data=ag_train_data,  # Eğitim verisi (churn sütununu içerir)
    presets=ag_presets,
    time_limit=ag_time_limit,
    # AutoGluon'un K-Fold CV yapmasını sağlamak için (eğer train_data yeterince büyükse otomatik yapar):
    # num_bag_folds=5, # İstenen kat sayısı
    # num_stack_levels=1, # Stacking katmanı sayısı (genellikle 1 yeterli)
    # num_bag_sets=1 # Bagging seti sayısı
)

# ----- AutoGluon Değerlendirme -----
print("\n--- AutoGluon Leaderboard ---")
# Leaderboard, AutoGluon'un denediği modellerin doğrulama skorlarını gösterir
ag_leaderboard = ag_predictor.leaderboard(ag_train_data, silent=False)
# Not: ag_train_data vermek, AutoGluon'un fit sırasında ayırdığı iç doğrulama setindeki skorları gösterir.

# En iyi modelin özellik önemlerini alalım
print("\n--- AutoGluon Feature Importance ---")
ag_feature_importance = ag_predictor.feature_importance(data=ag_train_data)
print(ag_feature_importance)

# Yarışma metriğini OOF tahminleri üzerinden hesaplamayı deneyelim
# AutoGluon OOF tahminlerini genellikle fit sırasında hesaplar ve predictor objesi üzerinden erişilebilir
try:
    oof_predictions_ag = ag_predictor.predict_proba(ag_train_data, as_multiclass=False)  # Pozitif sınıf olasılıkları
    y_true_ag = ag_train_data[ag_label]
    custom_score_ag = ing_hubs_datathon_metric(y_true_ag, oof_predictions_ag)
    print(f"\nAutoGluon OOF Özel Metrik Skoru: {custom_score_ag:.5f}")

    # Kendi optimize ettiğin modelle karşılaştır
    print(f"Optimize LGBM Skoru (Optuna OOF): {best_score_lgbm:.5f}")  # Önceki adımdan
    print(f"Optimize XGBoost Skoru (Optuna OOF): {best_score_xgb:.5f}")  # Önceki adımdan

except Exception as e:
    print(f"\nAutoGluon OOF özel metrik skoru hesaplanamadı: {e}")

# ----- AutoGluon Tahmin ve Gönderim -----
print("\n--- AutoGluon Test Seti Tahmini ---")
# Pozitif sınıfın (churn=1) olasılıklarını al
ag_test_pred_proba = ag_predictor.predict_proba(ag_test_data, as_multiclass=False)

print("AutoGluon tahminleri oluşturuldu.")

ag_submission = pd.DataFrame({
    'cust_id': df_referance_test['cust_id'],  # Orijinal test referansından cust_id
    'churn': ag_test_pred_proba  # İstenen sütun adı
})

ag_submission_file = "submission_autogluon.csv"
ag_submission.to_csv(ag_submission_file, index=False)
print(f"\nAutoGluon gönderim dosyası '{ag_submission_file}' kaydedildi.")
print(ag_submission.head())


#################################################################################
#submission dosyası oluşturma

print("\n--- Gönderim Dosyası Oluşturuluyor ---")

submission_df_final = pd.DataFrame({'cust_id': df_referance_test['cust_id']})
# DÜZELTME: Sütun adını 'churn_probability' yap
submission_df_final['churn'] = mean_test_predictions_final
# DÜZELTME: Dosya adını daha açıklayıcı yap
submission_file_final = f"submission_{final_model_type}.csv"
submission_df_final.to_csv(submission_file_final, index=False)

print(f"Final gönderim dosyası '{submission_file_final}' olarak kaydedildi.")
print(submission_df_final.head())







