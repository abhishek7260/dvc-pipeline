import pandas as pd
import numpy as np
import  os

train_file = "./data/raw/train.csv"
test_file = "./data/raw/test.csv"

train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)
def remove_outliers(df,columns=None):
    if columns is None:
        columns=df.select_dtypes(include=['number']).columns.tolist()
    for col in columns:
        q1=df[col].quantile(0.25)   
        q3=df[col].quantile(0.75)   
        iqr=q3-q1
        lower=q1-1.5*iqr
        upper=q3+1.5*iqr
        df=df[(df[col]>=lower) & (df[col]<=upper)]
    return df
processes_train_data=remove_outliers(train_data)
processes_test_data=remove_outliers(test_data)
data_path=os.path.join("data","processed")
os.makedirs(data_path,exist_ok=True)
processes_train_data.to_csv(os.path.join(data_path,"process_train.csv"),index=False)
processes_test_data.to_csv(os.path.join(data_path,"process_test.csv"),index=False)
