import openpyxl
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.pipeline import Pipeline

#Converting training file into csv format
excel = openpyxl.load_workbook("Participants_Data/Data_Train.xlsx")
data = excel.active
col = csv.writer(open("Data_Train.csv", 
                      'w', 
                      newline=""))
for r in data.rows:
    col.writerow([cell.value for cell in r])
  

df = pd.read_csv("Data_Train.csv")
df = df.drop(["Title", "Author", "Synopsis"], axis=1)
#Converting test file into csv format
excel_1 = openpyxl.load_workbook("Participants_Data/Data_Test.xlsx")
data_1 = excel_1.active
col_1 = csv.writer(open("Data_test.csv",
                         'w',
                         newline=""))
                        
for r in data_1.rows:
    col_1.writerow([cell.value for cell in r])

df_1 = pd.read_csv("Data_test.csv")
print(df_1.head)


#Cleaning Reviews
df["Reviews_cleaned"] = df["Reviews"].apply(lambda x:x.split(' ')[0])
df["Reviews_cleaned"] = df["Reviews_cleaned"].astype(float)
df = df.drop([], axis=1)

#Cleaning Ratings
df["Ratings_cleaned"] = df['Ratings'].apply(lambda x:str(x).replace('customer reviews', ''))
df["Ratings_cleaned"] = df["Ratings_cleaned"].apply(lambda x:str(x).replace('customer review', ''))
df["Ratings_cleaned"] = df["Ratings_cleaned"].apply(lambda x:str(x).replace(',', ''))
df["Ratings_cleaned"] = df["Ratings_cleaned"].astype(int)

#Encoding Book Category
LE = LabelEncoder()
df["BookCategory_Encoded"] = LE.fit_transform(df["BookCategory"])

#Dropping all processed columns
df = df.drop(["Genre","Ratings","BookCategory", "Reviews", "Edition"], axis=1)

print(df.head())

#Scaling data and getting X
X = df.drop(["Price"], axis=1)
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)

#Y axis
y = df["Price"]

df.to_csv("Training_values.csv", index=False)#Removes index which might later affect models
#Train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state=12)

print(X_test.shape)
print(X_train)
print(X_train.shape)
def train_ml_model(X_train, y_train, model):
    if model == 'lr':
        
        model = LinearRegression()

    elif model == 'rf':

        model = RandomForestRegressor()

    elif model == 'xg':

        model = XGBRegressor()
    
    elif model == 'hr':
        model = HuberRegressor()

    elif model == 'ra':

        model = RANSACRegressor()

    elif model == 'ts':
        model = TheilSenRegressor()

    model.fit(X_train, np.log(y_train))

    return model

def prediction(training, model):
    training = train_ml_model(X_train, y_train, model)
    predict = training.predict(X_test)
    predict = np.exp(predict)
    return print(predict)

def model_evaluate(model, X_test, y_test):
    
    predictions=model.predict(X_test)

    predictions=np.exp(predictions)

    mse=mean_squared_error(y_test,predictions)

    mae=mean_absolute_error(y_test,predictions)

    mape=mean_absolute_percentage_error(y_test,predictions)

    msle=mean_squared_log_error(y_test,predictions)

    mse=round(mse,2)
    mae=round(mae,2)
    mape=round(mape,2)
    msle=round(msle,2)
    
    return [mse,mae,mape,msle]

model_lr = train_ml_model(X_train, y_train, 'lr')
model_xg = train_ml_model(X_train, y_train, 'xg')
model_rf = train_ml_model(X_train, y_train, 'rf')
model_hr = train_ml_model(X_train, y_train, 'hr')
model_ra = train_ml_model(X_train, y_train, 'ra')
model_ts = train_ml_model(X_train, y_train, 'ts')


lr_eval = model_evaluate(model_lr, X_test, y_test)
xg_eval = model_evaluate(model_xg, X_test, y_test)
rf_eval = model_evaluate(model_rf, X_test, y_test)
hr_eval = model_evaluate(model_hr, X_test, y_test)
ra_eval = model_evaluate(model_ra, X_test, y_test)
ts_eval = model_evaluate(model_ts, X_test, y_test)

print(lr_eval, xg_eval, rf_eval, hr_eval, ra_eval, ts_eval)
prediction(train_ml_model, 'xg')