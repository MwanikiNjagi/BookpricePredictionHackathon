from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


df = pd.read_csv("Data_test.csv")
df = df.drop(["Title", "Author", "Synopsis"], axis=1)
df_train = pd.read_csv("Training_values.csv")

#Cleaning Reviews
df["Reviews_cleaned"] = df["Reviews"].apply(lambda x:x.split(' ')[0])
df["Reviews_cleaned"] = df["Reviews_cleaned"].astype(float)
df = df.drop([], axis=1)

#Cleaning Ratings
df["Ratings_cleaned"] = df['Ratings'].apply(lambda x:str(x).replace('customer reviews', ''))
df["Ratings_cleaned"] = df["Ratings_cleaned"].apply(lambda x:str(x).replace('customer review', ''))
df["Ratings_cleaned"] = df["Ratings_cleaned"].apply(lambda x:str(x).replace(',', ''))
df["Ratings_cleaned"] = df["Ratings_cleaned"].astype(int)
print(df["Ratings_cleaned"].head(100))
#Encoding Book Category
LE = LabelEncoder()
df["BookCategory_Encoded"] = LE.fit_transform(df["BookCategory"])

#Dropping all processed columns
df = df.drop(["Genre","Ratings","BookCategory", "Reviews", "Edition"], axis=1)



#Scaling data and getting X
X_test = df

#Scalers are not needed as this is Gradient boosting. Likelihood of getting similar value will be high
#scaler = MinMaxScaler() 
#X_test = scaler.fit_transform(X_test)
print(X_test)
print(X_test.shape)
#Y axis


print(df.head(20))
X_train = df_train.drop(["Price"], axis=1)
y_train = df_train["Price"]
print(X_train)


model = XGBRegressor()
model.fit(X_train, np.log(y_train))

predict = model.predict(X_test)
predict = np.exp(predict)
print(predict) 

df["Price"] = predict
print(df.head())

df.to_excel("Submission.xlsx")
