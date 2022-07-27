# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Importación de los datos
url='https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
df=pd.read_csv(url)

# Transformación de variables
obj=df.select_dtypes('object').columns
df[obj]=df[obj].astype('category')

# Selección de variables explicativas y target
X=df.drop(columns='charges')
y=df['charges']

# Selección muestra de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)

# Tratamiento de las variables
X_train['age_rang']=pd.cut(X_train['age'],bins=[10,20,30,40,50,60,70])
X_train['age_rang']=X_train['age_rang'].astype('category')
X_train=X_train.drop(columns='age')

# Replico para la muestra de prueba
X_test['age_rang']=pd.cut(X_test['age'],bins=[10,20,30,40,50,60,70])
X_test['age_rang']=X_test['age_rang'].astype('category')
X_test=X_test.drop(columns='age')

#Encoding
X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test) # transformo también las variables para la prueba

# Escalamiento de los datos
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train.select_dtypes(['int','float']))
X_test_sc = sc.transform (X_test.select_dtypes(['int','float']))
X_train['bmi']=pd.DataFrame(X_train_sc,columns=['bmi','children'])['bmi']
X_train['children']=pd.DataFrame(X_train_sc,columns=['bmi','children'])['children']
X_test['bmi']=pd.DataFrame(X_test_sc,columns=['bmi','children'])['bmi']
X_test['children']=pd.DataFrame(X_test_sc,columns=['bmi','children'])['children']

# Se eliminan algunas variables que no fueron significativas
X_train_2=X_train.drop(columns=['region_northwest','age_rang_(30, 40]'])
X_test_2=X_test.drop(columns=['region_northwest','age_rang_(30, 40]'])

# Modelo mejorado
lr = LinearRegression()
lr.fit(X_train_2,y_train)
y_pred=lr.predict(X_test_2)

# Se guarda el modelo mejorado
import pickle
filename = '/workspace/Linear-model-Regression/models/finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))

