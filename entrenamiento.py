# importar libreria pandas para manejar datasets
import pandas as pd

# importar sklearn para la regresion lineal y para la configuracion del entrenamiento
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# metricas para la valoracion de las predicciones del modelo
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# para poder enviar y cargar  el archivo .pkl
import pickle




# creamos un data set con los datos del csv
df = pd.read_csv("datos_helados.csv",sep=";")

# limpiar nombres de columnas
df.columns = df.columns.str.strip().str.lower()
x = df[["temperatura"]]
y = df["ventas"]

print(df)
# configuracion del entrenamiento-test
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.2, random_state=42)

# crear modelo de regresion lineal
modelo = LinearRegression()

# entrenar modelo
modelo.fit(x_train,y_train)

# prediccion
predict = modelo.predict(x_test)


# imprimir las medias 

mae = mean_absolute_error(y_test,predict)
mse = mean_squared_error(y_test,predict)
r2 = r2_score(y_test,predict)

print("\nMetricas Analisis de la prediccion.")
print("\nMean Absolute Error: ", mae)
print("\nMean Squared Error: ", mse)
print("\nR2_Score: ", r2)


# exportar el modelo.pkl
with open ("datos_helados.pkl","wb") as f:
    pickle.dump(modelo,f)

print("Modelo de entrenamiento pkl creado correctamente.")

