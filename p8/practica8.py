import pandas as pd
import numpy as np     
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, silhouette_score
from sklearn.cluster import KMeans

#datos csv
df = pd.read_csv("nyka_top_brands_cosmetics_product_reviews_cleaned.csv",parse_dates=['review_date'])


df = df.sort_values(by='review_date')
por_dia = df.groupby("review_date")["product_rating"].mean().reset_index()

por_dia["Tiempo"] = range(1, len(por_dia) + 1)

#Modelo
X = por_dia[["Tiempo"]].values.reshape(-1, 1)
y = por_dia["product_rating"].values

model = LinearRegression()
model.fit(X, y)

futuredays = 30

#Predicciones
future_time = np.array(list(range(len(por_dia) + 1, len(por_dia) + futuredays + 1))
                       ).reshape(-1, 1)


forecasting = model.predict(future_time)

ultima_fecha = por_dia["review_date"].iloc[-1]
fechas_futuras = [ultima_fecha + pd.Timedelta(days=i) for i in range(1, futuredays + 1)]

#Visualización
plt.figure(figsize=(8,8))
plt.scatter(por_dia["Tiempo"], por_dia["product_rating"], color='green', label='Promedio Diario de Product Rating')
plt.plot(por_dia["Tiempo"], model.predict(X), color='blue', label='Línea de Tendencia')
plt.scatter(future_time, forecasting, color='red', label='Predicciones Futuras')
plt.xlabel('Tiempo - Días')
plt.ylabel('Product Rating Promedio Diario')
plt.title('Tendencia y Predicciones Futuras de Product Rating')
plt.legend()
plt.grid(True)
plt.show()

#Resultados
print("Predicciones Futuras de Product Rating para los próximos 30 días:")
for fecha, pred in zip(fechas_futuras, forecasting):
    print(fecha.date(), pred)