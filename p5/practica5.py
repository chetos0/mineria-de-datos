import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

df = pd.read_csv("nyka_top_brands_cosmetics_product_reviews_cleaned.csv")   

x = df[["mrp","product_rating"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train) 

#predicciones
y_pred = model.predict(X_test)      
r2 = r2_score(y_test, y_pred)
print(f"R² score: {r2:.3f}")
m_s_e = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {m_s_e:.3f}")
#sns.scatterplot( x=y_test,y=y_pred)
#print(mean_squared_error(y_test, y_pred))

#ecuacion estadistica
intercept = model.intercept_
coefficients = model.coef_

print(f"Coeficientes Bo : {intercept:.2f}")
print(f"Coeficientes B1 : {coefficients[0]:.4f}")
print(f"Coeficientes B2 : {coefficients[1]:.4f}")

ecuacion = f"Price = {intercept:.2f} + ({coefficients[0]}   * mrp ) + {coefficients[1]:.4f} * product_rating)"
print("Ecuacion estadistica:\n", ecuacion)


#graficas

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
plt.title(f"Prediccion de Precios (R²={r2:.3f})")
plt.xlabel("Precio Real"); plt.ylabel("Precio Predicho")
plt.tight_layout()
plt.grid(True)
plt.show()

residuo = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuo, alpha=0.5, color="pink")
plt.axhline(0, color="brown", linestyle="--")
plt.title("Residuos vs Precios Predichos")
plt.xlabel("Precios Predichos"); plt.ylabel("Residuos")
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure()
sns.heatmap(df[["mrp","product_rating","price"]].corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()