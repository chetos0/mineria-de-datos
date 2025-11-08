import pandas as pd
import numpy as np     
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
 

df = pd.read_csv("nyka_top_brands_cosmetics_product_reviews_cleaned.csv")

cols = ["mrp", "product_rating", "price","review_rating"]
df [cols] = df[cols].apply(pd.to_numeric, errors="coerce")  
df = df.dropna(subset=cols + ["brand_name"])

topMarcas = 5
top_Marcas = df["brand_name"].value_counts().head(topMarcas).index.tolist()

df_top = df[df["brand_name"].isin(top_Marcas)].copy()

X = df_top[cols].values
y = df_top["brand_name"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#crear knn
Pipeline_steps = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "chebyshev"]  
}

#testing
gridS = GridSearchCV(Pipeline_steps, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
gridS.fit(X_train, y_train)
best_model = gridS.best_estimator_
y_pred = best_model.predict(X_test)
accur = accuracy_score(y_test, y_pred)

#resutados
print(f"Accuracy del modelo KNN: {accur:.3f}")
print("Mejores hiperparámetros:", gridS.best_params_)
print(f"Mejores parámetros: k={gridS.best_params_['knn__n_neighbors']}")
print(f"Métrica de distancia: {gridS.best_params_['knn__metric']}")
print(f"Total de predicciones: {len(y_test)}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

#visualizacion
conf_matrix = confusion_matrix(y_test, y_pred, labels=top_Marcas,normalize="true")
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=top_Marcas, yticklabels=top_Marcas)
plt.title("Matriz de Confusión Normalizada")
plt.xlabel("Marca Predicha")   
plt.ylabel("Marca Real")
plt.tight_layout()
plt.show()


valores_test = [3,5,7,9,11]
vt = []
for val in valores_test:
    mdl = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=val,weights="distance",p=2))])
    s = cross_val_score(mdl, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    vt.append(s.mean())

plt.figure()
plt.plot(valores_test, vt, marker="o", color="green")
plt.title("KNN: Accuracy vs Número de Vecinos")
plt.xlabel("Número de Vecinos (k)")
plt.ylabel("Accuracy Promedio (5Fold CV)")
plt.grid(True)
plt.tight_layout()
plt.show()







