import pandas as pd
import numpy as np     
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, silhouette_score
from sklearn.cluster import KMeans

df = pd.read_csv("nyka_top_brands_cosmetics_product_reviews_cleaned.csv")

cols = ["mrp", "product_rating", "price","review_rating"]
df [cols] = df[cols].apply(pd.to_numeric, errors="coerce")  
df = df.dropna(subset=cols )

X= df [cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#vista antees de organizar

plt.figure(figsize=(10,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], alpha=0.5, color="green")
plt.title("Datos sin Organizar")
plt.xlabel("mrp"); plt.ylabel("product_rating")
plt.grid(True)
plt.show()


#metodo elbow

inertia = []    
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_values, inertia, marker='o')
plt.title("Método Elbow para Selección de K")
plt.xlabel("Número de Clusters K")
plt.ylabel("Inercia")   
plt.grid(True)
plt.show()


#aplicar kmeans con k=4, en la grafica aparece como buen punto de corte
kmeans_final = KMeans(n_clusters=4, random_state=42)

clusters = kmeans_final.fit_predict(X_scaled)
centroids = kmeans_final.cluster_centers_

df['Cluster'] = clusters

#visualizacion de los clusters
plt.figure(figsize=(8,6))

colors = ['pink', 'green', 'yellow','blue']

sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=df['Cluster'], palette=colors, alpha=0.6)
plt.scatter(centroids[:,0], centroids[:,1], s=300, c='red', marker='X', label='Centroides')
plt.title("Clusters de Productos Cosméticos")
plt.xlabel("mrp")
plt.ylabel("product_rating")
plt.legend()
plt.grid(True)
plt.show()

#evaluacion/test
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score:.3f}")


#analisis

cluster_summary = df.groupby('Cluster')[cols].mean().round(2)
print("\nResumen de Clusters:")
print(cluster_summary)


