
import matplotlib.pyplot as plt

import pandas as pd

file_path = "nyka_top_brands_cosmetics_product_reviews_cleaned.csv"
df = pd.read_csv(file_path)

colors = ["pink", "lightgreen", "yellow", "blue","purple","red","orange","cyan","magenta","lime"]
#Histogramas 
#1.visualizar como se distribuyen las calificaciones de las reseñas
#2.visualizar como se distribuyen los product_rating
#3.visualizar cómo se distribuyen los precios de los productos

num_columns = [c for c in ["review_rating", "product_rating", "price"] if c in df.columns]
for i,col in enumerate(num_columns):
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=30, edgecolor="black",color=colors[i % len(colors)])
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel(f"Frecuencia de {col}")
    plt.tight_layout()
    plt.show()
 

#Boxplots por marca

#1.visualizar la distribución de ratings por marca (solo las Top 5).
#2.visualizar la distribución de product_rating por marca (solo las Top 5).
#3.visualizar la distribución de precios por marca (solo las Top 5)

if "brand_name" in df.columns and num_columns:
    
    top_brands = df["brand_name"].value_counts().head(5).index
    for col in [c for c in num_columns if c != "len_words"]: 
        plt.figure(figsize=(8,4))
        df[df["brand_name"].isin(top_brands)].boxplot(column=col, by="brand_name", grid=False)
        plt.title(f"Boxplot de {col} por marca (Top {5})")
        plt.suptitle("")  
        plt.xlabel("Marca")
        plt.ylabel(col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


#Gráficas de pastel
#1.visualizar la proporción de usuarios pro_user
#2.visualizar la proporción de usuarios is_a_buyer
#3.visualizar la proporción de las marcas Top 5 

pie_cols = []

if "pro_user" in df.columns: pie_cols.append("pro_user")
if "is_a_buyer" in df.columns: pie_cols.append("is_a_buyer")
if "brand_name" in df.columns: pie_cols.append("brand_name")

for i,col in enumerate(pie_cols):
    counts = df[col].value_counts(dropna=False)
    if col == "brand_name":
        counts = counts.head(5)  
    plt.figure()
    counts.plot(kind="pie", autopct="%1.1f%%", startangle=90,colors=colors[:len(counts)])
    plt.title(f"Distribución de {col}" + (" (Top 5)" if col=="brand_name" else ""))
    plt.ylabel("")  
    plt.tight_layout()
    plt.show()
    
#Scatter plots
#1.price vs product_rating
#2.price vs review_rating
#3.mrp vs price

pairs = []
if {"price", "product_rating"}.issubset(df.columns): pairs.append(("price", "product_rating"))
if {"price", "review_rating"}.issubset(df.columns): pairs.append(("price", "review_rating"))
if {"mrp", "price"}.issubset(df.columns): pairs.append(("mrp", "price"))

for i,(x, y) in enumerate(pairs):
    plt.figure()
    plt.scatter(df[x], df[y], alpha=0.3,color=colors[i % len(colors)])
    plt.title(f"Scatter: {x} VS {y}")
    plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout()
    plt.show()
    

#Bar plots
# Top 10 marcas por número de reseñas                               
if "brand_name" in df.columns:
    plt.figure(figsize=(9,4))
    tc = df["brand_name"].value_counts().head(10)
    colorss = [colors[i % len(colors)] for i in range(len(tc))]
    tc.plot(kind="bar", color=colorss, edgecolor="black")
    plt.title("Top 10 marcas por número de reseñas")
    plt.xlabel("Marca"); plt.ylabel("Conteo")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
   

    
