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
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
import scipy.stats as stats
from scipy.stats import kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations
from sklearn.metrics import mean_squared_error, r2_score


#Practica 1

def practica1():
    print("Cargando y limpiando datos...")
    file =  "nyka_top_brands_cosmetics_product_reviews.csv"
    print(file)
    reviews_df = pd.read_csv("nyka_top_brands_cosmetics_product_reviews.csv")
    print(reviews_df.head())
    reviews_df.drop_duplicates()
    reviews_df = reviews_df.drop(columns=[ 'review_id','author','product_rating_count','product_tags','review_label','product_url'])
    print(reviews_df.info())
    reviews_df ['review_date']= pd.to_datetime(reviews_df['review_date'], errors='coerce').dt.date
    #reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'], errors='coerce')
    print(reviews_df.info('review_date'))

    reviews_df ['review_text']= reviews_df['review_text'].fillna('No review')
    reviews_df ['review_text']= reviews_df['review_text'].str.lower()
    reviews_df ['review_title']= reviews_df['review_title'].str.lower()
    reviews_df ['brand_name'] = reviews_df['brand_name'].str.lower()
    #reviews_df ['review_title'] = reviews_df['review_title'].str.strip("[""]")

    reviews_df ['review_title'] = reviews_df['review_title'].replace(r'[\U00010000-\U0010ffff]', '', regex=True)
    reviews_df['review_title'] = reviews_df['review_title'].str.replace('"', '').str.replace("'", '')

    reviews_df ['review_text'] = reviews_df['review_text'].replace(r'[\U00010000-\U0010ffff]', '', regex=True)
    reviews_df ['review_rating'] = reviews_df['review_rating'].fillna(reviews_df['review_rating'].median())

    reviews_df.to_csv('nyka_top_brands_cosmetics_product_reviews_cleaned.csv', index=False)

    print("Número de registros:", reviews_df.shape[0])
    print("Columnas finales:", reviews_df.columns.tolist())
    print(reviews_df.info())
    return reviews_df

#Practica 2
def practica2(df,):
    print("si funciona")
# ******************************************RESUMEN GENERAL*************************************

    print("\n\t\t\tESTADISTICA GENERAL")
    general_stats = df.describe(include="all", percentiles=[.05, .25, .5, .75, .95]).T
    print(general_stats.fillna("-"))



    #******************************************POR ENTIDADES**********************************************

    # Por PRODUCTO
    if "product_id" in df.columns:
        print("\n\t\t\tESTADÍSTICAS POR PRODUCTO ")
        print()
        prod_stadistics = (
            df.groupby("product_id").agg({
                **({"review_rating": ["mean","std","min","max","count"]} if "review_rating" in df.columns else {}),
                **({"product_rating": ["mean","std","min","max"]} if "product_rating" in df.columns else {}),
                **({"price": ["mean","std","min","max"]} if "price" in df.columns else {}),
                **({"mrp": ["mean","std","min","max"]} if "mrp" in df.columns else {}),
                **({"len_words": ["mean","std","min","max"]} if "len_words" in df.columns else {})
            })
            .sort_values(("review_rating","count") if "review_rating" in df.columns else ("price","mean"),
                        ascending=False)
        )
        print(prod_stadistics.head(15))


    # Por MARCA

    if "brand_name" in df.columns:
        print("\n\t\t\tESTADÍSTICAS POR MARCA ")
        print()
        brand_stadistics = (
            df.groupby("brand_name").agg({
                **({"review_rating": ["mean","std","min","max","count"]} if "review_rating" in df.columns else {}),
                **({"product_rating": ["mean","std","min","max"]} if "product_rating" in df.columns else {}),
                **({"price": ["mean","std","min","max"]} if "price" in df.columns else {}),
                **({"mrp": ["mean","std","min","max"]} if "mrp" in df.columns else {}),
                **({"len_words": ["mean","std","min","max"]} if "len_words" in df.columns else {})
            }).sort_values(("review_rating","count") if "review_rating" in df.columns else ("price","mean"),
                        ascending=False)
        )
        print(brand_stadistics.head(15))


    #Por USUARIO

    for user in ["is_a_buyer", "pro_user"]:
        if user in df.columns:
            print(f"\n\t\t\tESTADÍSTICAS POR {user} ")
            print()
            user_stadistics = (
                df.groupby(user).agg({
                    **({"review_rating": ["mean","std","min","max","count"]} if "review_rating" in df.columns else {}),
                    **({"product_rating": ["mean","std","min","max"]} if "product_rating" in df.columns else {}),
                    **({"price": ["mean","std","min","max"]} if "price" in df.columns else {}),
                    **({"len_words": ["mean","std","min","max"]} if "len_words" in df.columns else {})
                })
            )
            print(user_stadistics)

    #Marcas o productos con mejor rating

    if "brand_name" in df.columns and "review_rating" in df.columns:
        print("\n\t\t\tTOP MARCAS CON MEJOR RESEÑAS")
        print()
        bn_top = df["brand_name"].value_counts().head(5)
        print(bn_top)

    if "product_id" in df.columns and "review_rating" in df.columns:
        print("\n\t\t\tTOP PRODUCTOS CON MEJOR RESEÑAS ")
        print()
        prod_top = df["product_id"].value_counts().head(5)
        print(prod_top)

#Practica 3
def practica3(df):
    
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

def practica4(df):
    
    #marcas que tienen mas datos para analizar
    df = df[df["brand_name"].isin(["herbal essences", "lakme","kay beauty", "maybelline new york", "nykaa cosmetics"])]

    df["product_rating"] = pd.to_numeric(df["product_rating"], errors="coerce")
    df["brand_name"] = df["brand_name"].astype("category")


    # Análisis de varianza (ANOVA) para product_rating entre marcas
    model = ols("product_rating ~ C(brand_name)", data=df).fit()
    anova1 = sm.stats.anova_lm(model, typ=2)
    print(anova1)

    #Analisis de Kruskal-Wallis para product_rating entre marcas
    groups = [df[df["brand_name"] == brand]["product_rating"].dropna() for brand in df["brand_name"].cat.categories]
    h_statics, p_value = kruskal(*groups)
    print("Kruskal-Wallis H-statistic:", h_statics)
    print("p-value:", p_value)

    print("ANOVA")

    #if anova1["PR(>F)"][0].iloc[0] < 0.05:
    a_anova= anova1["PR(>F)"].iloc[0]
    if a_anova < 0.05:
        print("Hay diferencias significativas entre las marcas.")
    else:
        print("No hay diferencias significativas entre las marcas.")

    print("Kruskal-Wallis")
    if p_value < 0.05:
        print("Hay diferencias significativas entre las marcas.")
    else:
        print("No hay diferencias significativas entre las marcas.")

def practica5(df):
    
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

def practica6(df):
    
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

def practica7(df):
    
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

    colors = ['pink', 'green', 'yellow']

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

def practica8(df):
    
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

def practica9(df):
    
    stopwords = set(STOPWORDS)


    #cargar imagen y modificrla
    img = Image.open("p9/florr.jpg")
    img = img.resize((800, 800))   
    img = img.convert("L")  
    mask = np.array(img)
    mask = 255 - mask  

    #Datos
    df = pd.read_csv("nyka_top_brands_cosmetics_product_reviews_cleaned.csv")

    #Texto
    words = " ".join(df['review_text'].dropna().astype(str))

    #WordCloud
    wordcloud = WordCloud(stopwords=stopwords,mask=mask,width=800,height=800,background_color="black",
                contour_width=2,contour_color="pink",max_words=500,min_font_size=5).generate(words)

    #visualizacion
    plt.figure(figsize=(8, 8), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def main():
    
    df = practica1()
    practica2(df)
    practica3(df)
    #practica4(df)
    #practica5(df)
    #practica6(df)
    #practica7(df)
    #practica8(df)
    #practica9(df)

if __name__ == "__main__":
    main()