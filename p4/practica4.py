import pandas as pd
import scipy.stats as stats
from scipy.stats import kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations

file = "nyka_top_brands_cosmetics_product_reviews_cleaned.csv"

df = pd.read_csv(file)
'''
print(df.head())
print(df.columns)
print(df.dtypes)

print(df["brand_name"].unique())
print(df["brand_name"].value_counts())

brand_counts = df["brand_name"].value_counts()
valid_brands = brand_counts[brand_counts > 6000].index

print(valid_brands)
'''

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