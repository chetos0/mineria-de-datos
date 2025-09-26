import pandas as pd
import numpy as np

file_path = "nyka_top_brands_cosmetics_product_reviews_cleaned.csv"
df = pd.read_csv(file_path)


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


