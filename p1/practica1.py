import pandas as pd


reviews_df = pd.read_csv('nyka_top_brands_cosmetics_product_reviews.csv')
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