from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np


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
