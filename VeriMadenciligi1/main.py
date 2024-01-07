# Pandas kütüphanesini import ediyoruz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Veri setlerini okuyoruz
df_kitaplar = pd.read_csv(r'C:\Users\lenovo\OneDrive\Masaüstü\Books.csv',low_memory=False)
df_kullanicilar = pd.read_csv(r'C:\Users\lenovo\OneDrive\Masaüstü\User.csv')
df_puanlar = pd.read_csv(r'C:\Users\lenovo\OneDrive\Masaüstü\Ratings.csv')

# Veri setlerini ekrana yazdırıyoruz
print("Kitaplar Veri Seti:")
print(df_kitaplar.head())
print("\nKullanıcılar Veri Seti:")
print(df_kullanicilar.head())
print("\nPuanlar Veri Seti:")
print(df_puanlar.head())

# En aktif 1000 kullanıcıyı ve en çok oylanan 1000 kitabı seçme
top_users = df_puanlar['User-ID'].value_counts().head(1000).index
top_books = df_puanlar['ISBN'].value_counts().head(1000).index

# Veri setini filtreleme
filtered_ratings = df_puanlar[(df_puanlar['User-ID'].isin(top_users)) & (df_puanlar['ISBN'].isin(top_books))]

# Kullanıcı-kitap matrisini oluşturma
user_book_matrix = filtered_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Eğitim veri setindeki sütun isimlerini alma
column_names = user_book_matrix.columns

# Tahmin yapmak için kullanılan veri setting sütun isimlerini güncelleme
user_book_matrix.columns = column_names

# kNN modelini eğitme
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_matrix)



# Bir kullanıcı için kitap önerme
query_index = 1  # Öneri almak istediğiniz kullanıcının ID'si
distances, indices = model_knn.kneighbors(user_book_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Kullanıcı #{0} için kitap önerileri:\n'.format(user_book_matrix.index[query_index]))
    else:
        print('{0}: {1}, mesafe: {2}'.format(i, df_kitaplar.loc[indices.flatten()[i]]['Book-Title'], distances.flatten()[i]))



# Kullanıcı için önerilen kitaplar ve mesafeler
books = [df_kitaplar.loc[indices.flatten()[i]]['Book-Title'] for i in range(1, len(distances.flatten()))]
distances = distances.flatten()[1:]

# Bar plot oluşturma
plt.figure(figsize=(14,8))
plt.barh(books, distances, color='skyblue')
plt.xlabel('Mesafe')
plt.ylabel('Kitaplar')
plt.title('Kullanıcı #{0} için Önerilen Kitaplar'.format(user_book_matrix.index[query_index]))
plt.gca().invert_yaxis()  # Kitapları en yakından en uzaga doğru sırala
plt.show()

























