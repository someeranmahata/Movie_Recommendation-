import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import os
import requests


# DATA
apikey = "ff526b0160e62f1c6091c7428704fed5"
cv = CountVectorizer(max_features = 5000, stop_words = 'english')
mov = pickle.load(open("movies_details.pkl", "rb"))
movies = pd.DataFrame(mov)

vectors = cv.fit_transform(movies['tags']).toarray()
similar = pickle.load(open("vectors.pkl", "rb"))

# CHECK
# print(movies[movies['title'] == 'The Avengers'].index[0])
# print(movies.iloc[16]['title'])
# print(movies.iloc[16]['tags'])



# # methods
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={apikey}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    
    if data['results']:
        poster_path = data['results'][0]['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500" + poster_path
        return full_path
    else:
        return "https://via.placeholder.com/500x750?text=Poster+Not+Found"

def recommend(query, top_n=5):
    query = query.lower()
    user_ch = stem(query)
    data = cv.transform([user_ch])
    sim = cosine_similarity(data, similar).flatten()
    top_indices = sim.argsort()[-top_n:][::-1]
    return movies.iloc[top_indices]['title'].tolist()
    
print(recommend("fantasy danielRadcliffe"))


# # taking input
st.title('MOVIES RECOMMENDATION')
st.text("(You can mention any Movie/Actors or Directors)")

select_input = st.text_input("Enter the type of movie you want to watch : ")
st.text("NOTE : if you are mentioning actors name write it wthout space saperated")

if st.button("Recommend"):
    titles = recommend(select_input)
    posters_path = []

    for i in titles:
        posters_path.append(fetch_poster(i))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(titles[0])
        st.image(posters_path[0])
    with col2:
        st.text(titles[0])
        st.image(posters_path[1])
    with col3:
        st.text(titles[2])
        st.image(posters_path[2])
    with col4:
        st.text(titles[3])
        st.image(posters_path[3])
    with col5:
        st.text(titles[4])
        st.image(posters_path[4])              



