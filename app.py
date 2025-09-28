import pickle
# import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request




# DATA
load_dotenv()
api_key = os.getenv("apikey")
cv = CountVectorizer(max_features = 5000, stop_words = 'english')
mov = pickle.load(open("movies_details.pkl", "rb"))
movies = pd.DataFrame(mov)

vectors = cv.fit_transform(movies['tags']).toarray()
#similar = pickle.load(open("vectors.pkl", "rb"))   

# CHECK
print(movies.head(0))



# # methods
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
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
    sim = cosine_similarity(data, vectors).flatten()
    top_indices = sim.argsort()[-top_n:][::-1]
    return movies.iloc[top_indices]['title'].tolist()



# # taking input
# st.title('MOVIES RECOMMENDATION')
# st.text("(You can mention any Movie/Actors or Directors)")

# select_input = st.text_input("Enter the type of movie you want to watch : ")
# st.text("NOTE : if you are mentioning actors name write it wthout space saperated")

# if st.button("Recommend"):
#     titles = recommend(select_input)

#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.text(titles[0])
        
#     with col2:
#         st.text(titles[1])
#     with col3:
#         st.text(titles[2])
#     with col4:
#         st.text(titles[3])
#     with col5:
#         st.text(titles[4])

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            output = evaluate_input(user_input)
    return render_template("index.html", output=output)

def evaluate_input(a):
    print("valuating for : ", a)
    return recommend(a)

if __name__ == "__main__":
    app.run(debug=True)