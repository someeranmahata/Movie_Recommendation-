import pickle
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
import streamlit as st

# DATA
load_dotenv()
api_key = "ff526b0160e62f1c6091c7428704fed5"
cv = CountVectorizer(max_features = 5000, stop_words = 'english')
mov = pickle.load(open("movies_details.pkl", "rb"))
movies = pd.DataFrame(mov)

vectors = cv.fit_transform(movies['tags']).toarray()
#similar = pickle.load(open("vectors.pkl", "rb"))   


# # methods
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        response = requests.get(url)
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return "https://via.placeholder.com/500x750?text=Poster+Not+Found"

    except:
        return "https://via.placeholder.com/500x750?text=Error"


def recommend(query, top_n=5):
    query = query.lower()
    user_ch = stem(query)
    data = cv.transform([user_ch])
    sim = cosine_similarity(data, vectors).flatten()
    top_indices = sim.argsort()[-top_n:][::-1]
    return movies.iloc[top_indices]['title'].tolist()





# # taking input

st.title('MOVIES RECOMMENDATION')
st.text("(You can mention any Movie/Actors or Directors)")

select_input = st.text_input("Enter the type of movie you want to watch : ")
st.text("NOTE : if you are mentioning actors name write it wthout space saperated")

if st.button("Recommend"):
    titles = recommend(select_input)

    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for i in range(5):
        with cols[i]:
            poster = fetch_poster(titles[i])
            st.image(poster)
            st.caption(titles[i])


# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     output = None
#     if request.method == "POST":
#         user_input = request.form.get("user_input")
#         if user_input:
#             output = evaluate_input(user_input)
#     return render_template("index.html", output=output)

# def evaluate_input(a):
#     print("valuating for : ", a)
#     return recommend(a)

# if __name__ == "__main__":
#     app.run(debug=True)