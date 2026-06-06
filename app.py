import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request
# import streamlit as st
import os
# DATA
load_dotenv()

api_key = os.getenv("TMDB_API_KEY")


cv = CountVectorizer(max_features = 5000, stop_words = 'english')
mov = pickle.load(open("movies_details.pkl", "rb"))
# mov = pickle.load(open("movie_list.pkl", "rb"))

movies = pd.DataFrame(mov)


vectors = cv.fit_transform(movies['tags']).toarray()

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
    results = []

    for idx in top_indices:
        title = movies.iloc[idx]['title']

        results.append({
            "title": title,
            "poster": fetch_poster(title)
        })
    print("ouptut for : ", query, "\n", results)
    return results

# recommend("DanielRadcliffe")


# # taking input

# st.title('MOVIES RECOMMENDATION')
# st.text("(You can mention any Movie/Actors or Directors)")

# select_input = st.text_input("Enter the type of movie you want to watch : ")
# st.text("NOTE : if you are mentioning actors name write it wthout space saperated")

# if st.button("Recommend"):
#     titles = recommend(select_input)

#     col1, col2, col3, col4, col5 = st.columns(5)
#     cols = [col1, col2, col3, col4, col5]

#     for i in range(5):
#         with cols[i]:
#             poster = fetch_poster(titles[i])
#             st.image(poster)
#             st.caption(titles[i])


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":

        user_input = request.form.get("user_input")

        if user_input:
            recommendations = recommend(user_input)

    return render_template(
        "index.html",
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)

# while(True):
#     name = input("Enter name:")
#     recommend(name)



'''
# import pickle

import os


api_key = os.getenv("TMDB_API_KEY")



import pickle
import requests

movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

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

def recommend(movie,n = 5):
    matches = movies[
        movies["title"].str.contains(movie, case=False, na=False)
    ]

    if matches.empty:
        print(f"Movie '{movie}' not found")
        return []

    idx = matches.index[0]

    distances = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in distances[1:n + 1]:

        title = movies.iloc[i[0]]["title"]

        recommendations.append({
            "title": title,
            "poster": fetch_poster(title)
        })
    
    return recommendations


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":

        user_input = request.form.get("user_input")

        if user_input:
            recommendations = recommend(user_input)

    return render_template(
        "index.html",
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)

    '''
