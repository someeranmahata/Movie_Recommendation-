import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request
import pandas as pd

load_dotenv()

api_key = os.getenv("TMDB_API_KEY")

cv = CountVectorizer(max_features=5000, stop_words='english')

mov = pickle.load(open("movies_details.pkl", "rb"))
movie_list = pickle.load(open("movie_list.pkl", "rb"))

movies = pd.DataFrame(mov)
movies2 = pd.DataFrame(movie_list)

vectors = cv.fit_transform(movies['tags'])

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

    return results


def recommend2(movie, n=5):

    matches = movies2[
        movies2["title"].str.contains(movie, case=False, na=False)
    ]

    if matches.empty:
        return []

    idx = matches.index[0]

    distances = cosine_similarity(
        vectors[idx],
        vectors
    ).flatten()

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in movie_list[1:n + 1]:

        title = movies2.iloc[i[0]]["title"]

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

        query = request.form.get("user_input")
        search_type = request.form.get("search_type")
        count = int(request.form.get("count", 5))

        if query:

            if search_type == "actor":
                recommendations = recommend(query, count)

            elif search_type == "movie":
                recommendations = recommend2(query, count)

    return render_template(
        "index.html",
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)