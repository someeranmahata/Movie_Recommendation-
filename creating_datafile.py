import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

mov = pickle.load(open("movies_details.pkl", "rb"))
movies = pd.DataFrame(mov)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])

pickle.dump(vectors, open("vectors.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))