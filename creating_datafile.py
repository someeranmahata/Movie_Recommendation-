import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(max_features = 5000, stop_words = 'english')
mov = pickle.load(open("movies_details.pkl", "rb"))
movies = pd.DataFrame(mov)
vectors = cv.fit_transform(movies['tags']).toarray()

similar = cosine_similarity(vectors)
pickle.dump(vectors, open("vectors.pkl", "wb"))