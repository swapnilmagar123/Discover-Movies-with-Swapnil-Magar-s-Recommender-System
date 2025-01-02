from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the movie DataFrame
with open('movies.pkl', 'rb') as file:
    movies = pickle.load(file)

# Vectorize the tags for similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectorized_tags = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectorized_tags)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return ["Movie not found"]
    
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommendations = [movies.iloc[i[0]].title for i in movie_list]
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    movie_list = movies['title'].tolist()
    if request.method == 'POST':
        movie = request.form['movie']
        recommendations = recommend(movie)
    return render_template('index.html', recommendations=recommendations, movie_list=movie_list)

if __name__ == "__main__":
    app.run(debug=True)
