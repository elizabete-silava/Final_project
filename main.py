import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data
credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")

# Merge datasets
movies_df = movies_df.merge(credits_df, on="title")
movies_df = movies_df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# Drop missing values
movies_df.dropna(inplace=True)

# Convert strings to lists
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i["name"])
    return l

movies_df["genres"] = movies_df["genres"].apply(convert)
movies_df["keywords"] = movies_df["keywords"].apply(convert)

def convert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i["name"])
            counter += 1
        else:
            break
    return l

movies_df["cast"] = movies_df["cast"].apply(convert3)

def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            l.append(i["name"])
    return l

movies_df["crew"] = movies_df["crew"].apply(fetch_director)

# Clean data
movies_df["overview"] = movies_df["overview"].apply(lambda x: x.split())
movies_df["genres"] = movies_df["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df["keywords"] = movies_df["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df["cast"] = movies_df["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df["crew"] = movies_df["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

# Combine tags
movies_df["tags"] = movies_df["overview"] + movies_df["genres"] + movies_df["keywords"] + movies_df["cast"] + movies_df["crew"]

# Convert tags to lowercase and join them
new_df = movies_df[["movie_id", "title", "tags"]]
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_df['tags'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend(movie_title, cosine_sim=cosine_sim):
    idx = new_df[new_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return new_df['title'].iloc[movie_indices]

# Test the recommendation system
movie_title = "Avatar"
print("Movies similar to", movie_title, ":")
print(recommend(movie_title))





