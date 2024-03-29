import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommendationSystem:
    def __init__(self, credits_file, movies_file):
        self.credits_file = credits_file
        self.movies_file = movies_file
        self.movies_df = None
        self.cosine_sim = None
        self.new_df = None
        self._load_data()
        self._process_data()
        self._compute_similarity()

    def _load_data(self):
        self.credits_df = pd.read_csv(self.credits_file)
        self.movies_df = pd.read_csv(self.movies_file)

    def _process_data(self):
        self.movies_df = self.movies_df.merge(self.credits_df, on="title")
        self.movies_df = self.movies_df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
        self.movies_df.dropna(inplace=True)
        self.movies_df["genres"] = self.movies_df["genres"].apply(self._convert)
        self.movies_df["keywords"] = self.movies_df["keywords"].apply(self._convert)
        self.movies_df["cast"] = self.movies_df["cast"].apply(self._convert3)
        self.movies_df["crew"] = self.movies_df["crew"].apply(self._fetch_director)
        self.movies_df["overview"] = self.movies_df["overview"].apply(lambda x: x.split())
        self.movies_df["genres"] = self.movies_df["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies_df["keywords"] = self.movies_df["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies_df["cast"] = self.movies_df["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies_df["crew"] = self.movies_df["crew"].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies_df["tags"] = self.movies_df["overview"] + self.movies_df["genres"] + self.movies_df["keywords"] + self.movies_df["cast"] + self.movies_df["crew"]

        self.new_df = self.movies_df[["movie_id", "title", "tags"]]
        self.new_df["tags"] = self.new_df["tags"].apply(lambda x: " ".join(x))
        self.new_df["tags"] = self.new_df["tags"].apply(lambda x: x.lower())

    def _convert(self, obj):
        l = []
        for i in ast.literal_eval(obj):
            l.append(i["name"])
        return l

    def _convert3(self, obj):
        l = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                l.append(i["name"])
                counter += 1
            else:
                break
        return l

    def _fetch_director(self, obj):
        l = []
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                l.append(i["name"])
        return l

    def _compute_similarity(self):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.new_df['tags'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def recommend(self, movie_title):
        idx = self.new_df[self.new_df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return self.new_df['title'].iloc[movie_indices]

# Test the recommendation system
def main():
    credits_file = "credits.csv"
    movies_file = "movies.csv"
    movie_recommendation_system = MovieRecommendationSystem(credits_file, movies_file)
    movie_title = "Avatar"
    print("Movies similar to", movie_title, ":")
    print(movie_recommendation_system.recommend(movie_title))

if __name__ == "__main__":
    main()
