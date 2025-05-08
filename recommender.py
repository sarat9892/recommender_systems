import implicit
import scipy

from data_preproc import load_user_artist_plays_matrix, ArtistRetriever

class ImplicitRecommender:

    def __init__(self, artist_retriever: ArtistRetriever, implicit_model: implicit.recommender_base.RecommenderBase):
        """
        :param artist_retriever: A class that retrieves the artist from the dataset
        :param implicit_model: An implicit recommender system model
        """

        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artist_matrix: scipy.sparse.csr_matrix):
        self.implicit_model.fit(user_artist_matrix)

    def recommend(self, user_id: int, user_artist_matrix: scipy.sparse.csr_matrix, n: int = 10):
        artist_ids, scores = self.implicit_model.recommend(user_id, user_artist_matrix[user_id], N=n)

        artists = [self.artist_retriever.fetch_artist_name(artist_id) for artist_id in artist_ids]

        return artists, scores


if __name__ == "__main__":

    artist_data_path = "data/artists.dat"
    user_artist_path = "data/user_artists.dat"

    coo_matrix = load_user_artist_plays_matrix(user_artist_path)
    user_artists_matrix = coo_matrix.tocsr()

    artist_retriever_instance = ArtistRetriever()
    artist_retriever_instance.load_artists(artist_data_path)

    # instantiate ALS using implicit
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever_instance, implict_model)
    recommender.fit(user_artists_matrix)
    artists, scores = recommender.recommend(5, user_artists_matrix, n=5)

    # print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")




