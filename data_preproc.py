from pathlib import Path

import pandas as pd
import scipy


class ArtistRetriever:

    def __init__(self):
        self.artists_df = None

    def load_artists(self, data_path):
        artists_df = pd.read_csv(data_path, sep="\t")
        artists_df = artists_df.set_index("id")
        artists_df = artists_df.drop(["url", "pictureURL"], axis=1)
        self.artists_df = artists_df

    def fetch_artist_name(self, artist_id):
        return self.artists_df.loc[artist_id, "name"]


def load_user_artist_plays_matrix(file_path):
    user_artist_interactions = pd.read_csv(file_path, sep="\t")

    user_artist_interactions.set_index(["userID", "artistID"], inplace=True)

    plays_data = user_artist_interactions.weight.astype(float)
    matrix_rows = user_artist_interactions.index.get_level_values(0)
    matrix_columns = user_artist_interactions.index.get_level_values(1)

    coo = scipy.sparse.coo_matrix(
        (
            plays_data, (matrix_rows, matrix_columns)
        )
    )

    return coo

if __name__ == "__main__":

    artist_data_path = "data/artists.dat"
    user_artist_path = "data/user_artists.dat"

    user_artist_plays = pd.read_csv(user_artist_path, sep="\t")
    user_artist_plays.set_index(["userID", "artistID"], inplace=True)

    print("User Artist Interactions - Play Counts")
    print(user_artist_plays)

    print("Unique User Count")
    print(len(user_artist_plays.index.get_level_values(0).unique()))

    print("Unique Artist Count")
    print(len(user_artist_plays.index.get_level_values(1).unique()))

    coo_matrix = load_user_artist_plays_matrix(user_artist_path)
    csr_matrix = coo_matrix.tocsr()

    artist_retriever_instance = ArtistRetriever()
    artist_retriever_instance.load_artists(artist_data_path)

    artist = artist_retriever_instance.fetch_artist_name(815)
    print(artist)
