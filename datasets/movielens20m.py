import os
import logging

from aprec.utils.os_utils import mkdir_p_local, get_dir, console_logging, shell
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.api.catalog import Catalog
from aprec.datasets.download_file import download_file
from requests.exceptions import ConnectionError

DATASET_NAME = 'ml-20m'
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/{}.zip".format(DATASET_NAME)
BACKUP_MOVIELENS_URL = "https://web.archive.org/web/20220128015818/https://files.grouplens.org/datasets/movielens/{}.zip".format(DATASET_NAME)
MOVIELENS_DIR = "data/movielens20m"
MOVIELENS_FILE = "movielens.zip"
MOVIELENS_FILE_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR, MOVIELENS_FILE)
MOVIELENS_DIR_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR)
RATINGS_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'ratings.csv')
MOVIES_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'movies.csv')


def extract_movielens_dataset():
    if os.path.isfile(RATINGS_FILE):
        logging.info("movielens dataset is already extracted")
        return
    shell("unzip -o {} -d {}".format(MOVIELENS_FILE_ABSPATH, MOVIELENS_DIR_ABSPATH))
    dataset_dir = os.path.join(MOVIELENS_DIR_ABSPATH, DATASET_NAME)
    for filename in os.listdir(dataset_dir):
        shell("mv {} {}".format(os.path.join(dataset_dir, filename), MOVIELENS_DIR_ABSPATH))
    shell("rm -rf {}".format(dataset_dir))


def prepare_data():
    try:
        download_file(MOVIELENS_URL, MOVIELENS_FILE, MOVIELENS_DIR)
    except ConnectionError:
        download_file(BACKUP_MOVIELENS_URL, MOVIELENS_FILE, MOVIELENS_DIR)

    extract_movielens_dataset()


def get_movielens20m_actions(min_rating=4.0):
    prepare_data()
    catalog, movie_genres, movie_titles = get_movies_catalog()
    with open(RATINGS_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                user_id, movie_id, rating_str, timestamp_str = line.strip().split(',')
                rating = float(rating_str)
                timestamp = int(timestamp_str)
                if rating >= min_rating:
                    genres = movie_genres.get(movie_id, [])
                    movie_title = movie_titles.get(movie_id, [])

                    yield Action(user_id, movie_id, timestamp,
                                 {"rating": rating, "genres": genres, "title": movie_title})


def get_movies_catalog():
    prepare_data()
    catalog = Catalog()
    movie_genres = {}
    movie_titles = {}
    with open(MOVIES_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                splits = line.strip().split(",")
                movie_id = splits[0]
                genres = splits[-1].split("|")
                title = ",".join(splits[1:-1]).strip('"')
                item = Item(movie_id).with_title(title).with_tags(genres)
                catalog.add_item(item)
                movie_genres[movie_id] = genres
                movie_titles[movie_id] = title
    return catalog, movie_genres, movie_titles



if __name__ == "__main__":
    console_logging()
    prepare_data()
