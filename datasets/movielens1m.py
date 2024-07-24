import os
import logging
from aprec.api.catalog import Catalog
from aprec.api.item import Item

from aprec.utils.os_utils import get_dir, console_logging, shell
from aprec.api.action import Action
from aprec.datasets.download_file import download_file
from requests.exceptions import ConnectionError

DATASET_NAME = 'ml-1m'
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/{}.zip".format(DATASET_NAME)
MOVIELENS_BACKUP_URL = "https://web.archive.org/web/20220128015818/https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_DIR = "data/movielens1m"
MOVIELENS_FILE = "movielens.zip"
MOVIELENS_FILE_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR, MOVIELENS_FILE)
MOVIELENS_DIR_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR)
RATINGS_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'ratings.dat')
MOVIES_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'movies.dat')


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
        download_file(MOVIELENS_URL,  MOVIELENS_FILE, MOVIELENS_DIR)
    except ConnectionError:
        download_file(MOVIELENS_BACKUP_URL,  MOVIELENS_FILE, MOVIELENS_DIR)
        
    extract_movielens_dataset()


def get_genre_title_dict():
    prepare_data()
    genre_dict = {}
    title_dict = {}
    raw_data = open(MOVIES_FILE, 'rb').read()
    raw_data = raw_data.decode('latin-1')
    for line in raw_data.rstrip("\n").split('\n'):
        movie_id, title, genres = line.strip().split('::')
        genre_dict[movie_id] = genres.split('|')
        title_dict[movie_id] = title
    return genre_dict, title_dict


def get_movielens1m_actions(min_rating=0.0):
    prepare_data()
    catalog, movie_genres, movie_titles = get_movies_catalog()
    actions = []
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



def reproduce_ber4rec_preprocessing():
    min_users_per_item = 5
    item_cnt = {}
    all_actions=get_movielens1m_actions()
    for action in all_actions:
        item_cnt[action.item_id] = item_cnt.get(action.item_id, 0) + 1
    items_filter = set()
    for action in all_actions:
        if item_cnt[action.item_id] >= min_users_per_item:
            items_filter.add(action.item_id)
    actions = list(filter(lambda action: action.item_id in items_filter, all_actions))
    return actions
        
def get_movies_catalog():
    prepare_data()
    catalog = Catalog()  
    movie_genres = {}
    movie_titles = {}
    raw_data = open(MOVIES_FILE, 'rb').read()
    raw_data = raw_data.decode('latin-1')
    for line in raw_data.rstrip("\n").split('\n'):
        movie_id, title, genres_raw = line.strip().split('::')
        genres = genres_raw.split('|')
        item = Item(movie_id).with_title(title).with_tags(genres)
        catalog.add_item(item)
        movie_genres[movie_id] = genres
        movie_titles[movie_id] = title
    return catalog, movie_genres, movie_titles




    
if __name__ == "__main__":
    console_logging()
    prepare_data()