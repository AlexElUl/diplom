import os
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Text, Float, ForeignKey, UniqueConstraint, text,
)

_WEB_DIR = Path(__file__).resolve().parent
_SRC_DATA_DIR = _WEB_DIR.parent / "data_processing_training" / "data"
_DB_DIR = _WEB_DIR / "data"
_DEFAULT_DB_PATH = _DB_DIR / "ratings.db"
MOVIES_CSV = _SRC_DATA_DIR / "movies_train.csv"
RATINGS_CSV = _SRC_DATA_DIR / "user_ratings_emulated.csv"

DB_URL = os.environ.get("RECSYS_DB_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

MOVIE_COLUMNS = [
    "id", "name", "year", "genres_list", "countries", "rating_kp",
    "rating_imdb", "votes_kp", "movie_length", "poster_url", "description",
]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "title"]

_engine = None
_ensured = False
metadata = MetaData()

movies = Table(
    "movies", metadata,
    Column("row_order", Integer, primary_key=True),
    Column("id", Integer, nullable=False, unique=True, index=True),
    Column("name", Text, nullable=False),
    Column("year", Integer, nullable=False),
    Column("countries", Text),
    Column("rating_kp", Float, nullable=False),
    Column("rating_imdb", Float),
    Column("votes_kp", Integer, nullable=False),
    Column("movie_length", Float),
    Column("poster_url", Text),
    Column("description", Text),
)

genres = Table(
    "genres", metadata,
    Column("movie_id", Integer, ForeignKey("movies.id"), nullable=False, index=True),
    Column("pos", Integer, nullable=False),
    Column("genre", Text, nullable=False),
    UniqueConstraint("movie_id", "pos", name="uq_genre_pos"),
)

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
)

ratings = Table(
    "ratings", metadata,
    Column("rid", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False, index=True),
    Column("movie_id", Integer, nullable=False, index=True),
    Column("rating", Integer, nullable=False),
    Column("title", Text),
)


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {}
        if DB_URL.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            if DB_URL == f"sqlite:///{_DEFAULT_DB_PATH}":
                _DB_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DB_URL, connect_args=connect_args, future=True)
    return _engine


def init_schema(drop=False):
    eng = get_engine()
    if drop:
        metadata.drop_all(eng)
    metadata.create_all(eng)


def _parse_genres(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _seed_movies(eng):
    df = pd.read_csv(MOVIES_CSV)
    assert list(df.columns) == MOVIE_COLUMNS, (
        f"Неожиданные колонки movies_train.csv: {list(df.columns)}"
    )
    df = df.reset_index(drop=True)
    parsed_genres = df["genres_list"].apply(_parse_genres)

    movies_rows = []
    genres_rows = []
    for row_order, (_, m) in enumerate(df.iterrows()):
        movies_rows.append({
            "row_order": row_order,
            "id": int(m["id"]),
            "name": m["name"],
            "year": int(m["year"]),
            "countries": None if pd.isna(m["countries"]) else m["countries"],
            "rating_kp": float(m["rating_kp"]),
            "rating_imdb": None if pd.isna(m["rating_imdb"]) else float(m["rating_imdb"]),
            "votes_kp": int(m["votes_kp"]),
            "movie_length": None if pd.isna(m["movie_length"]) else float(m["movie_length"]),
            "poster_url": None if pd.isna(m["poster_url"]) else m["poster_url"],
            "description": None if pd.isna(m["description"]) else m["description"],
        })
        for pos, genre in enumerate(parsed_genres.iloc[row_order]):
            genres_rows.append({"movie_id": int(m["id"]), "pos": pos, "genre": genre})

    pd.DataFrame(movies_rows).to_sql("movies", eng, if_exists="append", index=False)
    if genres_rows:
        pd.DataFrame(genres_rows).to_sql("genres", eng, if_exists="append", index=False)
    return df, parsed_genres


def _seed_ratings(eng):
    if not RATINGS_CSV.exists():
        return
    df = pd.read_csv(RATINGS_CSV)
    expected = {"user_id", "movie_id", "rating", "title"}
    assert expected.issubset(df.columns), (
        f"В {RATINGS_CSV.name} нет нужных колонок: {expected - set(df.columns)}"
    )
    out = df[["user_id", "movie_id", "rating", "title"]].copy()
    out.to_sql("ratings", eng, if_exists="append", index=False)

    uids = sorted(int(u) for u in out["user_id"].unique())
    pd.DataFrame({"id": uids}).to_sql("users", eng, if_exists="append", index=False)


def _verify_movies_order(orig_df, orig_genres):
    from_db = read_movies_df()
    assert len(from_db) == len(orig_df), (
        f"Кол-во строк не совпало: БД={len(from_db)}, CSV={len(orig_df)}"
    )
    assert from_db["id"].tolist() == orig_df["id"].astype(int).tolist(), (
        "ПОРЯДОК id из БД не совпадает с CSV — feature_matrix рассинхронизируется!"
    )
    mismatch = sum(
        1 for a, b in zip(from_db["genres_list"].tolist(), orig_genres.tolist()) if a != b
    )
    assert mismatch == 0, f"genres_list не совпал в {mismatch} строках"


def _seed_from_csv(eng):
    orig_df, orig_genres = _seed_movies(eng)
    _seed_ratings(eng)
    _verify_movies_order(orig_df, orig_genres)


def ensure_db():
    global _ensured
    if _ensured:
        return
    eng = get_engine()
    init_schema(drop=False)
    with eng.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM movies")).scalar()
    if not n:
        _seed_from_csv(eng)
    _ensured = True


def rebuild_from_csv():
    global _ensured
    init_schema(drop=True)
    _seed_from_csv(get_engine())
    _ensured = True


def read_movies_df():
    eng = get_engine()
    df = pd.read_sql(
        "SELECT id, name, year, countries, rating_kp, rating_imdb, "
        "votes_kp, movie_length, poster_url, description "
        "FROM movies ORDER BY row_order",
        eng,
    )
    g = pd.read_sql("SELECT movie_id, genre FROM genres ORDER BY movie_id, pos", eng)
    g_map = g.groupby("movie_id")["genre"].apply(list).to_dict()
    df["genres_list"] = df["id"].map(lambda mid: g_map.get(mid, []))

    for col in ("countries", "poster_url", "description"):
        df[col] = df[col].where(df[col].notna(), np.nan)

    return df[MOVIE_COLUMNS]


def read_ratings_df():
    eng = get_engine()
    df = pd.read_sql(
        "SELECT user_id, movie_id, rating, title FROM ratings ORDER BY rid", eng
    )
    if df.empty:
        return pd.DataFrame(columns=RATING_COLUMNS)
    return df[RATING_COLUMNS]


def write_ratings_df(df):
    eng = get_engine()
    out = df[RATING_COLUMNS].copy()
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM ratings"))
        if len(out):
            out.to_sql("ratings", conn, if_exists="append", index=False)
        conn.execute(text("DELETE FROM users"))
        uids = sorted(int(u) for u in out["user_id"].unique())
        if uids:
            conn.execute(
                text("INSERT INTO users (id) VALUES (:id)"),
                [{"id": u} for u in uids],
            )


if __name__ == "__main__":
    print(f"DB_URL: {DB_URL}")
    rebuild_from_csv()
    print("Пересев из CSV завершён.")