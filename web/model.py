import os
import ast
import shutil
import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
import implicit

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data_processing_training' / 'data'
MODELS_PATH = ROOT / 'data_processing_training' / 'models'
CURRENT_PATH = MODELS_PATH / 'current'
ARCHIVE_PATH = MODELS_PATH / 'archive'

_models_cache = {}
_knn_neighbors_cache = {}

ALS_FILES = ['als_model.pkl', 'user_to_idx.pkl', 'movie_to_idx.pkl',
             'movie_ids.pkl', 'user_ids.pkl', 'als_calibrator.pkl']


def _safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x if isinstance(x, list) else []


def _ensure_current_dir():
    CURRENT_PATH.mkdir(parents=True, exist_ok=True)
    for fname in ALS_FILES:
        cur = CURRENT_PATH / fname
        old = MODELS_PATH / fname
        if not cur.exists() and old.exists():
            shutil.copy(old, cur)


def list_als_versions():
    versions = ['current']
    if ARCHIVE_PATH.exists():
        archived = sorted(
            [d.name for d in ARCHIVE_PATH.iterdir() if d.is_dir()],
            reverse=True
        )
        versions.extend(archived)
    return versions


def _version_path(version):
    if version == 'current':
        return CURRENT_PATH
    return ARCHIVE_PATH / version


def load_models(version='current'):
    if version in _models_cache:
        return _models_cache[version]

    _ensure_current_dir()

    df_movies = pd.read_csv(DATA_PATH / "movies_train.csv")
    df_movies = df_movies.reset_index(drop=True)
    df_movies['genres_list'] = df_movies['genres_list'].apply(_safe_literal_eval)

    knn = joblib.load(MODELS_PATH / 'knn_model.pkl')
    feature_matrix = joblib.load(MODELS_PATH / 'feature_matrix.pkl')
    alpha = joblib.load(MODELS_PATH / 'best_alpha.pkl')

    assert len(df_movies) == feature_matrix.shape[0], (
        f"Рассинхрон: df_movies={len(df_movies)}, "
        f"feature_matrix={feature_matrix.shape[0]}. "
        f"Переобучите KNN (movies.ipynb) или проверьте movies_train.csv"
    )

    vpath = _version_path(version)
    als = joblib.load(vpath / 'als_model.pkl')
    user_to_idx = joblib.load(vpath / 'user_to_idx.pkl')
    movie_to_idx = joblib.load(vpath / 'movie_to_idx.pkl')
    movie_ids = joblib.load(vpath / 'movie_ids.pkl')
    user_ids = joblib.load(vpath / 'user_ids.pkl')

    calib_path = vpath / 'als_calibrator.pkl'
    calib = joblib.load(calib_path) if calib_path.exists() else None

    movie_id_to_feat_idx = {mid: i for i, mid in enumerate(df_movies['id'])}
    df_movies_by_id = df_movies.set_index('id')

    df_ratings = pd.read_csv(DATA_PATH / 'user_ratings_emulated.csv')
    mask = [
        (r in user_to_idx and m in movie_to_idx)
        for r, m in zip(df_ratings['user_id'], df_ratings['movie_id'])
    ]
    df_filt = df_ratings[mask]
    rows = [user_to_idx[r] for r in df_filt['user_id']]
    cols = [movie_to_idx[m] for m in df_filt['movie_id']]
    data = list(df_filt['rating'])
    interaction_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids))
    )

    models = {
        'version': version,
        'df_movies': df_movies,
        'df_movies_by_id': df_movies_by_id,
        'knn': knn,
        'feature_matrix': feature_matrix,
        'als': als,
        'calib': calib,
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx,
        'movie_ids': movie_ids,
        'user_ids': user_ids,
        'alpha': alpha,
        'interaction_matrix': interaction_matrix,
        'movie_id_to_feat_idx': movie_id_to_feat_idx,
    }
    _models_cache[version] = models
    return models


def users_not_in_model(ratings_df, version='current'):
    models = load_models(version)
    user_to_idx = models['user_to_idx']
    data_users = set(int(u) for u in ratings_df['user_id'].unique())
    model_users = set(int(u) for u in user_to_idx.keys())
    return sorted(data_users - model_users)


def reset_cache():
    global _models_cache, _knn_neighbors_cache
    _models_cache = {}
    _knn_neighbors_cache = {}


def _genres_of(df_movies_by_id, movie_id):
    try:
        g = df_movies_by_id.at[movie_id, 'genres_list']
    except KeyError:
        return []
    return g if isinstance(g, list) else _safe_literal_eval(g)


def _get_knn_neighbors(feat_idx, knn, feature_matrix):
    if feat_idx not in _knn_neighbors_cache:
        distances, indices = knn.kneighbors(
            feature_matrix[feat_idx:feat_idx + 1]
        )
        _knn_neighbors_cache[feat_idx] = (distances[0], indices[0])
    return _knn_neighbors_cache[feat_idx]


def predict_als(user_id, movie_id, als, user_to_idx, movie_to_idx, calib=None):
    user_idx = user_to_idx.get(user_id)
    movie_idx = movie_to_idx.get(movie_id)
    if user_idx is not None and movie_idx is not None:
        raw = np.dot(als.user_factors[user_idx], als.item_factors[movie_idx])
        if calib is not None:
            pred = calib.predict([[raw]])[0]
        else:
            pred = 5 + raw * 5
        return float(np.clip(pred, 1, 10))
    return 5.0


def predict_content(user_id, movie_id, ratings_df, knn, feature_matrix, movie_id_to_feat_idx):
    feat_idx = movie_id_to_feat_idx.get(movie_id)
    if feat_idx is None:
        return 5.0

    user_movies = ratings_df[ratings_df['user_id'] == user_id]
    if len(user_movies) == 0:
        return 5.0

    distances, indices = _get_knn_neighbors(feat_idx, knn, feature_matrix)
    sim_by_feat = {idx: 1 - dist for idx, dist in zip(indices, distances)}

    num, den = 0.0, 0.0
    for mid, rating in zip(user_movies['movie_id'], user_movies['rating']):
        rated_feat_idx = movie_id_to_feat_idx.get(mid)
        if rated_feat_idx is None:
            continue
        sim = sim_by_feat.get(rated_feat_idx)
        if sim is not None and sim > 0:
            num += sim * rating
            den += sim

    if den == 0:
        return float(user_movies['rating'].mean())

    return float(np.clip(num / den, 1, 10))


def hybrid_recommend(user_id, ratings_df, n=10, version='current'):
    models = load_models(version)
    df_movies = models['df_movies']
    df_movies_by_id = models['df_movies_by_id']
    knn = models['knn']
    feature_matrix = models['feature_matrix']
    als = models['als']
    calib = models['calib']
    user_to_idx = models['user_to_idx']
    movie_ids = models['movie_ids']
    alpha = models['alpha']
    interaction_matrix = models['interaction_matrix']
    movie_id_to_feat_idx = models['movie_id_to_feat_idx']

    user_idx = user_to_idx.get(user_id)
    already_rated = set(int(m) for m in ratings_df[ratings_df['user_id'] == user_id]['movie_id'])

    if user_idx is not None:
        movie_to_idx = models['movie_to_idx']
        user_items = interaction_matrix[user_idx]
        rec_indices, _ = als.recommend(
            user_idx, user_items, N=200, filter_already_liked_items=True
        )

        results = []
        for m_idx in rec_indices:
            movie_id = int(movie_ids[m_idx])
            if movie_id in already_rated:
                continue
            als_score = predict_als(user_id, movie_id, als, user_to_idx, movie_to_idx, calib)
            content_score = predict_content(
                user_id, movie_id, ratings_df, knn, feature_matrix, movie_id_to_feat_idx
            )
            hybrid_score = alpha * als_score + (1 - alpha) * content_score
            results.append({
                'movie_id': movie_id,
                'hybrid_score': hybrid_score,
                'als_score': als_score,
                'content_score': content_score,
            })

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:n]

    user_likes = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 8)
    ]
    if len(user_likes) == 0:
        user_likes = ratings_df[ratings_df['user_id'] == user_id]
    if len(user_likes) == 0:
        return []

    feat_idx_to_movie_id = df_movies['id'].to_numpy()
    n_movies = len(feat_idx_to_movie_id)

    candidates = {}
    for mid in user_likes['movie_id']:
        feat_idx = movie_id_to_feat_idx.get(mid)
        if feat_idx is None:
            continue
        distances, indices = _get_knn_neighbors(feat_idx, knn, feature_matrix)

        for dist, nb_idx in zip(distances[1:], indices[1:]):
            if not (0 <= nb_idx < n_movies):
                continue
            nb_movie_id = int(feat_idx_to_movie_id[nb_idx])
            if nb_movie_id in already_rated:
                continue
            sim = float(1 - dist)
            if nb_movie_id not in candidates or candidates[nb_movie_id]['content_score'] < sim:
                candidates[nb_movie_id] = {
                    'movie_id': nb_movie_id,
                    'hybrid_score': sim,
                    'als_score': 0.0,
                    'content_score': sim,
                }

    sorted_candidates = sorted(
        candidates.values(), key=lambda x: x['hybrid_score'], reverse=True
    )
    return sorted_candidates[:n]


def find_similar_users(user_id, ratings_df, n=5, version='current'):
    models = load_models(version)
    als = models['als']
    user_to_idx = models['user_to_idx']
    df_movies_by_id = models['df_movies_by_id']

    user_idx = user_to_idx.get(user_id)
    if user_idx is None:
        return []

    user_vec = als.user_factors[user_idx]
    all_vecs = als.user_factors

    norm_user = np.linalg.norm(user_vec)
    norms = np.linalg.norm(all_vecs, axis=1)
    norms[norms == 0] = 1e-10
    similarities = all_vecs @ user_vec / (norms * norm_user + 1e-10)
    similarities[user_idx] = -1

    top_indices = np.argsort(similarities)[::-1][:n]
    idx_to_user = {v: k for k, v in user_to_idx.items()}

    def genres_of_user(uid):
        liked = ratings_df[
            (ratings_df['user_id'] == uid) & (ratings_df['rating'] >= 7)
        ]['movie_id']
        gset = set()
        for mid in liked:
            gset.update(_genres_of(df_movies_by_id, mid))
        return gset

    user_genres = genres_of_user(user_id)

    results = []
    for idx in top_indices:
        other_id = idx_to_user.get(idx)
        if other_id is None:
            continue
        other_genres = genres_of_user(other_id)
        common = user_genres & other_genres
        results.append({
            'user_id': other_id,
            'similarity': round(float(similarities[idx]), 3),
            'common_genres': sorted(common),
            'total_ratings': int(len(ratings_df[ratings_df['user_id'] == other_id])),
        })

    return results


def retrain_als(ratings_df):
    _ensure_current_dir()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    archive_dir = ARCHIVE_PATH / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    for fname in ALS_FILES:
        src = CURRENT_PATH / fname
        if src.exists():
            shutil.copy(src, archive_dir / fname)

    user_ids = ratings_df['user_id'].unique()
    movie_ids = ratings_df['movie_id'].unique()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    rows = [user_to_idx[u] for u in ratings_df['user_id']]
    cols = [movie_to_idx[m] for m in ratings_df['movie_id']]
    data = list(ratings_df['rating'])
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=30, random_state=42
    )
    model.fit(matrix)

    u_idx = ratings_df['user_id'].map(user_to_idx).values
    m_idx = ratings_df['movie_id'].map(movie_to_idx).values
    raw = np.sum(model.user_factors[u_idx] * model.item_factors[m_idx], axis=1)
    calib = LinearRegression().fit(raw.reshape(-1, 1), ratings_df['rating'].values)

    joblib.dump(model, CURRENT_PATH / 'als_model.pkl')
    joblib.dump(user_to_idx, CURRENT_PATH / 'user_to_idx.pkl')
    joblib.dump(movie_to_idx, CURRENT_PATH / 'movie_to_idx.pkl')
    joblib.dump(user_ids, CURRENT_PATH / 'user_ids.pkl')
    joblib.dump(movie_ids, CURRENT_PATH / 'movie_ids.pkl')
    joblib.dump(calib, CURRENT_PATH / 'als_calibrator.pkl')

    reset_cache()

    return {
        'timestamp': timestamp,
        'n_users': len(user_ids),
        'n_movies': len(movie_ids),
        'n_ratings': len(ratings_df),
        'archived_to': str(archive_dir),
    }


def parse_kinopoisk_html(html_text):
    import re
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, 'lxml')
    items = soup.find_all('div', class_='item')

    parsed = []
    for item in items:
        name_rus_el = item.find('div', class_='nameRus')
        if not name_rus_el:
            continue
        link = name_rus_el.find('a')
        if not link:
            continue

        full_rus = link.get_text(strip=True)
        year_m = re.search(r'(\b19\d{2}|\b20\d{2})', full_rus)
        year = int(year_m.group(1)) if year_m else None
        rus_title = re.sub(r'\s*\(.*$', '', full_rus).strip()

        kp_id = None
        kp_m = re.search(r'/(?:film|series)/(\d+)', link.get('href', ''))
        if kp_m:
            kp_id = int(kp_m.group(1))

        name_eng_el = item.find('div', class_='nameEng')
        eng_title = name_eng_el.get_text(strip=True) if name_eng_el else None

        rate_el = item.find('div', class_=re.compile(r'rateNow\d+'))
        rating = None
        if rate_el and rate_el.get('vote'):
            try:
                rating = int(rate_el.get('vote'))
            except ValueError:
                rating = None

        if rating is not None and rus_title:
            parsed.append({
                'kp_id': kp_id,
                'rus_title': rus_title,
                'eng_title': eng_title,
                'year': year,
                'rating': rating,
            })

    return parsed


def map_ratings_to_catalog(parsed_ratings):
    models = load_models()
    df_movies = models['df_movies']

    title_col = 'name' if 'name' in df_movies.columns else 'title'
    df_by_id = df_movies.set_index('id')
    catalog_ids = set(int(i) for i in df_movies['id'])

    matched = []
    unmatched = []
    for r in parsed_ratings:
        kp_id = r.get('kp_id')
        if kp_id is not None and kp_id in catalog_ids:
            matched.append({
                'movie_id': int(kp_id),
                'title': df_by_id.loc[kp_id, title_col],
                'rating': r['rating'],
            })
        else:
            unmatched.append(f"{r['rus_title']} ({r['year']})")

    return matched, unmatched