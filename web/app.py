import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shutil
import sys
import ast

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data_processing_training' / 'data'
MODELS_PATH = ROOT / 'data_processing_training' / 'models'
WEB_DATA_DIR = Path(__file__).parent / 'data'
RATINGS_PATH = WEB_DATA_DIR / 'user_ratings_emulated.csv'

sys.path.insert(0, str(Path(__file__).parent))
from model import (load_models, hybrid_recommend, find_similar_users, retrain_als,
                   parse_kinopoisk_html, map_ratings_to_catalog,
                   list_als_versions, users_not_in_model)

WEB_DATA_DIR.mkdir(exist_ok=True)
if not RATINGS_PATH.exists():
    original = DATA_PATH / 'user_ratings_emulated.csv'
    if original.exists():
        shutil.copy(original, RATINGS_PATH)

st.set_page_config(page_title="Рекомендательная система", layout="wide")
st.title("Гибридная рекомендательная система")

st.sidebar.header("Версия модели ALS")
available_versions = list_als_versions()
version_labels = {v: ("Текущая (current)" if v == 'current' else f"Архив: {v}")
                  for v in available_versions}
als_version = st.sidebar.selectbox(
    "Модель ALS",
    available_versions,
    format_func=lambda v: version_labels.get(v, v),
)

@st.cache_resource
def get_models(version):
    return load_models(version)

models = get_models(als_version)
df_movies = models['df_movies']
if 'name' in df_movies.columns and 'title' not in df_movies.columns:
    df_movies = df_movies.rename(columns={'name': 'title'})

@st.cache_data
def load_ratings():
    if RATINGS_PATH.exists():
        return pd.read_csv(RATINGS_PATH)
    return pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'title'])

def save_ratings(df):
    df.to_csv(RATINGS_PATH, index=False)
    st.cache_data.clear()

ratings_df = load_ratings()

def show_poster(poster_url, width=120):
    if pd.notna(poster_url) and str(poster_url).startswith('http'):
        st.image(poster_url, width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px;height:{int(width*1.4)}px;background:#2a2a2a;"
            f"border-radius:6px;display:flex;align-items:center;justify-content:center;"
            f"color:#666;font-size:11px;text-align:center'>Нет постера</div>",
            unsafe_allow_html=True
        )

def show_description(desc, limit=300):
    if not isinstance(desc, str) or not desc.strip():
        return
    desc = desc.strip()
    if len(desc) <= limit:
        st.caption(desc)
    else:
        st.caption(desc[:limit].rstrip() + "…")
        with st.expander("Читать полностью"):
            st.write(desc)

st.sidebar.header("Параметры")
user_id = st.sidebar.number_input(
    "ID пользователя", min_value=1, max_value=100000, value=1, step=1
)

user_in_als = user_id in models['user_to_idx']
if user_in_als:
    st.sidebar.success(f"Пользователь есть в модели «{version_labels.get(als_version, als_version)}»")
else:
    st.sidebar.warning(f"Пользователя нет в выбранной модели — cold start (KNN)")

missing = users_not_in_model(ratings_df, version=als_version)
with st.sidebar.expander(f"Не в модели ALS: {len(missing)} польз."):
    st.caption(
        f"Эти пользователи есть в данных, но отсутствуют в модели "
        f"«{version_labels.get(als_version, als_version)}» — для них работает cold start. "
        f"Чтобы добавить их в модель, переобучите ALS."
    )
    if missing:
        preview = ", ".join(str(u) for u in missing[:50])
        if len(missing) > 50:
            preview += f" … (+{len(missing) - 50})"
        st.text(preview)
    else:
        st.caption("Все пользователи из данных присутствуют в модели.")

if st.sidebar.button("Новый пользователь"):
    existing_ids = set(ratings_df['user_id'].unique()) | set(models['user_to_idx'].keys())
    new_id = max(existing_ids) + 1 if existing_ids else 1
    st.session_state['new_user_id'] = int(new_id)
    st.sidebar.info(f"Создан пользователь #{new_id}. Укажите его ID в поле выше и оцените фильмы.")

if 'new_user_id' in st.session_state:
    st.sidebar.caption(f"Последний созданный ID: {st.session_state['new_user_id']}")

n_user_ratings = len(ratings_df[ratings_df['user_id'] == user_id])
if n_user_ratings > 0:
    with st.sidebar.expander(f"Удалить пользователя #{user_id}"):
        st.caption(f"Будут удалены все {n_user_ratings} оценок. Действие необратимо.")
        confirm_delete = st.checkbox("Подтверждаю удаление", key="confirm_del")
        if st.button("Удалить пользователя", disabled=not confirm_delete):
            ratings_df = ratings_df[ratings_df['user_id'] != user_id]
            save_ratings(ratings_df)
            st.success(f"Пользователь #{user_id} удалён.")
            st.rerun()

st.sidebar.divider()
st.sidebar.header("Оценить фильм")
movie_title_input = st.sidebar.text_input("Название фильма (или часть)")

if movie_title_input:
    matches = df_movies[
        df_movies['title'].str.contains(movie_title_input, case=False, na=False)
    ]['title'].head(5).tolist()
    if matches:
        selected_title = st.sidebar.selectbox("Выберите фильм из списка", matches)
    else:
        selected_title = None
        st.sidebar.caption("Фильм не найден")
else:
    selected_title = None
    matches = []

rating_value = st.sidebar.slider("Оценка", min_value=1, max_value=10, value=7)
st.sidebar.caption(
    "1–3 — не понравился · 4–6 — нейтрально · 7–8 — хорошо · 9–10 — отлично"
)

def add_rating(title, rating):
    global ratings_df
    movie = df_movies[df_movies['title'] == title]
    if len(movie) == 0:
        st.sidebar.error("Фильм не найден")
        return
    movie_id = int(movie.iloc[0]['id'])
    existing = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id)
    ]
    if len(existing) > 0:
        ratings_df.loc[existing.index, 'rating'] = rating
    else:
        new_row = pd.DataFrame(
            [[user_id, movie_id, rating, title]],
            columns=['user_id', 'movie_id', 'rating', 'title']
        )
        ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    save_ratings(ratings_df)
    st.sidebar.success(f"Сохранено: «{title}» — {rating}/10")
    st.rerun()

if st.sidebar.button("Сохранить оценку", disabled=(selected_title is None)):
    add_rating(selected_title, rating_value)

st.sidebar.divider()
st.sidebar.header("Управление моделью")
st.sidebar.caption(
    "Переобучает ALS на всех текущих оценках, включая новых пользователей. "
    "Старая модель сохраняется в архив."
)
if st.sidebar.button("Переобучить ALS"):
    with st.spinner("Переобучаем модель..."):
        info = retrain_als(ratings_df)
        get_models.clear()
    st.sidebar.success(
        f"Готово! Пользователей: {info['n_users']}, "
        f"оценок: {info['n_ratings']}. "
        f"Старая версия в архиве ({info['timestamp']})."
    )
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Импорт оценок")
st.sidebar.caption(
    "Загрузите HTML-страницу «Мои оценки» с Кинопоиска — "
    "система создаст нового пользователя с этими оценками."
)

@st.dialog("Импорт оценок с Кинопоиска")
def import_dialog():
    st.write(
        "Сохраните страницу своих оценок на Кинопоиске "
        "(«Мои оценки» → Ctrl+S → веб-страница) и загрузите HTML-файл."
    )
    uploaded = st.file_uploader("HTML-файл с оценками", type=['html', 'htm'])

    if uploaded is not None:
        html_text = uploaded.read().decode('utf-8', errors='ignore')
        parsed = parse_kinopoisk_html(html_text)

        if not parsed:
            st.error("Не удалось распознать оценки в файле. "
                     "Проверьте, что это страница «Мои оценки» с Кинопоиска.")
            return

        matched, unmatched = map_ratings_to_catalog(parsed)
        st.info(
            f"Найдено оценок в файле: **{len(parsed)}**  \n"
            f"Сопоставлено с каталогом: **{len(matched)}**  \n"
            f"Не найдено в каталоге: **{len(unmatched)}**"
        )

        if len(matched) == 0:
            st.warning("Ни один фильм не найден в каталоге. Импорт невозможен.")
            return

        if st.button("Импортировать и создать пользователя"):
            global ratings_df
            existing_ids = set(ratings_df['user_id'].unique()) | set(models['user_to_idx'].keys())
            new_id = int(max(existing_ids) + 1) if existing_ids else 1

            new_rows = pd.DataFrame([
                {'user_id': new_id, 'movie_id': m['movie_id'],
                 'rating': m['rating'], 'title': m['title']}
                for m in matched
            ])
            ratings_df = pd.concat([ratings_df, new_rows], ignore_index=True)
            save_ratings(ratings_df)
            st.session_state['imported_user_id'] = new_id
            st.success(
                f"Импортировано {len(matched)} оценок. "
                f"Создан пользователь #{new_id}. "
                f"Укажите этот ID в поле слева, чтобы получить рекомендации."
            )

if st.sidebar.button("Загрузить оценки с Кинопоиска"):
    import_dialog()

if 'imported_user_id' in st.session_state:
    st.sidebar.success(f"Импортированный пользователь: #{st.session_state['imported_user_id']}")

tab_recs, tab_history, tab_catalog = st.tabs([
    "Рекомендации", "История оценок", "Каталог фильмов"
])

with tab_recs:
    n_recs = st.slider("Количество рекомендаций", min_value=5, max_value=20, value=10)

    if st.button("Получить рекомендации"):
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        user_likes = user_ratings[user_ratings['rating'] >= 8]

        if len(user_ratings) == 0:
            st.info("Оцените хотя бы несколько фильмов, чтобы получить рекомендации.")
        else:
            with st.spinner("Считаем рекомендации..."):
                results = hybrid_recommend(user_id, ratings_df, n=n_recs, version=als_version)

            if not results:
                st.warning("Не удалось сформировать рекомендации. Оцените больше фильмов.")
            else:
                mode = "гибридный (ALS + KNN)" if user_in_als else "cold start (только KNN)"
                st.subheader(f"Рекомендации для пользователя {user_id}")

                col_info1, col_info2, col_info3 = st.columns(3)
                col_info1.metric("Всего оценок", len(user_ratings))
                col_info2.metric("Высокие оценки (>=8)", len(user_likes))
                col_info3.metric("Режим", mode)

                st.divider()

                rows = []
                for i, rec in enumerate(results, 1):
                    movie_info = df_movies[df_movies['id'] == rec['movie_id']]
                    if len(movie_info) == 0:
                        continue
                    m = movie_info.iloc[0]
                    genres = m['genres_list']
                    if isinstance(genres, str):
                        try:
                            genres = ast.literal_eval(genres)
                        except Exception:
                            genres = []
                    rows.append({
                        "№": i,
                        "Название": m['title'],
                        "Год": int(m['year']) if pd.notna(m.get('year')) else "—",
                        "Рейтинг IMDb": round(float(m['rating_imdb']), 1) if pd.notna(m.get('rating_imdb')) else "—",
                        "Рейтинг КП": round(float(m['rating_kp']), 1) if pd.notna(m.get('rating_kp')) else "—",
                        "Жанры": ", ".join(genres),
                        "Описание": m['description'] if 'description' in m.index and pd.notna(m.get('description')) else "",
                        "Гибридный скор": round(rec['hybrid_score'], 4),
                        "ALS скор": round(rec['als_score'], 4),
                        "Контентный скор": round(rec['content_score'], 4),
                    })

                if rows:
                    for row_data in rows:
                        col_poster, col_info = st.columns([1, 5])
                        movie_info = df_movies[df_movies['title'] == row_data['Название']]
                        poster = None
                        if len(movie_info) > 0 and 'poster_url' in movie_info.columns:
                            poster = movie_info.iloc[0]['poster_url']
                        with col_poster:
                            show_poster(poster, width=100)
                        with col_info:
                            st.markdown(f"**{row_data['№']}. {row_data['Название']}** ({row_data['Год']})")
                            st.caption(f"IMDb: {row_data['Рейтинг IMDb']} · КП: {row_data['Рейтинг КП']} · {row_data['Жанры']}")
                            st.caption(f"Скор: {row_data['Гибридный скор']} (ALS: {row_data['ALS скор']}, KNN: {row_data['Контентный скор']})")
                            show_description(row_data['Описание'])
                        st.divider()

                if user_in_als:
                    st.divider()
                    st.subheader("Похожие пользователи")
                    similar = find_similar_users(user_id, ratings_df, n=5, version=als_version)
                    if similar:
                        for su in similar:
                            genres_str = ", ".join(su["common_genres"][:5]) if su["common_genres"] else "—"
                            uid = su["user_id"]
                            sim = su["similarity"]
                            cnt = su["total_ratings"]
                            st.markdown(f"**Пользователь #{uid}** — сходство: `{sim}`  \nОценок: {cnt} · Общие жанры: {genres_str}")

with tab_history:
    user_history = ratings_df[ratings_df['user_id'] == user_id].copy()
    if len(user_history) == 0:
        st.info("Вы ещё ничего не оценили.")
    else:
        def rating_label(r):
            if r >= 9:   return f"{r}/10 — Отлично"
            if r >= 7:   return f"{r}/10 — Хорошо"
            if r >= 4:   return f"{r}/10 — Нейтрально"
            return       f"{r}/10 — Не понравился"

        merge_cols = ['id', 'genres_list', 'rating_imdb', 'rating_kp']
        if 'description' in df_movies.columns:
            merge_cols.append('description')
        if 'poster_url' in df_movies.columns:
            merge_cols.append('poster_url')
        user_history = user_history.merge(
            df_movies[merge_cols],
            left_on='movie_id', right_on='id',
            how='left'
        )

        def genres_to_str(x):
            if isinstance(x, list):
                return ", ".join(x)
            if isinstance(x, str):
                try:
                    return ", ".join(ast.literal_eval(x))
                except (ValueError, SyntaxError):
                    return "—"
            return "—"

        user_history['Жанры'] = user_history['genres_list'].apply(genres_to_str)

        user_history = user_history.sort_values('rating', ascending=False)
        user_history['Оценка'] = user_history['rating'].apply(rating_label)

        for _, hr in user_history.iterrows():
            col_poster, col_info = st.columns([1, 6])
            poster = hr['poster_url'] if 'poster_url' in user_history.columns else None
            with col_poster:
                show_poster(poster, width=80)
            with col_info:
                imdb = round(float(hr['rating_imdb']), 1) if pd.notna(hr.get('rating_imdb')) else "—"
                kp = round(float(hr['rating_kp']), 1) if pd.notna(hr.get('rating_kp')) else "—"
                st.markdown(f"**{hr['title']}**")
                st.caption(f"IMDb: {imdb} · КП: {kp} · {hr['Жанры']}")
                st.markdown(hr['Оценка'])
                show_description(hr['description'] if 'description' in user_history.columns else None)
            st.divider()

        st.divider()
        st.subheader("Изменить оценку")
        edit_titles = user_history['title'].tolist()
        edit_title = st.selectbox("Фильм", edit_titles, key="edit_select")
        edit_rating = st.slider("Новая оценка", 1, 10, 7, key="edit_slider")
        col_e1, col_e2 = st.columns(2)
        if col_e1.button("Обновить"):
            add_rating(edit_title, edit_rating)
        if col_e2.button("Удалить оценку"):
            movie_row = df_movies[df_movies['title'] == edit_title]
            if len(movie_row) > 0:
                mid = int(movie_row.iloc[0]['id'])
                ratings_df = ratings_df[
                    ~((ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == mid))
                ]
                save_ratings(ratings_df)
                st.success(f"Оценка «{edit_title}» удалена.")
                st.rerun()

        if user_in_als:
            st.divider()
            st.subheader("Пользователи со схожими интересами")
            similar_hist = find_similar_users(user_id, ratings_df, n=5, version=als_version)
            if similar_hist:
                cols = st.columns(len(similar_hist))
                for col, su in zip(cols, similar_hist):
                    genres_str = ", ".join(su['common_genres'][:3]) if su['common_genres'] else "—"
                    col.metric(f"#{su['user_id']}", f"{su['similarity']}")
                    col.caption(f"{su['total_ratings']} оценок · {genres_str}")
            else:
                st.info("Похожие пользователи не найдены.")

with tab_catalog:
    st.subheader("Каталог фильмов")
    st.caption("Найдите фильм и оцените его прямо из каталога.")

    col_f1, col_f2, col_f3 = st.columns(3)
    search_query = col_f1.text_input("Поиск по названию", key="catalog_search")
    
    all_genres_flat = set()
    for g in df_movies['genres_list']:
        try:
            lst = ast.literal_eval(g) if isinstance(g, str) else g
            all_genres_flat.update(lst)
        except Exception:
            pass
    genre_filter = col_f2.selectbox(
        "Жанр", ["Все"] + sorted(all_genres_flat), key="genre_filter"
    )
    year_min, year_max = int(df_movies['year'].min()), int(df_movies['year'].max())
    year_range = col_f3.slider(
        "Год выпуска", year_min, year_max, (2000, year_max), key="year_range"
    )

    filtered = df_movies.copy()
    if search_query:
        filtered = filtered[filtered['title'].str.contains(search_query, case=False, na=False)]
    if genre_filter != "Все":
        filtered = filtered[
            filtered['genres_list'].apply(
                lambda x: genre_filter in (ast.literal_eval(x) if isinstance(x, str) else x)
            )
        ]
    filtered = filtered[
        (filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])
    ]
    filtered = filtered.sort_values('votes_kp', ascending=False)

    PAGE_SIZE = 50
    total = len(filtered)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    st.caption(f"Найдено фильмов: {total}")

    if 'catalog_page' not in st.session_state:
        st.session_state.catalog_page = 0
    filter_key = f"{search_query}|{genre_filter}|{year_range}"
    if st.session_state.get('catalog_filter_key') != filter_key:
        st.session_state.catalog_page = 0
        st.session_state.catalog_filter_key = filter_key

    page = st.session_state.catalog_page
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)

    display_df = filtered.iloc[start:end].copy()
    display_df['genres_list'] = display_df['genres_list'].apply(
        lambda x: ", ".join(ast.literal_eval(x) if isinstance(x, str) else x)
    )
    display_df['year'] = display_df['year'].astype(int)
    display_df['rating_imdb'] = display_df['rating_imdb'].round(1)
    display_df['rating_kp'] = display_df['rating_kp'].round(1)

    has_poster = 'poster_url' in display_df.columns
    cols = (['poster_url'] if has_poster else []) + ['title', 'year', 'rating_kp', 'rating_imdb', 'genres_list']
    rename_map = {
        'poster_url': 'Постер', 'title': 'Название', 'year': 'Год',
        'rating_kp': 'КП', 'rating_imdb': 'IMDb', 'genres_list': 'Жанры'
    }
    catalog_display = display_df[cols].rename(columns=rename_map).reset_index(drop=True)

    col_config = {}
    if has_poster:
        col_config["Постер"] = st.column_config.ImageColumn("Постер", width="small")

    st.dataframe(
        catalog_display,
        column_config=col_config,
        width='stretch',
        height=400,
    )

    col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
    if col_p1.button("Назад", disabled=(page == 0)):
        st.session_state.catalog_page -= 1
        st.rerun()
    col_p2.markdown(
        f"<div style='text-align:center; padding-top:8px'>Страница {page + 1} из {total_pages} "
        f"&nbsp;·&nbsp; фильмы {start + 1}–{end}</div>",
        unsafe_allow_html=True
    )
    if col_p3.button("Вперёд", disabled=(page >= total_pages - 1)):
        st.session_state.catalog_page += 1
        st.rerun()

    st.divider()
    st.subheader("Быстрая оценка")
    quick_title = st.selectbox(
        "Выберите фильм", display_df['title'].tolist(), key="quick_select"
    )
    quick_rating = st.slider("Оценка", 1, 10, 7, key="quick_rating")
    if st.button("Оценить", key="quick_save"):
        add_rating(quick_title, quick_rating)