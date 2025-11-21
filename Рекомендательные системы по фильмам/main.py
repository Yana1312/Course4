import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Система рекомендаций фильмов", layout="wide")
st.title("Система рекомендаций фильмов")


class MovieRecommender:
    def __init__(self):
        self.df = pd.read_csv('Films.csv')
        self.genre_columns = [col for col in self.df.columns if
                              col not in ['movieId', 'title', 'userId', 'rating', 'YearPublic']]
        self.tfidf_matrix = None
        self.movies_df = None
        self.tfidf_vectorizer = None
        self.popularite = None
        self.movies_features = None
        self.genre_stats_df = None
        self.prepare_popularity_data()
        self.prepare_title_similarity()
        self.prepare_genre_stats()

    def prepare_popularity_data(self):
        avg_ratings = self.df.groupby(['movieId', 'title'])['rating'].mean().reset_index().rename(
            columns={'rating': 'avg_rating'})
        avg = pd.DataFrame(avg_ratings).sort_values('avg_rating', ascending=False)

        cnt_ratings = self.df.groupby(['movieId', 'title'])['rating'].count().reset_index().rename(
            columns={'rating': 'count_rating'})
        cnt = pd.DataFrame(cnt_ratings).sort_values('count_rating', ascending=False)

        self.popularite = avg.merge(cnt, on=['movieId', 'title'])

        scaler = MinMaxScaler(feature_range=(0.3, 1))
        v_normalized = scaler.fit_transform(self.popularite[["count_rating"]]).flatten()
        self.popularite['count_rating_normalized'] = v_normalized

        v = self.popularite["count_rating"]
        v_norm = self.popularite["count_rating_normalized"]
        R = self.popularite["avg_rating"]
        m = v.quantile(0.90)
        c = R.mean()

        self.popularite['w_score_original'] = ((v * R) + (m * c)) / (v + m)
        self.popularite['w_score_normalized'] = ((v_norm * R) + (m * c)) / (v_norm + m)

    def prepare_title_similarity(self):
        self.movies_features = self.df.drop_duplicates('movieId')[['movieId', 'title'] + self.genre_columns]
        self.movies_df = self.df[['movieId', 'title']].drop_duplicates()
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['title'])

    def prepare_genre_stats(self):
        avg_ratings = self.df.groupby(['movieId', 'title'])['rating'].mean().reset_index().rename(
            columns={'rating': 'avg_rating'})
        cnt_ratings = self.df.groupby(['movieId', 'title'])['rating'].count().reset_index().rename(
            columns={'rating': 'count_rating'})
        popularite = avg_ratings.merge(cnt_ratings, on=['movieId', 'title'])

        movies_with_genres = self.df[['movieId'] + self.genre_columns].drop_duplicates()
        popularite_with_genres = popularite.merge(movies_with_genres, on='movieId')

        v = popularite_with_genres["count_rating"]
        R = popularite_with_genres["avg_rating"]
        m = v.quantile(0.90)
        c = R.mean()
        popularite_with_genres['w_score_original'] = ((v * R) + (m * c)) / (v + m)

        genre_stats = []
        for genre in self.genre_columns:
            genre_movies = popularite_with_genres[popularite_with_genres[genre] == 1]
            if len(genre_movies) > 0:
                avg_weighted = genre_movies['w_score_original'].mean()
                movie_count = len(genre_movies)
                total_ratings = genre_movies['count_rating'].sum()
                avg_rating = genre_movies['avg_rating'].mean()
                genre_stats.append({
                    'genre': genre,
                    'avg_weighted_score': avg_weighted,
                    'movie_count': movie_count,
                    'total_ratings': total_ratings,
                    'avg_rating': avg_rating
                })

        self.genre_stats_df = pd.DataFrame(genre_stats)
        self.top_10_genres = self.genre_stats_df.nlargest(10, 'avg_weighted_score')

    def get_top_movies_by_genre(self, genre):
        genre_movies = self.df[self.df[genre] == 1]

        movie_stats = genre_movies.groupby(['movieId', 'title']).agg({
            'rating': ['mean', 'count'],
            'YearPublic': 'first'
        }).round(2)

        movie_stats.columns = ['avg_rating', 'rating_count', 'year']
        movie_stats = movie_stats.reset_index()

        movie_stats = movie_stats[movie_stats['rating_count'] >= 10]

        if movie_stats.empty:
            return []

        top_movies = movie_stats.sort_values(['avg_rating', 'rating_count'], ascending=[False, False])

        top_10 = top_movies.head(10)

        result = []
        for idx, row in top_10.iterrows():
            result.append({
                'rank': len(result) + 1,
                'title': row['title'],
                'year': row['year'],
                'avg_rating': row['avg_rating'],
                'rating_count': row['rating_count'],
                'genre': genre
            })

        return result

    def get_top_popular_movies(self, method='original', n=10):
        if method == 'original':
            col = 'w_score_original'
            method_name = 'Оригинальный метод'
        else:
            col = 'w_score_normalized'
            method_name = 'MinMax нормализация'

        result = self.popularite.sort_values(col, ascending=False).head(n)
        return result, method_name

    def get_top_genres(self):
        return self.top_10_genres[['genre', 'avg_weighted_score', 'movie_count', 'total_ratings', 'avg_rating']]

    def get_user_high_rated_movies(self, user_id, min_rating=4.0, limit=10):
        try:
            user_movies = self.df[(self.df['userId'] == user_id) & (self.df['rating'] >= min_rating)]
            user_movies = user_movies.sort_values('rating', ascending=False).head(limit)

            high_rated = []
            for _, row in user_movies.iterrows():
                high_rated.append({
                    'movie_id': row['movieId'],
                    'title': row['title'],
                    'user_rating': row['rating'],
                    'genres': self.get_movie_genres(row['movieId'])
                })
            return high_rated
        except Exception as e:
            return []

    def get_movie_genres(self, movie_id):
        try:
            movie_data = self.df[self.df['movieId'] == movie_id].iloc[0]
            genres = []
            for genre in self.genre_columns:
                if movie_data[genre] == 1:
                    genres.append(genre)
            return ', '.join(genres) if genres else 'Не указаны'
        except:
            return 'Не указаны'

    def enhanced_collaborative_filtering(self, user_id, n_recommendations=10):
        try:
            user_item_matrix = self.df.pivot_table(index='userId', columns='movieId', values='rating')

            user_means = user_item_matrix.mean(axis=1)
            user_item_centered = user_item_matrix.sub(user_means, axis=0)
            user_item_filled = user_item_centered.fillna(0)

            user_similarity = cosine_similarity(user_item_filled)

            if user_id not in user_item_matrix.index:
                available_users = list(user_item_matrix.index)
                return [], [], f"Пользователь {user_id} не найден. Доступные пользователи: {available_users}"

            user_index = user_item_matrix.index.get_loc(user_id)

            similar_users_indices = []
            similarity_scores = []

            for i in range(len(user_similarity[user_index])):
                if i != user_index and user_similarity[user_index][i] > 0.1:
                    similar_users_indices.append(i)
                    similarity_scores.append(user_similarity[user_index][i])

            if not similar_users_indices:
                similar_users_indices = np.argsort(user_similarity[user_index])[::-1][1:11]
                similarity_scores = [user_similarity[user_index][i] for i in similar_users_indices]

            user_rated_movies = self.df[self.df['userId'] == user_id]['movieId'].values

            candidates = {}

            for similar_user_idx, similarity_score in zip(similar_users_indices, similarity_scores):
                similar_user_id = user_item_matrix.index[similar_user_idx]

                similar_user_ratings = self.df[(self.df['userId'] == similar_user_id) &
                                               (self.df['rating'] >= 3.5)]

                for _, movie_row in similar_user_ratings.iterrows():
                    movie_id = movie_row['movieId']

                    if movie_id in user_rated_movies:
                        continue

                    if movie_id not in candidates:
                        movie_title = movie_row['title']
                        candidates[movie_id] = {
                            'title': movie_title,
                            'total_weighted_rating': 0,
                            'total_similarity': 0,
                            'rating_count': 0,
                            'genres': self.get_movie_genres(movie_id)
                        }

                    candidates[movie_id]['total_weighted_rating'] += movie_row['rating'] * similarity_score
                    candidates[movie_id]['total_similarity'] += similarity_score
                    candidates[movie_id]['rating_count'] += 1

            recommendations = []
            for movie_id, data in candidates.items():
                if data['total_similarity'] > 0 and data['rating_count'] >= 2:
                    predicted_rating = data['total_weighted_rating'] / data['total_similarity']
                    predicted_rating = max(1.0, min(5.0, predicted_rating))

                    avg_similarity = data['total_similarity'] / data['rating_count']
                    confidence = avg_similarity * np.log1p(data['rating_count'])
                    confidence = min(1.0, confidence)

                    recommendations.append({
                        'movie_id': movie_id,
                        'title': data['title'],
                        'predicted_rating': predicted_rating,
                        'genres': data['genres'],
                        'confidence': confidence,
                        'recommendation_count': data['rating_count'],
                        'avg_similarity': avg_similarity
                    })

            recommendations.sort(key=lambda x: (x['predicted_rating'], x['confidence']), reverse=True)

            high_rated_movies = self.get_user_high_rated_movies(user_id)

            return high_rated_movies, recommendations[:n_recommendations], "Успешно"

        except Exception as e:
            return [], [], f"Ошибка: {str(e)}"

    def content_based_recommendations(self, movie_title):
        try:
            self.movies_features = self.df.drop_duplicates('movieId')[['movieId', 'title'] + self.genre_columns]
            self.movies_features.set_index('movieId', inplace=True)
            matching_movies = self.movies_features[
                self.movies_features['title'].str.contains(movie_title, case=False, na=False)]
            if matching_movies.empty:
                return [], f"Фильм '{movie_title}' не найден в базе данных"

            if len(matching_movies) > 1:
                target_movie_id = matching_movies.index[0]
                movie_title = matching_movies.iloc[0]['title']
            else:
                target_movie_id = matching_movies.index[0]

            similarity_matrix = cosine_similarity(self.movies_features[self.genre_columns])
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=self.movies_features.index,
                columns=self.movies_features.index
            )

            similar_movies = similarity_df[target_movie_id].sort_values(ascending=False)
            recommendations = similar_movies[similar_movies.index != target_movie_id].head(10)

            recommended_titles = self.movies_features.loc[recommendations.index, 'title']
            similarity_scores = recommendations.values

            results = []
            for movie_id, score in zip(recommended_titles.index, similarity_scores):
                title = recommended_titles[movie_id]
                results.append((movie_id, title, score))

            return results, f"Рекомендации для фильма: {movie_title}"

        except Exception as e:
            return [], f"Ошибка при поиске рекомендаций: {str(e)}"

    def find_similar_movies(self, movie_title):
        try:
            matching_movies = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False, na=False)]
            if matching_movies.empty:
                return [], f"Фильм '{movie_title}' не найден в базе данных"

            if len(matching_movies) > 1:
                input_vector = self.tfidf_vectorizer.transform([movie_title])
                similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

                best_match_idx = np.argmax(similarities)
                actual_movie_title = self.movies_df.iloc[best_match_idx]['title']
            else:
                actual_movie_title = matching_movies.iloc[0]['title']

            input_vector = self.tfidf_vectorizer.transform([actual_movie_title])
            similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

            results_df = self.movies_df.copy()
            results_df['similarity_score'] = similarities
            results_df = results_df[results_df['title'] != actual_movie_title]
            results_df = results_df.sort_values('similarity_score', ascending=False)

            if hasattr(self, 'movies_features'):
                final_results = results_df.head(10).merge(
                    self.movies_features.reset_index()[['movieId', 'title'] + self.genre_columns],
                    on=['movieId', 'title'],
                    how='left'
                )
            else:
                final_results = results_df.head(10)

            recommendations = []
            for idx, row in final_results.iterrows():
                recommendations.append((row['movieId'], row['title'], row['similarity_score']))

            return recommendations, f"Фильмы похожие на: {actual_movie_title}"

        except Exception as e:
            return [], f"Ошибка при поиске похожих фильмов: {str(e)}"

    def get_all_movies(self):
        return self.df[['movieId', 'title']].drop_duplicates().sort_values('title')

    def add_user_rating(self, user_id, movie_id, rating):
        existing_rating = self.df[
            (self.df['userId'] == user_id) &
            (self.df['movieId'] == movie_id)
            ]

        if not existing_rating.empty:
            return False, "Вы уже оценили этот фильм"

        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating],
            'title': [self.df[self.df['movieId'] == movie_id].iloc[0]['title']],
            'YearPublic': [self.df[self.df['movieId'] == movie_id].iloc[0]['YearPublic']]
        })

        for genre in self.genre_columns:
            new_rating[genre] = self.df[self.df['movieId'] == movie_id].iloc[0][genre]

        self.df = pd.concat([self.df, new_rating], ignore_index=True)
        return True

    def get_personal_recommendations(self, user_id, n_recommendations=10):
        return self.enhanced_collaborative_filtering(user_id, n_recommendations)


def main():
    st.sidebar.title("Навигация")
    section = st.sidebar.radio("Выберите раздел:",
                               ["Популярные фильмы",
                                "Коллаборативная фильтрация",
                                "Рекомендации по жанру",
                                "Рекомендации по названию",
                                "Мои оценки и рекомендации",
                                "Справка по командам"])

    if 'recommender' not in st.session_state:
        st.session_state.recommender = MovieRecommender()
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = 999

    recommender = st.session_state.recommender

    if section == "Популярные фильмы":
        show_popular_movies(recommender)
    elif section == "Коллаборативная фильтрация":
        show_collaborative_filtering(recommender)
    elif section == "Рекомендации по жанру":
        show_genre_recommendations(recommender)
    elif section == "Рекомендации по названию":
        show_title_recommendations(recommender)
    elif section == "Мои оценки и рекомендации":
        show_personal_recommendations(recommender)
    elif section == "Справка по командам":
        show_help()


def show_popular_movies(recommender):
    st.header("Топ популярных фильмов")
    method = st.radio("Метод расчета:",
                      ["Оригинальный метод", "MinMax нормализация"])

    method_key = 'original' if method == "Оригинальный метод" else 'normalized'
    result_df, method_name = recommender.get_top_popular_movies(method_key, 10)

    st.subheader(f"Топ 10 фильмов ({method_name})")

    result_display = result_df[['title', 'avg_rating', 'count_rating',
                                'w_score_original' if method_key == 'original' else 'w_score_normalized']].copy()
    result_display.columns = ['Название фильма', 'Средний рейтинг', 'Количество оценок', 'Взвешенная оценка']
    result_display['Средний рейтинг'] = result_display['Средний рейтинг'].round(3)
    result_display['Взвешенная оценка'] = result_display['Взвешенная оценка'].round(3)

    st.dataframe(result_display, use_container_width=True)


def show_collaborative_filtering(recommender):
    st.header("Рекомендации на основе коллаборативной фильтрации")

    user_id = st.number_input("ID пользователя:",
                              min_value=1,
                              max_value=610,
                              value=10,
                              help="Введите ID пользователя для которого нужно получить рекомендации")

    if st.button("Получить рекомендации", type="primary"):
        with st.spinner("Анализируем ваши предпочтения и ищем рекомендации..."):
            high_rated_movies, recommendations, message = recommender.enhanced_collaborative_filtering(user_id, 10)

        if recommendations or high_rated_movies:
            st.success(f"Рекомендации для пользователя {user_id}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ваши высокооцененные фильмы")
                if high_rated_movies:
                    for i, movie in enumerate(high_rated_movies, 1):
                        with st.container():
                            st.write(f"**{i}. {movie['title']}**")
                            st.write(f"   Ваша оценка: **{movie['user_rating']}**")
                            st.write(f"   Жанры: {movie['genres']}")
                            st.divider()
                else:
                    st.info("Не найдено высокооцененных фильмов (оценка ≥ 4.0)")

            with col2:
                st.subheader("Рекомендуемые фильмы")
                if recommendations:
                    for i, movie in enumerate(recommendations, 1):
                        with st.container():
                            st.write(f"**{i}. {movie['title']}**")
                            st.write(f"   Предсказанная оценка: **{movie['predicted_rating']:.2f}**")
                            st.write(f"   Жанры: {movie['genres']}")
                            st.write(f"   Уверенность: {movie['confidence']:.2f}")
                            st.divider()
                else:
                    st.info("Не удалось найти рекомендации")

        else:
            st.error(message)


def show_genre_recommendations(recommender):
    st.header("Рекомендации по жанру")

    available_genres = [col for col in recommender.df.columns if
                        col not in ['movieId', 'title', 'userId', 'rating', 'YearPublic']]

    selected_genre = st.selectbox("Выберите жанр:", available_genres)

    if st.button("Показать топ фильмов по жанру", type="primary"):
        with st.spinner("Ищем лучшие фильмы в выбранном жанре..."):
            top_movies = recommender.get_top_movies_by_genre(selected_genre)

        if top_movies:
            st.success(f"Топ-10 фильмов в жанре '{selected_genre}'")

            for movie in top_movies:
                with st.container():
                    st.write(f"**{movie['rank']}. {movie['title']} ({movie['year']})**")
                    st.write(f"   Средний рейтинг: **{movie['avg_rating']:.2f}**")
                    st.write(f"   Количество оценок: {movie['rating_count']}")
                    st.divider()
        else:
            st.error(f"Не найдено фильмов в жанре '{selected_genre}' с достаточным количеством оценок")

    top_genres_df = recommender.get_top_genres()
    display_df = top_genres_df.copy()
    display_df.columns = ['Жанр', 'Взвешенный рейтинг', 'Количество фильмов', 'Всего оценок', 'Средний рейтинг']
    display_df['Взвешенный рейтинг'] = display_df['Взвешенный рейтинг'].round(3)
    display_df['Средний рейтинг'] = display_df['Средний рейтинг'].round(3)

    st.subheader("Топ-10 жанров по взвешенному рейтингу")
    st.dataframe(display_df, use_container_width=True)


def show_title_recommendations(recommender):
    st.header("Рекомендации по названию фильма")

    movie_title = st.text_input("Введите название фильма:",
                                value="Toy Story",
                                help="Введите название фильма для поиска похожих по названию",
                                key="title_input")

    if st.button("Найти похожие фильмы по названию", type="primary"):
        with st.spinner("Анализируем названия и ищем рекомендации..."):
            recommendations, message = recommender.find_similar_movies(movie_title)

        if recommendations:
            st.success(message)

            for i, (movie_id, title, similarity) in enumerate(recommendations, 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.write(f"**{i}.**")
                    with col2:
                        st.write(f"**{title}**")
                    with col3:
                        st.write(f"{similarity:.3f}")
                st.divider()
        else:
            st.error(message)


def show_personal_recommendations(recommender):
    st.header("Мои оценки и рекомендации")

    st.subheader("Добавить оценку фильму")

    all_movies = recommender.get_all_movies()
    if st.session_state.current_user_id in st.session_state.user_ratings:
        rated_movie_ids = [rating['movie_id'] for rating in
                           st.session_state.user_ratings[st.session_state.current_user_id]]
        available_movies = all_movies[~all_movies['movieId'].isin(rated_movie_ids)]
    else:
        available_movies = all_movies

    if available_movies.empty:
        st.warning("Вы оценили все фильмы в базе данных!")
        selected_movie = None
    else:
        selected_movie = st.selectbox("Выберите фильм:", available_movies['title'].values)

    # all_movies = recommender.get_all_movies()
    # selected_movie = st.selectbox("Выберите фильм:", all_movies['title'].values)

    rating = st.slider("Ваша оценка:", 1.0, 5.0, 3.0, 0.5)

    if st.button("Добавить оценку", type="primary"):
        movie_id = all_movies[all_movies['title'] == selected_movie].iloc[0]['movieId']
        success = recommender.add_user_rating(st.session_state.current_user_id, movie_id, rating)

        if success:
            if st.session_state.current_user_id not in st.session_state.user_ratings:
                st.session_state.user_ratings[st.session_state.current_user_id] = []

            st.session_state.user_ratings[st.session_state.current_user_id].append({
                'movie_id': movie_id,
                'title': selected_movie,
                'rating': rating
            })
            st.success(f"Оценка {rating} добавлена для фильма '{selected_movie}'")

    st.subheader("Мои оценки")
    if st.session_state.current_user_id in st.session_state.user_ratings and st.session_state.user_ratings[
        st.session_state.current_user_id]:
        user_ratings = st.session_state.user_ratings[st.session_state.current_user_id]

        ratings_df = pd.DataFrame(user_ratings)
        st.dataframe(ratings_df[['title', 'rating']], use_container_width=True)
    else:
        st.info("Вы еще не оценили ни одного фильма")

    st.subheader("Персональные рекомендации")
    if st.button("Получить рекомендации на основе моих оценок", type="primary"):
        if st.session_state.current_user_id in st.session_state.user_ratings and st.session_state.user_ratings[
            st.session_state.current_user_id]:
            with st.spinner("Анализируем ваши оценки и ищем рекомендации..."):
                high_rated_movies, recommendations, message = recommender.get_personal_recommendations(
                    st.session_state.current_user_id, 10)

            if recommendations:
                st.success("Рекомендации на основе ваших оценок")

                for i, movie in enumerate(recommendations, 1):
                    with st.container():
                        st.write(f"**{i}. {movie['title']}**")
                        st.write(f"   Предсказанная оценка: **{movie['predicted_rating']:.2f}**")
                        st.write(f"   Жанры: {movie['genres']}")
                        st.write(f"   Уверенность: {movie['confidence']:.2f}")
                        st.divider()
            else:
                st.error("Не удалось найти рекомендации на основе ваших оценок")
        else:
            st.error("Сначала добавьте несколько оценок фильмам")


def show_help():
    st.header("Справка по командам и параметрам")

    st.markdown("""
    ### Популярные фильмы

    **Параметры:**
    - **Метод расчета**: Выбор между оригинальным методом и нормализацией MinMax

    **Алгоритм:**
    ```python
    WR = (v × R + m × C) / (v + m)
    ```
    - `v` - количество оценок фильма
    - `R` - средний рейтинг фильма  
    - `m` - минимальный порог оценок (90-й перцентиль)
    - `C` - средний рейтинг по всем фильмам

    ### Коллаборативная фильтрация

    **Параметры:**
    - **ID пользователя**: Число от 1 до 610

    **Алгоритм:**
    1. Находятся пользователи с похожими вкусами (косинусная схожесть)
    2. Анализируются высокооцененные фильмы похожих пользователей
    3. Рассчитывается предсказанная оценка для каждого рекомендованного фильма
    4. Отображаются как высокооцененные пользователем фильмы, так и рекомендации

    ### Рекомендации по жанру

    **Алгоритм:**
    1. Для каждого фильма вычисляется взвешенный рейтинг
    2. Для каждого жанра рассчитывается средний взвешенный рейтинг всех фильмов этого жанра
    3. Жанры сортируются по убыванию среднего взвешенного рейтинга
    4. Отображаются топ-10 жанров

    ### Рекомендации по названию

    **Параметры:**
    - **Название фильма**: Фильм для которого ищем похожие названия

    **Алгоритм:**
    1. Названия фильмов векторизуются с помощью TF-IDF
    2. Вычисляется косинусное сходство между векторными представлениями
    3. Находятся фильмы с наиболее похожими названиями

    ### Мои оценки и рекомендации

    **Функциональность:**
    1. Выбор фильма из базы данных
    2. Добавление оценки выбранному фильму
    3. Просмотр истории ваших оценок
    4. Получение персональных рекомендаций на основе ваших оценок
    """)


if __name__ == "__main__":
    main()