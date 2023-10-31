from fastapi import FastAPI, HTTPException
import pandas as pd
import fastparquet
import dask.dataframe as dd
import dask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title='Mi primer API/ PROYECTO INDIVIDUAL')

#http://127.0.0.1:8000  --> UBICACION

@app.get("/")
async def read_root():
    return {"Mi primera API"}

@app.get("/")
async def index():
    return {'Esta API fue realizado por Jeferson'}

@app.get('/about/')
async def about():
    return {'Proyecto individual Data Science'}

@app.get('/PlayTimeGenre/{genero}')
async def PlayTimeGenre(genero: str):
    try:
        # Cargar los DataFrames
        steamGames = pd.read_csv('Data/steamGames_df.csv')
        df_desanidadaItem = pd.read_parquet('Data/df_desanidadaItem.parquet')

        # Filtramos genres
        genero_filtrado = steamGames[steamGames['genres'].apply(lambda x: genero in x)]

        # Verificar si no hay datos para el género
        if genero_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos para el género {genero}")

        # Unir los data frames mediante 'item_id'
        merged_df = pd.merge(genero_filtrado, df_desanidadaItem, on='item_id')

        # Verificar si no hay datos de horas jugadas para el género
        if merged_df.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos de horas jugadas para el género {genero}")

        # Convertir la columna playtime_forever a horas
        merged_df['playtime_forever'] = merged_df['playtime_forever'] / 60

        # Encontrar el año con más horas jugadas
        max_hours_year = merged_df.groupby('release_date')['playtime_forever'].sum().idxmax()

        # Convertir max_hours_year a un tipo serializable (por ejemplo, str)
        max_hours_year = str(max_hours_year)

        return {"Año de lanzamiento con más horas jugadas para el Género " + genero: max_hours_year}
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Error al cargar los archivos de datos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/UserForGenre/{genero}')
async def UserForGenre(genero:str):
    try:
        # Cargar los DataFrames utilizando Dask
        df_games = dd.read_csv('Data/steamGames_df.csv')
        df_item = dd.read_parquet('Data/df_desanidadaItem.parquet')
        df_reviews = dd.read_csv('Data/df_desanidadaReviews.csv')

        # Filtrar juegos por género
        juegos_genero = df_games[df_games['genres'].str.contains(genero, case=False, na=False)]

        # Realizar la fusión de DataFrames
        df_merged = dd.merge(df_reviews, df_item, on='item_id')
        df_merged = dd.merge(df_merged, juegos_genero, on='item_id')

        # Dividir los datos en particiones más pequeñas
        df_merged = df_merged.repartition(npartitions=4)  # Ajusta el número de particiones según tus necesidades

        # Realizar cálculos con Dask
        df_merged['Año'] = df_merged['posted']
        df_merged['playtime_forever'] = df_merged['playtime_forever'] / 60  # Convertir minutos a horas

        horas_por_año = df_merged.groupby(['user_id', 'Año'])['playtime_forever'].sum().reset_index()

        usuario_max_horas = horas_por_año.groupby('user_id')['playtime_forever'].sum().idxmax()
        max_horas = horas_por_año[horas_por_año['user_id'] == usuario_max_horas]

        acumulacion_horas = horas_por_año.groupby('Año')['playtime_forever'].sum().reset_index()
        acumulacion_horas = acumulacion_horas.rename(columns={'playtime_forever': 'Horas'})

        # Calcular los resultados
        with dask.config.set(scheduler='threads'):  # Usar el scheduler de hilos para cálculos locales
            usuario_max_horas_result = usuario_max_horas.compute()
            usuario_max_horas_result = max_horas.compute()
            acumulacion_horas_result = acumulacion_horas.compute()

        resultado = {
            "Usuario con más horas jugadas para " + genero: usuario_max_horas_result,
            "Horas jugadas": acumulacion_horas_result.to_dict(orient='records')
        }

        return resultado
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Error al cargar los archivos de datos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.get('/UsersRecommend/{anio}')  
async def UsersRecommend(año:int):
    try:
        # Cargar los DataFrames
        df_reviews = pd.read_csv('Data/df_desanidadaReviews.csv')
        df_item = pd.read_parquet('Data/df_desanidadaItem.parquet')

        # Filtrar las revisiones por año, recomendaciones positivas y comentarios positivos/neutrales
        reviews_filtradas = df_reviews[(df_reviews['posted'] == año) & (df_reviews['recommend'] == True) & (df_reviews['sentiment_analysis'] >= 1)]

        # Verificar si no hay datos después de aplicar los filtros
        if reviews_filtradas.empty:
            raise Exception(f"No hay datos para el año {año} con los filtros especificados.")

        # Unir las revisiones con la información del juego
        df_merged = reviews_filtradas.merge(df_item, on='item_id')

        # Contar la cantidad de recomendaciones por juego
        recomendaciones_por_juego = df_merged.groupby('item_name')['recommend'].sum().reset_index()

        # Verificar si no hay datos después de la fusión
        if recomendaciones_por_juego.empty:
            raise Exception(f"No hay juegos recomendados para el año {año} con los filtros especificados.")

        # Obtener el top 3 de juegos más recomendados
        top_juegos_recomendados = recomendaciones_por_juego.nlargest(3, 'recommend')

        # Crear el resultado
        resultado = [{"Puesto 1": top_juegos_recomendados.iloc[0]['item_name']},
                    {"Puesto 2": top_juegos_recomendados.iloc[1]['item_name']},
                    {"Puesto 3": top_juegos_recomendados.iloc[2]['item_name']}]

        return resultado

    except FileNotFoundError:
        raise Exception("No se pudo cargar uno o más archivos de datos. Verifica la existencia de los archivos en las rutas especificadas.")

    except Exception as e:
        raise Exception(f"Se produjo un error inesperado: {str(e)}")
    
@app.get('/UsersNotRecommend/{anio}')
async def UsersNotRecommend(año:int):
    try:
        # Cargar los DataFrames
        df_reviews = pd.read_csv('Data/df_desanidadaReviews.csv')
        df_item = pd.read_parquet('Data/df_desanidadaItem.parquet')

        # Filtrar las revisiones por año, recomendaciones negativas y comentarios negativos
        reviews_filtradas = df_reviews[(df_reviews['posted'] == año) & (df_reviews['recommend'] == False) & (df_reviews['sentiment_analysis'] == 0)]

        # Verificar si no hay datos después de aplicar los filtros
        if reviews_filtradas.empty:
            raise Exception(f"No hay datos de juegos menos recomendados para el año {año} con los filtros especificados.")

        # Unir las revisiones con la información del juego
        df_merged = reviews_filtradas.merge(df_item, on='item_id')

        # Contar la cantidad de juegos menos recomendados
        juegos_menos_recomendados = df_merged['item_name'].value_counts().reset_index()
        juegos_menos_recomendados.columns = ['item_name', 'count']

        # Obtener el top 3 de juegos menos recomendados
        top_juegos_menos_recomendados = juegos_menos_recomendados.nlargest(3, 'count')

        # Crear el resultado
        resultado = [{"Puesto 1": top_juegos_menos_recomendados.iloc[0]['item_name']},
                    {"Puesto 2": top_juegos_menos_recomendados.iloc[1]['item_name']},
                    {"Puesto 3": top_juegos_menos_recomendados.iloc[2]['item_name']}]

        return resultado

    except FileNotFoundError:
        raise Exception("No se pudo cargar uno o más archivos de datos. Verifica la existencia de los archivos en las rutas especificadas.")

    except Exception as e:
        raise Exception(f"Se produjo un error inesperado: {str(e)}")
    
@app.get('/Sentiment_analysis/{anio}')
async def sentiment_analysis(año:int):
    try:
        # Cargar los DataFrames
        steamGames = pd.read_csv('Data/steamGames_df.csv')
        df_reviews = pd.read_csv('Data/df_desanidadaReviews.csv')

        # Filtrar los juegos por el año de lanzamiento
        juegos_año = steamGames[steamGames['release_date'] == año]
        # Verificar si no hay datos después de aplicar los filtros
        if juegos_año.empty:
            raise Exception(f"No hay datos para el año {año} con los filtros especificados.")

        # Unir las reseñas con los datos de los juegos para el año dado
        df_merged = df_reviews.merge(juegos_año, left_on='item_id', right_on='item_id')

        # Calcular la cantidad de registros para cada categoría de análisis de sentimiento
        resultados = {
            'Negative': len(df_merged[df_merged['sentiment_analysis'] == 0]),
            'Neutral': len(df_merged[df_merged['sentiment_analysis'] == 1]),
            'Positive': len(df_merged[df_merged['sentiment_analysis'] == 2])
        }

        return resultados
    except FileNotFoundError:
        raise Exception("No se pudo cargar uno o más archivos de datos. Verifica la existencia de los archivos en las rutas especificadas.")

    except Exception as e:
        raise Exception(f"Se produjo un error inesperado: {str(e)}")
    
@app.get("/recomendacion_juego/{product_id}")
async def recomendacion_juego(product_id: int):
    try:
        steamGames = pd.read_csv('Data/steamGames_df.csv')
        # Obtén el juego de referencia
        target_game = steamGames[steamGames['item_id'] == product_id]

        if target_game.empty:
            return {"message": "No se encontró el juego de referencia."}

        # Combina las etiquetas (tags) y géneros en una sola cadena de texto
        target_game_tags_and_genres = ' '.join(target_game['tags'].fillna('').astype(str) + ' ' + target_game['genres'].fillna('').astype(str))

        # Crea un vectorizador TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        # Configura el tamaño del lote para la lectura de juegos
        chunk_size = 100   
        similarity_scores = None

        # Procesa los juegos por lotes utilizando chunks
        for chunk in pd.read_csv('Data/steamGames_df.csv', chunksize=chunk_size):
            # Combina las etiquetas (tags) y géneros de los juegos en una sola cadena de texto
            chunk_tags_and_genres = ' '.join(chunk['tags'].fillna('').astype(str) + ' ' + chunk['genres'].fillna('').astype(str))

            # Aplica el vectorizador TF-IDF al lote actual de juegos y al juego de referencia
            tfidf_matrix = tfidf_vectorizer.fit_transform([target_game_tags_and_genres, chunk_tags_and_genres])

            # Calcula la similitud entre el juego de referencia y los juegos del lote actual
            if similarity_scores is None:
                similarity_scores = cosine_similarity(tfidf_matrix)
            else:
                similarity_scores = cosine_similarity(tfidf_matrix, X=similarity_scores)

        if similarity_scores is not None:
            # Obtiene los índices de los juegos más similares
            similar_games_indices = similarity_scores[0].argsort()[::-1]

            # Recomienda los juegos más similares (puedes ajustar el número de recomendaciones)
            num_recommendations = 5
            recommended_games = steamGames.loc[similar_games_indices[1:num_recommendations + 1]]

            # Devuelve la lista de juegos recomendados
            return recommended_games[['app_name', 'tags', 'genres']].to_dict(orient='records')

        return {"message": "No se encontraron juegos similares."}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}