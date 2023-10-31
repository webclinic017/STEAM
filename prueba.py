#Segundo
import pandas as pd

def UserForGenre(genero:str):
    try:
        # Cargar los DataFrames
        df_games = pd.read_csv('Data/steamGames_df.csv')
        df_item = pd.read_parquet('Data/df_desanidadaItem.parquet')
        df_reviews = pd.read_csv('Data/df_desanidadaReviews.csv')

        # Filtrar juegos por género
        juegos_genero = df_games[df_games['genres'] == genero]

        # Convertir la columna 'item_id' en df_item a int32 para que coincida con df_reviews
        df_item['item_id'] = df_item['item_id'].astype('int32')

        # Unir DataFrames para obtener horas jugadas por usuario y año para ese género
        df_merged = df_reviews.merge(df_item, on='item_id')
        df_merged = df_merged.merge(juegos_genero, left_on='item_id', right_on='appid')

        # Calcular horas jugadas por año para cada usuario
        df_merged['posted'] = pd.to_datetime(df_merged['posted'], unit='s')  # Convertir a fecha
        df_merged['Año'] = df_merged['posted'].dt.year
        horas_por_año = df_merged.groupby(['user_id', 'Año'])['playtime_forever'].sum().reset_index()

        # Encontrar el usuario con más horas jugadas para ese género
        usuario_max_horas = horas_por_año.groupby('user_id')['playtime_forever'].sum().idxmax()
        max_horas = horas_por_año[horas_por_año['user_id'] == usuario_max_horas]

        # Crear el resultado
        resultado = {
            "Usuario con más horas jugadas para " + genero: usuario_max_horas,
            "Horas jugadas": max_horas.to_dict(orient='records')
        }

        return resultado

    except FileNotFoundError:
        raise Exception("No se pudo cargar uno o más archivos de datos. Verifica la existencia de los archivos en las rutas especificadas.")

    except Exception as e:
        raise Exception(f"Se produjo un error inesperado: {str(e)}")

re = UserForGenre('Action')
print(re)
