# Librerias -------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
from scipy.stats import kurtosis, skew, norm, t

# a) --------------------------------------------------------------------------------

# Descargamos la información de los activos desde 2010
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start='2010-01-01',end='2025-04-03')['Close']
    return df

activos = ['GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA','SPY']
nombres_activos = {
    "GOOGL":"Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NFLX": "Netflix",
    "TSLA": "Tesla",
    "SPY": "SPDR S&P 500 ETF Trust"
}

df_precios=obtener_datos(activos)

st.title("Evaluación de métricas de riesgo y rendimiento para activos financieros")

activo_seleccionado = st.selectbox("Selecciona una activo", activos)

nombre_mostrado = nombres_activos.get(activo_seleccionado, activo_seleccionado)

# b) --------------------------------------------------------------------------------

#Calculamos los rendimientos diarios, la media, el sesgo y el exceso de curtosis

if activo_seleccionado:
    @st.cache_data
    def calcular_rendimientos(df):
        return df.pct_change().dropna()
    df_rendimientos = calcular_rendimientos(df_precios[activo_seleccionado])

    st.subheader(f"Métricas de Rendimiento - {nombre_mostrado}")

    st.write("""
    A través del cálculo de los rendimientos diarios de un activo, podemos obtener métricas clave que nos permiten evaluar su comportamiento y características estadísticas. Las métricas calculadas son:

    - **Rendimiento Medio Diario**: Representa el rendimiento promedio del activo por día desde 2010.
    - **Sesgo**: Mide la asimetría de la distribución de los rendimientos, indicándonos si los rendimientos tienden a ser más positivos o negativos.
    - **Curtosis**: Evalúa la "altitud" de las colas de la distribución de los rendimientos. Un valor elevado de curtosis sugiere una mayor probabilidad de observar movimientos extremos en los precios.
    """)
    
    media = df_rendimientos.mean()
    sesgo = skew(df_rendimientos)
    curtosis = kurtosis(df_rendimientos)
    
    col1, col2, col3= st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{media:.4%}")
    col2.metric("Sesgo", f"{sesgo:.4}")
    col3.metric("Curtosis", f"{curtosis:.4}")

# c) --------------------------------------------------------------------------------

# Calculo de VAR y ES con distintos métodos para múltiples valores de alpha
    @st.cache_data
    def calcular_var_es(df):
        NCS = [0.95, 0.975, 0.99]
        resultados =  pd.DataFrame(columns=['VaR (Normal)', 'ES (Normal)','VaR (t-Student)','ES (t-Student)','VaR (Histórico)', 'ES (Histórico)', 'VaR (Monte Carlo)', 'ES (Monte Carlo)'])
        df_size = len(df)
        for NC in NCS:
            mean = df.mean()
            stdev = df.std()

            #VaR Paramétrico(Normal)
            VaR_norm = norm.ppf(1-NC,loc=mean,scale=stdev)
            #ES Paramétrico (Normal)
            ES_norm = mean - (stdev * norm.pdf(norm.ppf(NC)) / (1 - NC))

            #VaR Paramétrico (t-student)
            nu = df_size - 1  # Grados de libertad
            VaR_t = t.ppf(1 - NC, nu, loc=mean, scale=stdev)
            #ES Paramétrico (t-student)
            ES_t = mean - (stdev * t.pdf(t.ppf(NC, nu), nu) / (1 - NC))

            # VaR Histórico
            VaR_hist = df.quantile(1 - NC)
            # ES Histórico
            ES_hist = df[df <= VaR_hist].mean()

            # VaR Monte Carlo
            sim_return = np.random.normal(mean, stdev, 100000)
            VaR_MC = np.percentile(sim_return, (1 - NC)*100)
            # ES Monte Carlo
            ES_MC = sim_return[sim_return <= VaR_MC].mean()
            
            resultados.loc[f"{int(NC * 100)}% de confianza"] = [VaR_norm, ES_norm, VaR_t, ES_t, VaR_hist, ES_hist, VaR_MC, ES_MC]
        return resultados

    var_es_results = calcular_var_es(df_rendimientos)
    st.subheader(f"Cálculo de Value at Risk y Expected Shortfall - {nombre_mostrado}")
    st.write("""
    En esta sección, calculamos el **Valor en Riesgo (VaR)** y el **Valor Esperado (ES)** del activo financiero bajo diferentes intervalos de confianza (0.95, 0.975, y 0.99) utilizando varios métodos de aproximación.

    - **Distribución Normal**: Asumimos que los rendimientos siguen una distribución normal para calcular el VaR y el ES.
    - **Distribución t-Student**: Se considera la distribución t-Student para tener en cuenta los posibles colas más gruesas de los rendimientos.
    - **Método Histórico**: Utilizando los datos históricos disponibles, se calcula el VaR y el ES directamente a partir de los percentiles de los rendimientos observados.
    - **Simulación de Monte Carlo**: Generando una gran cantidad de simulaciones de los rendimientos, se calcula el VaR y el ES a partir de las distribuciones obtenidas.
    """)
    st.dataframe(var_es_results)

# d) --------------------------------------------------------------------------------

# Cálculo de VaR y ES con ventanas móviles para alpha 0.05 y 0.01
    def rolling_var_es(df, window=252, CNS=[0.05, 0.01]):
        resultados = []

        for CN in CNS:
            VaR_histo = df.rolling(window).quantile(CN)
            ES_histo = df.rolling(window).apply(lambda x: x[x <= x.quantile(CN)].mean())
            VaR_parame = norm.ppf(CN) * df.rolling(window).std()
            ES_parame = df.rolling(window).mean() - (df.rolling(window).std() * norm.pdf(norm.ppf(CN)) / (1 - CN))

            resultados.append(pd.DataFrame({
                f'VaR (Histórico) {CN}': VaR_histo,
                f'ES (Histórico) {CN}': ES_histo,
                f'VaR (Parametrico) {CN}': VaR_parame,
                f'ES (Parametrico) {CN}': ES_parame
            }))

        # Concatenamos todos los DataFrames en uno solo
        return pd.concat(resultados, axis=1)

    #Aplicamos la función que creamos para los rendimientos del activo seleccionado
    df_var_es_rolling = rolling_var_es(df_rendimientos)

    st.subheader(f"Rolling Windows para VaR y ES (252 días) - {nombre_mostrado}")
    st.write("""
    El VaR y el ES son medidas clave para evaluar el riesgo de un activo o cartera. Para adaptar estas métricas a condiciones cambiantes, utilizamos el enfoque de **Rolling Windows**, en el cual se calcula el VaR y el ES con una ventana de 252 días, moviendo la ventana día a día para recalcular el riesgo en cada período.
    """)
      
    opciones = [
        "Rendimientos",
        "VaR (Histórico) 0.05", "VaR (Histórico) 0.01",
        "VaR (Parametrico) 0.05", "VaR (Parametrico) 0.01",
        "ES (Histórico) 0.05", "ES (Histórico) 0.01",
        "ES (Parametrico) 0.05", "ES (Parametrico) 0.01"
    ]

    series_seleccionadas = st.multiselect("Selecciona las medidas a visualizar", opciones, default=opciones)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Graficar Rendimientos
    if "Rendimientos" in series_seleccionadas:
        ax.plot(df_rendimientos.index, df_rendimientos, label=f"Rendimientos {nombre_mostrado}")

    # Graficar cada serie según la selección del usuario
    if "VaR (Histórico) 0.05" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Histórico) 0.05"], label="VaR Histórico 5%", linestyle="solid", color="red")

    if "VaR (Histórico) 0.01" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Histórico) 0.01"], label="VaR Histórico 1%", linestyle="solid", color="blue")

    if "VaR (Parametrico) 0.05" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Parametrico) 0.05"], label="VaR Paramétrico 5%", linestyle="solid", color="orange")

    if "VaR (Parametrico) 0.01" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Parametrico) 0.01"], label="VaR Paramétrico 1%", linestyle="solid", color="green")

    if "ES (Histórico) 0.05" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Histórico) 0.05"], label="ES Histórico 5%", linestyle="dotted", color="red")

    if "ES (Histórico) 0.01" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Histórico) 0.01"], label="ES Histórico 1%", linestyle="dotted", color="blue")

    if "ES (Parametrico) 0.05" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Parametrico) 0.05"], label="ES Paramétrico 5%", linestyle="dotted", color="orange")

    if "ES (Parametrico) 0.01" in series_seleccionadas:
        ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Parametrico) 0.01"], label="ES Paramétrico 1%", linestyle="dotted", color="green")

    # Configurar etiquetas y título
    ax.set_title("Rendimientos vs. VaR y ES", fontsize=15)
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)

    # Configurar leyenda y cuadrícula
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, title="Medidas seleccionadas",fontsize=14,title_fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

# e) --------------------------------------------------------------------------------

# Conteo y resumen de violaciones
    def Calcular_Violaciones(dfretornos, DataframeVaryES):
        # Alineamos los índices y eliminamos NaN
        df_aligned = DataframeVaryES.dropna()
        returns_aligned = dfretornos.reindex(df_aligned.index)
        
        resultados = []
        
        for columna in df_aligned.columns:
            # Extraemos alpha del nombre
            alpha = 0.05 if '0.05' in columna else 0.01 if '0.01' in columna else None
            
            # Calculamos violaciones
            violaciones = (returns_aligned < df_aligned[columna]).sum()
            total_datos = len(df_aligned[columna])
            porcentaje = (violaciones / total_datos) * 100
            
            #Para mostrar los resultados
            resultados.append({
                'Medida': columna.split()[0],  # VaR o ES
                'Método': columna.split()[1],
                'α': alpha,
                'Violaciones': violaciones,
                '% Observado': porcentaje,
                '% Esperado': alpha * 100 if alpha else 'N/A'
            })
        
        return pd.DataFrame(resultados)

    # Para ver la tabla
    st.subheader(f"Evaluación de Violaciones - {nombre_mostrado}")
    st.write("""
    En este análisis, comparamos la precisión de las estimaciones de riesgo usando VaR y ES calculados con dos métodos: histórico y paramétrico normal. 
    La tabla muestra el número de **violaciones** (cuando la pérdida real excede la estimación) para cada nivel de confianza.
    
    **Nota:** Una estimación adecuada debería tener un porcentaje de violaciones cercano al nivel de significancia α, idealmente menor al 2.5%.
    """)
    tabla_violaciones = Calcular_Violaciones(df_rendimientos, df_var_es_rolling)
    st.dataframe(tabla_violaciones.set_index("Medida"))

# f) --------------------------------------------------------------------------------

    def calcular_var_volatilidad_movil(serie_rendimientos, alphas=[0.05, 0.01], window=252):
        # Calculamos volatilidad móvil
        rolling_std = serie_rendimientos.rolling(window).std()
        
        resultados = pd.DataFrame(index=serie_rendimientos.index)
        
        for alpha in alphas:
            q_alpha = norm.ppf(alpha)
            resultados[f'VaR Vol Móvil ({alpha})'] = q_alpha * rolling_std
        
        return resultados.dropna()

    # Calculamos y mostramos
    var_vol_movil = calcular_var_volatilidad_movil(df_rendimientos)

    st.subheader(f"VaR con Volatilidad Móvil - {nombre_mostrado}")
    st.write("""
    En esta sección, se estima el VaR utilizando una volatilidad móvil bajo la suposición de una distribución normal considerando niveles de significancia de 0.05 y 0.01. 
    Además, se muestra la gráfica con los resultados y se evalúa la eficiencia de esta aproximación calculando el número de violaciones.
    """)

    # Graficamos
    fig,ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_rendimientos.index, df_rendimientos, label=f'Rendimientos {nombre_mostrado}', alpha=0.5)

    # Añadimos VaRs
    colors = ['red', 'blue']
    for i, alpha in enumerate([0.05, 0.01]):
        ax.plot(var_vol_movil.index, 
                var_vol_movil[f'VaR Vol Móvil ({alpha})'], 
                label=f'VaR {int(alpha*100)}%', 
                linestyle='-', 
                color=colors[i])

    ax.set_title('VaR con Volatilidad Móvil (Distribución Normal)',fontsize=15)
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # Violaciones para el nuevo VaR
    violaciones_vol_movil = Calcular_Violaciones(df_rendimientos, var_vol_movil)
    violaciones_vol_movil.iloc[:, 1] = 'Parametrico Normal'
    st.dataframe(violaciones_vol_movil.set_index("Medida"))




