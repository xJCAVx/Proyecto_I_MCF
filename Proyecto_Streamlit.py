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
def obtener_datos(stocks):
    df = yf.download(stocks, start='2010-01-01',end='2025-04-03')['Close']
    return df

activos = ['GOOGL','AMZN','META','NFLX','TSLA']
df_precios=obtener_datos(activos)

st.title("Análisis de Riesgos Financieros")
st.header("Evaluación de métricas de riesgo y rendimiento para activos financieros")

activo_seleccionado = st.selectbox("Selecciona una activo", activos)

# b) --------------------------------------------------------------------------------

#Calculamos los rendimientos diarios, la media, el sesgo y el exceso de curtosis
@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()
df_rendimientos = calcular_rendimientos(df_precios)

if activo_seleccionado:
    st.subheader(f"Métricas de Rendimiento: {activo_seleccionado}")

    media = df_rendimientos[activo_seleccionado].mean()
    sesgo = skew(df_rendimientos[activo_seleccionado])
    curtosis = kurtosis(df_rendimientos[activo_seleccionado])
    
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

    var_es_results = calcular_var_es(df_rendimientos[activo_seleccionado])
    st.subheader("Cálculo de VaR y Expected Shortfall")
    st.dataframe(var_es_results)

# d) --------------------------------------------------------------------------------

# Cálculo de VaR y ES con ventanas móviles para alpha 0.05 y 0.01
    @st.cache_data
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
    df_var_es_rolling = rolling_var_es(df_rendimientos[activo_seleccionado])

    st.subheader("Rolling Window para VaR y ES (252 días)")

    opciones = [
        "Rendimientos",
        "VaR (Histórico) 0.05", "VaR (Histórico) 0.01",
        "VaR (Parametrico) 0.05", "VaR (Parametrico) 0.01",
        "ES (Histórico) 0.05", "ES (Histórico) 0.01",
        "ES (Parametrico) 0.05", "ES (Parametrico) 0.01"
    ]

    series_seleccionadas = st.multiselect("Selecciona las medidas a visualizar", opciones, default=opciones)

    # # Graficamos según la selección
    # fig, ax = plt.subplots(figsize=(15, 8))

    # if "Rendimientos" in series_seleccionadas:
    #     ax.plot(df_rendimientos[activo_seleccionado].index, df_rendimientos[activo_seleccionado], label='Rendimientos')

    # if "VaR (Histórico) 0.05" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Histórico) 0.05"], label='VaR Histórico 5%', linestyle='-', color='red')
    # if "VaR (Histórico) 0.01" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Histórico) 0.01"], label='VaR Histórico 1%', linestyle='-', color='blue')

    # if "VaR (Parametrico) 0.05" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Parametrico) 0.05"], label='VaR Paramétrico 5%', linestyle='-', color='black')
    # if "VaR (Parametrico) 0.01" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["VaR (Parametrico) 0.01"], label='VaR Paramétrico 1%', linestyle='-', color='green')

    # if "ES (Histórico) 0.05" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Histórico) 0.05"], label='ES Histórico 5%', linestyle=':', color='purple')
    # if "ES (Histórico) 0.01" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Histórico) 0.01"], label='ES Histórico 1%', linestyle=':', color='orange')

    # if "ES (Parametrico) 0.05" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Parametrico) 0.05"], label='ES Paramétrico 5%', linestyle=':', color='brown')
    # if "ES (Parametrico) 0.01" in series_seleccionadas:
    #     ax.plot(df_var_es_rolling.index, df_var_es_rolling["ES (Parametrico) 0.01"], label='ES Paramétrico 1%', linestyle=':', color='pink')

    # ax.legend(loc='upper center', ncol=4,fontsize=14)
    # ax.tick_params(axis="both", labelsize=14)
    # ax.set_title("Rendimientos vs. VaR y ES", fontsize=20)
    # ax.set_xlabel("Fecha", fontsize=14)
    # plt.tight_layout()
    # st.pyplot(fig)


    fig = px.line()  # Inicializamos una figura vacía

    if "Rendimientos" in series_seleccionadas:
        fig.add_scatter(x=df_rendimientos[activo_seleccionado].index,
                        y=df_rendimientos[activo_seleccionado],
                        mode='lines',
                        name='Rendimientos')

    if "VaR (Histórico) 0.05" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["VaR (Histórico) 0.05"],
                        mode='lines',
                        name='VaR Histórico 5%', line=dict(color='red'))

    if "VaR (Histórico) 0.01" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["VaR (Histórico) 0.01"],
                        mode='lines',
                        name='VaR Histórico 1%', line=dict(color='blue'))

    if "VaR (Parametrico) 0.05" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["VaR (Parametrico) 0.05"],
                        mode='lines',
                        name='VaR Paramétrico 5%', line=dict(color='black'))

    if "VaR (Parametrico) 0.01" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["VaR (Parametrico) 0.01"],
                        mode='lines',
                        name='VaR Paramétrico 1%', line=dict(color='green'))

    if "ES (Histórico) 0.05" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["ES (Histórico) 0.05"],
                        mode='lines',
                        name='ES Histórico 5%', line=dict(color='purple', dash='dot'))

    if "ES (Histórico) 0.01" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["ES (Histórico) 0.01"],
                        mode='lines',
                        name='ES Histórico 1%', line=dict(color='orange', dash='dot'))

    if "ES (Parametrico) 0.05" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["ES (Parametrico) 0.05"],
                        mode='lines',
                        name='ES Paramétrico 5%', line=dict(color='brown', dash='dot'))

    if "ES (Parametrico) 0.01" in series_seleccionadas:
        fig.add_scatter(x=df_var_es_rolling.index,
                        y=df_var_es_rolling["ES (Parametrico) 0.01"],
                        mode='lines',
                        name='ES Paramétrico 1%', line=dict(color='pink', dash='dot'))

    # Agregar título y etiquetas
    fig.update_layout(title="Rendimientos vs. VaR y ES",title=0.5,
                    xaxis_title="Fecha",
                    font=dict(size=15))
    
    fig.update_layout(
        legend=dict(
            title="Medidas seleccionadas",  # Cambia el título de la leyenda
            orientation="h",  # Orientación horizontal
            yanchor="top",  
            y=-0.3,  # Ajusta la posición vertical debajo de la gráfica
            xanchor="center",
            x=0.5
        )
    )

    # Mostrar gráfico en Streamlit
    fig.update_layout(
    hovermode="x unified",  # Muestra etiquetas al pasar el cursor sobre la gráfica
    dragmode="pan"  # Permite desplazar la gráfica con el mouse
    )

    st.plotly_chart(fig, use_container_width=True)






