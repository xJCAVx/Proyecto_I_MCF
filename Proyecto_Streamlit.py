# Librerias -------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import kurtosis, skew, norm, t

# a) --------------------------------------------------------------------------------

# Descargamos la información de los activos desde 2010
def obtener_datos(stocks):
    df = yf.download(stocks, period="1y")['Close']
    return df
#start='2010-01-01'
activos = ['GOOGL','AMZN','META','NFLX','TSLA']
df_precios=obtener_datos(activos)

st.title("TITULO")
st.header("SUBTITULO")

activo_seleccionado = st.selectbox("Selecciona una activo", activos)

# b) --------------------------------------------------------------------------------

#Calculamos los rendimientos diarios, la media, el sesgo y el exceso de curtosis

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
    st.subheader("Subtitulo")
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

    # Aplicamos la función corregida
    df_var_es_rolling = rolling_var_es(df_rendimientos[activo_seleccionado])

    # Graficamos las ganancias y pérdidas junto con VaR y ES
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_rendimientos[activo_seleccionado].index, df_rendimientos[activo_seleccionado], label='Rendimientos')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'VaR (Histórico) 0.05'], label='VaR Histórico 5%', linestyle='dashed', color='red')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'VaR (Histórico) 0.01'], label='VaR Histórico 1%', linestyle='dashed', color='blue')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'VaR (Parametrico) 0.05'], label='VaR Paramétrico 5%', linestyle='dashed', color='black')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'VaR (Parametrico) 0.01'], label='VaR Paramétrico 1%', linestyle='dashed', color='green')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'ES (Histórico) 0.05'], label='ES Histórico 5%', linestyle='dashed', color='purple')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'ES (Histórico) 0.01'], label='ES Histórico 1%', linestyle='dashed', color='orange')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'ES (Parametrico) 0.05'], label='ES Paramétrico 5%', linestyle='dashed', color='brown')
    ax.plot(df_var_es_rolling.index, df_var_es_rolling[f'ES (Parametrico) 0.01'], label='ES Paramétrico 1%', linestyle='dashed', color='pink')
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Rendimientos vs. VaR y ES")

    st.pyplot(fig)











