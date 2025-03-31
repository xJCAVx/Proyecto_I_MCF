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
    resultados =  pd.DataFrame(columns=['VaR (Normal)', 'ES (Normal)','VaR (t-Student)','VaR (Histórico)', 'VaR (Monte Carlo)', 'ES (Normal)', 'ES (t-Student)', 'ES (Histórico)', 'ES (Monte Carlo)'])
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
        VaR_MC = np.percentile(sim_return, 1 - NC)
        # ES Monte Carlo
        ES_MC = sim_return[sim_return <= VaR_MC].mean()
        
        resultados[f'VaR (Normal) {NC}'] = VaR_norm
        resultados[f'ES (Normal) {NC}'] = ES_norm
        resultados[f'VaR (t-Student) {NC}'] = VaR_t
        resultados[f'ES (t-Student) {NC}'] = ES_t
        resultados[f'VaR (Histórico) {NC}'] = VaR_hist
        resultados[f'ES (Histórico) {NC}'] = ES_hist
        resultados[f'VaR (Monte Carlo) {NC}'] = VaR_MC
        resultados[f'ES (Monte Carlo) {NC}'] = ES_MC
    
        resultados_df = pd.DataFrame(resultados, index=[f"{NC*100}%" for NC in NCS])

    return resultados_df

var_es_results = df_rendimientos.apply(calcular_var_es)

print("Resultados de VaR y ES para cada activo con diferentes niveles de condianza y distribuciones:")
print(var_es_results)


