import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, t

def calcular_rendimientos(df):
    return df.pct_change().dropna()

# a) Carga y descarga de datos financieros
M7 = ['GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA']
df_precios = yf.download(M7, start='2010-01-01', end='2025-03-30', progress=False)['Close']

print("Datos de precios descargados")
print(df_precios.tail())

# b) Calculamos los rendimientos 
df_rendimientos = calcular_rendimientos(df_precios)

# Cálculo de rendimientos y estadísticas descriptivas
promedios = df_rendimientos.mean()
skewness = df_rendimientos.apply(lambda x: x.skew())
kurtosis_vals = df_rendimientos.apply(lambda x: x.kurtosis())

# Creamos un DataFrame con los resultados
estadisticas = pd.DataFrame({'Media': promedios, 'Sesgo': skewness, 'Curtosis': kurtosis_vals})

# Mostramos las estadísticas
print("Estadísticas descriptivas de los rendimientos")
print(estadisticas)

# c) Calculo de VAR y ES con distintos métodos para múltiples valores de alpha
def calcular_var_es(df, NCS=[0.05, 0.025, 0.01]):
    resultados = {}
    df_size = len(df)

    for NC in NCS:
        mean = df.mean()
        stdev = df.std()

        # VaR Paramétrico (normal)
        VaR_norm = norm.ppf(1 - NC, mean, stdev)

        # VaR Paramétrico (t-student)
        nu = df_size - 1  # Grados de libertad
        VaR_t = t.ppf(1 - NC, nu, loc=mean, scale=stdev)

        # VaR Histórico
        VaR_hist = df.quantile(1 - NC)

        # VaR Monte Carlo
        sim_return = np.random.normal(mean, stdev, 100000)
        VaR_MC = np.percentile(sim_return, (1 - NC) * 100)

        # ES Normal
        ES_norm = mean - (stdev * norm.pdf(norm.ppf(NC)) / (1 - NC))

        # ES t-student
        ES_t = mean - (stdev * t.pdf(t.ppf(NC, nu), nu) / (1 - NC))

        # ES Histórico
        ES_hist = df[df <= VaR_hist].mean()

        # ES Monte Carlo
        ES_MC = sim_return[sim_return <= VaR_MC].mean()

        resultados[f'VaR (Normal) {NC}'] = VaR_norm
        resultados[f'VaR (t-Student) {NC}'] = VaR_t
        resultados[f'VaR (Histórico) {NC}'] = VaR_hist
        resultados[f'VaR (Monte Carlo) {NC}'] = VaR_MC
        resultados[f'ES (Normal) {NC}'] = ES_norm
        resultados[f'ES (t-Student) {NC}'] = ES_t
        resultados[f'ES (Histórico) {NC}'] = ES_hist
        resultados[f'ES (Monte Carlo) {NC}'] = ES_MC

    return pd.Series(resultados)

# Aplicamos la función sobre cada columna de rendimientos de cada activo individualmente
var_es_results = pd.DataFrame()

# Iteramos sobre cada columna de df_rendimientos
for column in df_rendimientos.columns:
    var_es_results[column] = calcular_var_es(df_rendimientos[column])

# Mostramos los resultados
print("\nResultados de VaR y ES para cada activo con diferentes niveles de confianza y distribuciones:")
print(var_es_results)