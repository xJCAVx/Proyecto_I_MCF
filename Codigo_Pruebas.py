import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import kurtosis, skew, norm, t

def obtener_datos(stocks):
   
    df = yf.download(stocks, period = "1y")['Close']
    return df

def calcular_rendimientos(df):
    return df.pct_change().dropna()

# a) Carga y descarga de datos financieros
M7 = ['GOOGL','AMZN','META','NFLX','TSLA']
df_precios = yf.download(M7, start='2010-01-01', end= '2024-03-30', progress=False)

print("Datos de precios descargados")
print(df_precios.tail())

# b) Calculamos los rendimientos 
df_rendimientos = calcular_rendimientos(df_precios)

# Cálculo de rendimientos y estadísticas descriptivas
promedios = df_rendimientos.mean()
skewness = df_rendimientos.apply(lambda x:x.skew())
kurtosis_vals = df_rendimientos.apply(lambda x: x.kurtosis())

# Creamos un daata frame con los resultados
estadisticas = pd.DataFrame({'Media': promedios, 'Sesgo': skewness, 'Curtosis': kurtosis_vals})

# Mostramos las estadísticas
print("Estadísticas descriptivas de los rendimientos")
print(estadisticas)

# c) Calculo de VAR y ES con distintos métodos para múltiples valores de alpha
def calcular_var_es(df, alphas = [0.95, 0.975, 0.99]):
    resultados = {}
    df_size = len(df)

    for alpha in alphas:
        mean = df.mean()
        stdev = df.std()

        #VaR Paramétrico(normal)
        VaR_norm = norm.ppf(1 - alpha,mean,stdev)

        #VaR Paramétrico (t-student)
        nu = df_size - 1 # Grados de libertad
        VaR_t = t.ppf(1 - alpha, nu, mean, stdev)


        #VaR Histórico
        VaR_hist = df.quantile(1 - alpha)

        #VaR Monte Carlo 
        sim_return = np.random.normal(mean, stdev, 100000)
        VaR_MC = np.percentile(sim_return, (1 - alpha) * 100)

        #ES Normal
        ES_norm = mean - (stdev * norm.pdf(norm.ppf(alpha)) / (1 - alpha))

        #ES t-student
        ES_t = mean - (stdev * t.pdf(t.ppf(alpha,nu),nu) / (1-alpha))

        resultados[f'VaR (Normal){alpha}'] = VaR_norm
        resultados[f'VaR (t-Student){alpha}'] = VaR_t
        resultados[f'VaR (Histórico){alpha}'] = VaR_hist
        resultados[f'ES (Normal){alpha}'] = ES_norm
        resultados[f'ES (t-studen){alpha}'] = ES_t

        return pd.DataFrame(resultados)
    
    var_es_results = df_rendimientos.apply(calcular_var_es)

    print("Resultados de VaR y ES para cada activo con diferentes niveles de condianza y distribuciones:")
    print(var_es_results)