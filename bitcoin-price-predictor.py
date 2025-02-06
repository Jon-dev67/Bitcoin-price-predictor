import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import smtplib  
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Baixando os dados históricos do Bitcoin (últimos 2 anos, intervalo diário)
df = yf.download("BTC-USD", period="2y", interval="1d")

# Criando indicadores técnicos para análise do preço do Bitcoin
# Média móvel simples de 9 dias
df['SMA_9'] = df['Close'].rolling(window=9).mean()
# Média móvel simples de 21 dias
df['SMA_21'] = df['Close'].rolling(window=21).mean()
# Média exponencial de 9 dias
df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

# Índice de Força Relativa (RSI) - 14 dias
delta = df['Close'].diff()  # Diferença entre o preço de fechamento atual e anterior
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Ganho (apenas variações positivas)
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Perda (apenas variações negativas)
rs = gain / loss  # Razão entre ganho e perda
df['RSI'] = 100 - (100 / (1 + rs))  # Fórmula do RSI

# MACD - Moving Average Convergence Divergence (Convergência e Divergência de Médias Móveis)
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
# Linha de sinal do MACD (média exponencial de 9 dias do MACD)
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Remover valores nulos resultantes da criação dos indicadores
df.dropna(inplace=True)

#Normalizar os preços e indicadores técnicos
# Normalização dos dados para o intervalo entre 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[['Close', 'SMA_9', 'SMA_21', 'EMA_9', 'RSI', 'MACD', 'MACD_Signal']])

# Criar sequências de dados para treinamento
# Função que cria sequências de dados para previsão, dado o comprimento da sequência e a quantidade de dias a serem previstos
def create_sequences(data, seq_length, prediction_days):
    X, y = [], []  # Listas para armazenar as sequências de entrada (X) e as saídas (y)
    for i in range(len(data) - seq_length - prediction_days + 1):
        X.append(data[i:i+seq_length])  # Sequência de dados de entrada
        y.append(data[i+seq_length:i+seq_length+prediction_days, 0])  # Sequência de preços futuros
    return np.array(X), np.array(y)

seq_length = 60  # Comprimento das sequências de entrada (60 dias)
prediction_days = 5  # Número de dias a serem previstos
X, y = create_sequences(df_scaled, seq_length, prediction_days)

# Divisão dos dados em treinamento (80%) e teste (20%)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]  # Dados de treino
X_test, y_test = X[split:], y[split:]  # Dados de teste

# Criar o modelo LSTM
# Modelo sequencial com duas camadas LSTM, camadas Dropout para evitar overfitting, e camadas Dense para previsão final
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),  # LSTM com 128 unidades, retornando sequências
    Dropout(0.2),  # Dropout de 20% para evitar overfitting
    LSTM(128, return_sequences=False),  # Segunda camada LSTM sem retornar sequências
    Dropout(0.2),  # Dropout de 20% novamente
    Dense(50, activation='relu'),  # Camada densa com 50 neurônios e ReLU
    Dense(prediction_days)  # Camada de saída com um neurônio por dia previsto
])

# Compilando o modelo com otimizador 'adam' e função de perda 'mean_squared_error'
model.compile(optimizer="adam", loss="mean_squared_error")
# Treinando o modelo com 100 épocas e tamanho de batch de 32
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Fazer previsões
# Usando o modelo treinado para prever os dados de teste
predictions = model.predict(X_test)

#  Desnormalizar os dados (converter os dados escalados para seus valores originais)
# Ajustando as previsões para o formato original
predictions = scaler.inverse_transform(
    np.hstack((predictions, np.zeros((len(predictions), 6))))
)[:, :prediction_days]

# Convertendo os valores reais do teste para o formato original
y_test_original = scaler.inverse_transform(
    np.hstack((y_test, np.zeros((len(y_test), 6))))
)[:, :prediction_days]

# Sistema de alerta (verifica se a previsão é alta ou queda acima de um threshold e envia alerta)
def verificar_alerta(predictions, last_price, threshold=5):
    """
    Verifica se houve uma alta ou queda acima do threshold (%) e envia alerta por e-mail.
    """
    percentual = ((predictions[0] - last_price) / last_price) * 100  # Calculando a variação percentual
    if percentual > threshold:  # Se a variação for maior que o threshold
        alerta = f"ALERTA: Previsão de ALTA de {percentual:.2f}%!"
        print(alerta)
        enviar_email(alerta)  # Envia o alerta por e-mail
    elif percentual < -threshold:  # Se a variação for menor que o threshold (queda)
        alerta = f"ALERTA: Previsão de QUEDA de {percentual:.2f}%!"
        print(alerta)
        enviar_email(alerta)  # Envia o alerta por e-mail

# Função para enviar e-mails
def enviar_email(mensagem):
    """
    Envia um e-mail de alerta com a mensagem fornecida.
    """
    email_remetente = "devbackendpythonflask@gmail.com"  
    senha = "43996457979"  
    email_destinatario = "devbackendpythonflask@gmail.com"  

    msg = MIMEMultipart()  # Criando o objeto de mensagem
    msg["From"] = email_remetente
    msg["To"] = email_destinatario
    msg["Subject"] = " Alerta de Preço Bitcoin"  # Assunto do e-mail

    msg.attach(MIMEText(mensagem, "plain"))  # Corpo da mensagem

    try:
        # Configuração do servidor SMTP e envio do e-mail
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Inicia a conexão segura
        server.login(email_remetente, senha)  # Login no servidor de e-mail
        server.sendmail(email_remetente, email_destinatario, msg.as_string())  # Envia o e-mail
        server.quit()  # Fecha a conexão com o servidor
        print("E-mail de alerta enviado!")
    except Exception as e:
        print(f"erro ao enviar e-mail: {e}")

# Aplicar o sistema de alerta ao último preço real e à previsão do primeiro dia
last_price = df['Close'].iloc[-1]  # Último preço real conhecido
verificar_alerta(predictions[0], last_price)  # Verifica e envia alerta se necessário

# Visualizar os resultados (gráfico comparando os preços reais e previstos)
plt.figure(figsize=(14, 6))  # Define o tamanho do gráfico
for i in range(prediction_days):
    plt.plot(y_test_original[:, i], label=f"Real Dia {i+1}", linestyle="-")  # Preço real de cada dia
    plt.plot(predictions[:, i], label=f"Previsto Dia {i+1}", linestyle="dashed")  # Preço previsto de cada dia

plt.xlabel("Tempo")
plt.ylabel("Preço BTC")
plt.title("Previsão de Preço do Bitcoin para os Próximos 5 Dias")
plt.legend()  
plt.show()  