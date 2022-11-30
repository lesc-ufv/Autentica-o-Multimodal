import random
# data
import numpy as np
import pandas as pd
#operacoes dados
import operacoes_dados as op
from scipy import signal
from scipy import io
import const as c 

def batimentos_e_labels(sinal_avaliado):
    # le os batimentos e as distancias medias de cada pessoa no conjunto de dados CapnoBase
    lista_batimentos1 = [ler_dados(dado, 0, sinal_avaliado) for dado in range(1, 43)]
    distancias1 = [op.distancia_media(dado[sinal_avaliado], sinal_avaliado) for dado in lista_batimentos1]

    # le os batimentos e as distancias medias de cada pessoa no conjunto de dados BIDMC
    lista_batimentos2 = [ler_dados(dado, 1) for dado in range(1, 54)]
    distancias2 = [op.distancia_media(dado[sinal_avaliado], sinal_avaliado) for dado in lista_batimentos2]

    # le os batimentos e as distancias medias de cada pessoa no conjunto de dados TROIKA
    lista_batimentos3 = [ler_dados(dado, 2, sinal_avaliado) for dado in range(1, 13)]
    distancias3 = [op.distancia_media(dado[sinal_avaliado], sinal_avaliado) for dado in lista_batimentos3]

    # transforma os dados para variar de 0 a 1
    lista_batimentos1 = op.entre_0_1(lista_batimentos1, sinal_avaliado)
    lista_batimentos2 = op.entre_0_1(lista_batimentos2, sinal_avaliado)
    lista_batimentos3 = op.entre_0_1(lista_batimentos3, sinal_avaliado)

    # cria as listas com os batimentos para treino e labels para treino
    bat_treino = []
    label_trein = []

    # cria as listas com os batimentos para teste e labels para teste
    bat_test = []
    label_test = []

    # cria as listas com os batimentos para validacao e labels para validacao
    bat_vali = []
    label_vali = []

    # preenche cada uma das listas de acordo com o conjunto de dados CapnoBase
    for i in range(0, len(lista_batimentos1)):
        # obtem o sinal que sera usado
        lista_batimento = lista_batimentos1[i][sinal_avaliado].tolist()
        # obtem os picos
        peaks, _ = signal.find_peaks(lista_batimento, distance=distancias1[i])

        # colocando as etiquetas e batimentos de treino retirando 5 do bd
        if i <= 36:
            for p in range(0, int(len(peaks) * 0.7)):
                if 60 < peaks[p] < len(lista_batimento) - 60:
                    bat_treino.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                    label_trein.append(i)

        # colocando as etiquetas e batimentos de validacao
        for p in range(int(len(peaks) * 0.7), int(len(peaks) * 0.85)):
            if 60 < peaks[p] < len(lista_batimento) - 60:
                bat_vali.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                label_vali.append(i)

        # colocando as etiquetas e batimentos de teste
        for p in range(int(len(peaks) * 0.85), len(peaks)):
            if 60 < peaks[p] < len(lista_batimento) - 60:
                bat_test.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                label_test.append(i)

    # preenche cada uma das listas de acordo com o conjunto de dados BIDMC
    for i in range(0, len(lista_batimentos2)):
        lista_batimento = lista_batimentos2[i][sinal_avaliado].tolist()
        peaks, _ = signal.find_peaks(lista_batimento, distance=distancias2[i])

        # colocando as etiquetas e batimentos de treino retirando 5 do bd
        if i > 5:
            for p in range(0, int(len(peaks) * 0.7)):
                if 60 < peaks[p] < len(lista_batimento) - 60:
                    bat_treino.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                    label_trein.append(i + len(lista_batimentos1))

        # colocando as etiquetas e batimentos de validacao
        for p in range(int(len(peaks) * 0.7), int(len(peaks) * 0.85)):
            if 60 < peaks[p] < len(lista_batimento) - 60:
                bat_vali.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                label_vali.append(i + len(lista_batimentos1))

        # colocando as etiquetas e batimentos de teste
        for p in range(int(len(peaks) * 0.85), len(peaks)):
            if 60 < peaks[p] < len(lista_batimento) - 60:
                bat_test.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                label_test.append(i + len(lista_batimentos1))

    # preenche cada uma das listas de acordo com o conjunto de dados TROIKA
    for i in range(0, len(lista_batimentos3)):
        lista_batimento = lista_batimentos3[i][sinal_avaliado].tolist()
        peaks, _ = signal.find_peaks(lista_batimento, distance=distancias3[i])

        # Selecionando aleatoriamente a separacao entre validacao e treino
        vali_peaks = random.sample(range(0, len(peaks)), int(len(peaks) * 0.15))
        trein_peaks = [x for x in range(0, len(peaks)) if x not in vali_peaks]

        # colocando as etiquetas e batimentos de validacao
        for p in vali_peaks:
            if 60 < peaks[p] < len(lista_batimento) - 60:
                bat_vali.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                label_vali.append(i + len(lista_batimentos1) + len(lista_batimentos2))

        # colocando as etiquetas e batimentos de treino
        for p in trein_peaks:
            if 60 < peaks[p] < len(lista_batimento) - 60:
                bat_treino.append(lista_batimento[peaks[p] - 60:peaks[p] + 60])
                label_trein.append(i + len(lista_batimentos1) + len(lista_batimentos2))

    return bat_test, label_test, bat_treino, label_trein, bat_vali, label_vali

# Retorna o biosinal(PPG, ECG) da pessoa especificada de algum dos bancos de dados(0 = Capnobase, 1 = BIDMC)
def ler_dados(pessoa, banco_de_dados, sinal=""):
    df = pd.DataFrame()
    if banco_de_dados == 0:
        # A ordem cronologia dos dados nesse banco de dados
        number_list = [9, 15, 16, 18, 23, 28, 29, 30, 31, 32, 35, 38, 103, 104, 105, 115, 121, 122, 123, 125, 127, 128,
                       133, 134, 142, 147, 148, 149, 150, 309, 311, 312, 313, 322, 325, 328, 329, 330, 331, 332, 333,
                       370]
        # Extraindo do banco de dados o arquivo
        df = pd.read_csv(r"dataverse_csv/{}_8min_signal.csv".format(
            str(number_list[pessoa - 1]).zfill(4)))
        print(
            r"Lendo dataverse_csv/{}_8min_signal.csv".format(
                str(number_list[pessoa - 1]).zfill(4)))

        # retirando colunas nao usadas e renomeando as restantes
        df = df.drop(columns=["co2_y"])
        df = df.rename(columns={'pleth_y': "ppg", 'ecg_y': "ecg"})

    if banco_de_dados == 1:
        # Extraindo do banco de dados o arquivo
        df = pd.read_csv(r"bidmc_csv/bidmc_{}_Signals.csv".format(
            str(pessoa).zfill(2)))
        print(
            r"Lendo bidmc_csv/bidmc_{}_"
            r"Signals.csv".format(
                str(pessoa).zfill(2)))

        # Retirando as colunas nao usadas e renomeando as que são
        df = df.drop(columns=[" RESP"])
        df = df.drop(columns=["Time [s]"])
        if " MCL" in df.columns:
            df = df.drop(columns=[" MCL"])
        if " I" in df.columns:
            df = df.drop(columns=[" I"])
        if " V" in df.columns:
            df = df.drop(columns=[" V"])
        if " AVR" in df.columns:
            df = df.drop(columns=[" AVR"])
        if " ABP" in df.columns:
            df = df.drop(columns=[" ABP"])
        if " CVP" in df.columns:
            df = df.drop(columns=[" CVP"])
        df = df.rename(columns={' PLETH': "ppg", ' II': "ecg"})

    if banco_de_dados == 2:
        # extraindo os dados do banco de dados
        if pessoa == 1:
            tipo = 1
        else:
            tipo = 2
        mat = io.loadmat(r'troika/DATA_{}_TYPE{}.mat'.format(str(pessoa).zfill(2), str(tipo).zfill(2)))

        # Criando um dataframe com as respectivas estiquetas corretas
        d = {'ecg': mat['sig'][0], 'ppg': mat['sig'][1]}
        df = pd.DataFrame(data=d)
        print(r'Lendo troika/DATA_{}_TYPE{}.mat'.format(str(pessoa).zfill(2), str(tipo).zfill(2)))

    # Retirando o sinal nao desejado
    if sinal.lower() == "ppg":
        df = df.drop(columns=["ecg"])

    if sinal.lower() == "ecg":
        df = df.drop(columns=["ppg"])

    return df

