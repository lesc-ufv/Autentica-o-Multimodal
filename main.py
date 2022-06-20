# operacoes data
from scipy import signal
from scipy import io
import random
# data
import numpy as np
import pandas as pd
# IA
import tensorflow as tf
from keras import layers, models, losses

modo = 'validacao'  # validacao, treino ou teste


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


# Faz a lista de batimentos variar de 0 a 1
def entre_0_1(lista_batimentos, tipo_sinal):
    # Extraindo o minimo e maximo dos respectivos batimentos
    maxi = lista_batimentos[0][tipo_sinal].max()
    mini = lista_batimentos[0][tipo_sinal].min()

    # Transformando todos os sinais para variar de 0 a 1
    for i in range(0, len(lista_batimentos)):
        lista_batimentos[i][tipo_sinal] = (lista_batimentos[i][tipo_sinal] - mini)/(maxi - mini)
    return lista_batimentos


# Calculando a distancia media dos picos
def distancia_media(dado, tipo_sinal):
    peaks, _ = signal.find_peaks(dado)

    zip_iterator = zip(peaks, dado[peaks])

    a_dictionary = dict(zip_iterator)

    peaks = sorted(peaks, key=a_dictionary.__getitem__)

    if tipo_sinal == 'ecg':
        peaks = peaks[int(len(peaks)*(5/6)):len(peaks)]

    peaks.sort()

    soma_distancias = 0
    for i in range(0, len(peaks)-1):
        soma_distancias += peaks[i+1] - peaks[i]

    return soma_distancias/(len(peaks)-1)


def batimentos_e_labels(sinal_avaliado):
    # le os batimentos e as distancias medias de cada pessoa no conjunto de dados CapnoBase
    lista_batimentos1 = [ler_dados(dado, 0, sinal_avaliado) for dado in range(1, 43)]
    distancias1 = [distancia_media(dado[sinal_avaliado], sinal_avaliado) for dado in lista_batimentos1]

    # le os batimentos e as distancias medias de cada pessoa no conjunto de dados BIDMC
    lista_batimentos2 = [ler_dados(dado, 1) for dado in range(1, 54)]
    distancias2 = [distancia_media(dado[sinal_avaliado], sinal_avaliado) for dado in lista_batimentos2]

    # le os batimentos e as distancias medias de cada pessoa no conjunto de dados TROIKA
    lista_batimentos3 = [ler_dados(dado, 2, sinal_avaliado) for dado in range(1, 13)]
    distancias3 = [distancia_media(dado[sinal_avaliado], sinal_avaliado) for dado in lista_batimentos3]

    # transforma os dados para variar de 0 a 1
    lista_batimentos1 = entre_0_1(lista_batimentos1, sinal_avaliado)
    lista_batimentos2 = entre_0_1(lista_batimentos2, sinal_avaliado)
    lista_batimentos3 = entre_0_1(lista_batimentos3, sinal_avaliado)

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


def get_model_1(sinal_avaliado, bat_treino, label_trein):
    if modo == 'treino':
        # ppg
        if sinal_avaliado == 'ppg':
            # criação do modelo da cnn
            model = models.Sequential()
            model.add(layers.Conv1D(filters=32, kernel_size=4, activation="relu", input_shape=(120, 1)))
            model.add(layers.Conv1D(filters=32, kernel_size=4, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(filters=32, kernel_size=4, activation='relu'))
            model.add(layers.Conv1D(filters=32, kernel_size=4, activation='relu'))
            model.add(layers.MaxPooling1D(2))

            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(107))

            model.summary()

            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

            # colocando os dados para treino
            model.fit(bat_treino, label_trein, epochs=30)

            # salvando
            model.save("models2/cnnmodel"+sinal_avaliado)
        if sinal_avaliado == 'ecg':
            # ecg
            # criação do modelo da cnn
            model = models.Sequential()
            model.add(layers.Conv1D(filters=64, kernel_size=4, activation="relu", input_shape=(120, 1)))
            model.add(layers.Conv1D(filters=64, kernel_size=4, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(filters=64, kernel_size=4, activation='relu'))
            model.add(layers.Conv1D(filters=64, kernel_size=4, activation='relu'))
            model.add(layers.MaxPooling1D(2))

            model.add(layers.Flatten())
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dense(107))

            model.summary()

            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

            # colocando os dados para treino
            model.fit(bat_treino, label_trein, epochs=30)

            # salvando
            model.save("models2/cnnmodel"+sinal_avaliado)
    if modo == 'teste' or modo == 'validacao':
        # obtendo o modelo previamente treinado
        model = models.load_model('models2/cnnmodel'+sinal_avaliado)

        model.summary()
    return model


# Prepara as tentativas de cada batimento para cada
def combine(vetor, limite):
    lista = []
    labels = []
    for i in range(len(vetor)):
        for j in range(len(vetor)):
            lista.append(np.array([vetor[i], vetor[j]]))
            labels.append(int(i // limite == j // limite))
    return np.array(lista), np.array(labels)


# limita o tamanho dos dados
def limita(vetor, labels, limite_inf, limite_sup):
    vetor_limitado = []
    for i in range(len(vetor)):
        if i == 0 or labels[i] != labels[i - 1]:
            qnt = 0
        if limite_inf <= qnt < limite_sup:
            vetor_limitado.append(vetor[i])
        qnt += 1
    return vetor_limitado


# coloca os dados dentro da CNN1
def run_model(input_treino, label_trein,
              SIZE, EPOCH, sinal):
    modelI = cria_modelo(sinal)
    if sinal == 'ecg':
        MAX = 205
    if sinal == 'ppg':
        MAX = 300
    with tf.device('GPU'):
        for limite in range(SIZE, MAX, SIZE):
            input_treino_r = limita(input_treino, label_trein, limite - SIZE, limite)

            comb_trein, lcomb_treino = combine(input_treino_r, SIZE)

            lcomb_treino = np.expand_dims(lcomb_treino, axis=1)
            comb_trein = np.expand_dims(comb_trein, axis=3)

            modelI.fit(comb_trein, lcomb_treino, epochs=EPOCH)
    return modelI


# Cria o modelo da cnn-2
def cria_modelo(sinal_avaliado):
    if sinal_avaliado == 'ppg':
        x = [32, 32, 32, 32, 128]
    if sinal_avaliado == 'ecg':
        x = [64, 64, 64, 64, 128]

    modelI = models.Sequential()
    modelI.add(layers.Conv1D(x[0], kernel_size=2, activation="relu", input_shape=(2, 107, 1)))
    modelI.add(layers.Conv1D(x[1], kernel_size=2, activation="relu"))
    modelI.add(layers.MaxPooling2D(pool_size=(1, 2)))
    modelI.add(layers.Conv1D(x[2], kernel_size=2, activation="relu"))
    modelI.add(layers.MaxPooling2D(pool_size=(1, 2)))
    modelI.add(layers.Conv1D(x[3], kernel_size=2, activation="relu"))
    modelI.add(layers.Flatten())
    modelI.add(layers.Dense(x[4], activation="relu"))
    modelI.add(layers.Dense(2))

    modelI.summary()

    modelI.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return modelI


# Coloca os dados de etste na CNN-2 e retorna suas predicoes
def testar(input_teste, label_test, modelo, SIZE, MAX):
    predict = []
    for limite in range(SIZE, MAX, SIZE):

        input_teste_r = limita(input_teste, label_test, limite - SIZE, limite)

        comb_teste, lcomb_teste = combine(input_teste_r, SIZE)

        comb_teste = np.expand_dims(comb_teste, axis=3)

        predict.append(modelo.predict(comb_teste))
    return predict


# retorna a matriz confusao e outras metricas
def faz_matriz_confusao(matriz_red, SIZE, INICIO, TOTAL, MAX):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(INICIO, TOTAL):
        for j in range(INICIO, TOTAL):
            if i == j:
                TP += matriz_red[i][j]
                FN += (SIZE*SIZE*(int(MAX/SIZE))) - matriz_red[i][j]
            else:
                FP += matriz_red[i][j]
                TN += (SIZE*SIZE*(int(MAX/SIZE))) - matriz_red[i][j]

    print("         Aprovado Desaprovado")
    print("Aprovar", TP, FN)
    print("Desaprovar", FP, TN)
    print("\nAcurácia:", (TP+TN)/(TP+FP+TN+FN))
    print("Precision:", TP/(FP+TP))
    print("Recall: ", (TP/(SIZE*SIZE*(int(MAX/SIZE))))/(TOTAL-INICIO))
    print('FAR:', FP/(FP+TN))
    print('FRR:', FN/(FN+TP))


# faz a operacao binaria AND entre PPG e ECG
def and_vetores(ECG, PPG, SIZE, TOTAL):
    vetor_and = []
    for j in range(int(32/SIZE)):
        vetor_and.append([])
        for i in range(SIZE*SIZE*TOTAL*TOTAL):
            if np.argmax(ECG[j][i]) == 1 and np.argmax(PPG[j][i]) == 1:
                vetor_and[j].append(1)
            else:
                vetor_and[j].append(0)
    for j in range(int(32/SIZE), int(38/SIZE)):
        vetor_and.append([])
        for i in range(SIZE*SIZE*TOTAL*TOTAL):
            vetor_and[j].append(np.argmax(ECG[j][i]))
    return vetor_and


# Retorna uma matriz com o neuronio final mais ativo
def matriz_binario(predict, SIZE, TOTAL, MAX):
    matrizb = []
    for j in range(int(MAX/SIZE)):
        matrizb.append([])
        for i in range(SIZE*SIZE*TOTAL*TOTAL):
            matrizb[j].append(np.argmax(predict[j][i]))
    return matrizb


# Reduz a matriz num fator de SIZE
def matriz_reduzida(predict_binario, SIZE, TOTAL, MAX):
    matriz = [[0 for i in range(TOTAL)] for j in range(TOTAL)]
    for l in range(int(MAX/SIZE)):
        for i in range(TOTAL):
            for j in range(TOTAL):
                for k in range(SIZE):
                    for c in range(SIZE):
                        matriz[i][j] += predict_binario[l][(j * SIZE) + (i * TOTAL * SIZE * SIZE) + c + (k * TOTAL *
                                                                                                         SIZE)]
    return matriz


def main():
    predicts = {}  # previsoes das redes

    # colocando parametros
    MAX = {'ppg': 32, 'ecg': 38}  # maximo do total de batimentos nos pedacos para a CNN-2
    SIZE = 3  # quantidade de batimentos por pessoa para o treino da CNN-2
    TOTAL = 107  # Total de individuos
    for sinal_avaliado in ['ecg', 'ppg']:
        cnns = {}

        # obtendo as listas de etiquetas e batimentos
        bat_test, label_test, bat_treino, label_trein, bat_vali, label_vali = batimentos_e_labels(sinal_avaliado)

        # transformando as listas em numpy arrays
        bat_treino = np.array(bat_treino)
        label_trein = np.array(label_trein)
        bat_test = np.array(bat_test)
        label_test = np.array(label_test)

        # expandindo as dimensoes
        bat_test = np.expand_dims(bat_test, axis=2)
        label_trein = np.expand_dims(label_trein, axis=1)
        bat_treino = np.expand_dims(bat_treino, axis=2)
        label_test = np.expand_dims(label_test, axis=1)

        # obtendo a cnn-1
        cnns[sinal_avaliado+'CNN1'] = get_model_1(sinal_avaliado, bat_treino, label_trein)

        if modo == 'treino':  # treinando cnn-2
            input_treino = cnns[sinal_avaliado+'CNN1'].predict(bat_treino)

            EPOCH = 32

            cnns[sinal_avaliado+'CNN2'] = run_model(input_treino, label_trein,
                                                    SIZE, EPOCH,
                                                    sinal_avaliado)

            cnns[sinal_avaliado+'CNN2'].save("models2/cnnmodelI"+sinal_avaliado)
        if modo == 'validacao' or modo == 'teste':  # obtendo cnn-2
            cnns[sinal_avaliado+'CNN2'] = models.load_model('models2/cnnmodelI'+sinal_avaliado)

            cnns[sinal_avaliado+'CNN2'].summary()

        if modo == 'teste':  # testa a cnn-2
            input_teste = cnns[sinal_avaliado+'CNN1'].predict(bat_test)
            predicts[sinal_avaliado] = testar(input_teste, label_test, cnns[sinal_avaliado+'CNN2'], SIZE,
                                              MAX[sinal_avaliado])

        if modo == 'validacao' or modo == 'treino':  # valida a cnn-2
            input_valid = cnns[sinal_avaliado+'CNN1'].predict(bat_vali)
            predicts[sinal_avaliado] = testar(input_valid, label_vali, cnns[sinal_avaliado+'CNN2'], SIZE,
                                              MAX[sinal_avaliado])

    for sinal_avaliado in ['ecg', 'ppg']:  # mostrando os resultados para ECG e PPG separadamente
        print("\n\n\n Somente o conjunto de dados "+sinal_avaliado+" \n\n\n")
        vetor = matriz_binario(predicts[sinal_avaliado], SIZE, TOTAL, MAX[sinal_avaliado])
        reduand = matriz_reduzida(vetor, SIZE, TOTAL, MAX[sinal_avaliado])
        faz_matriz_confusao(reduand, SIZE, 0, TOTAL, MAX[sinal_avaliado])

    print("\n\n\n ECG and PPG \n\n\n")  # mostra os resultados para ECG AND PPG
    vetor_and = and_vetores(predicts['ecg'], predicts['ppg'], SIZE, TOTAL)
    reduand = matriz_reduzida(vetor_and, SIZE, TOTAL, MAX['ecg'])
    faz_matriz_confusao(reduand, SIZE, 0, TOTAL, MAX['ecg'])

    # mostrando os resultados separados por conjuntos de dados
    print("\n\n\n Somente o conjunto de dados TROIKA \n\n\n")
    faz_matriz_confusao(reduand, SIZE, 95, TOTAL, MAX['ecg'])
    print("\n\n\n Somente o conjunto de dados BIDMC \n\n\n")
    faz_matriz_confusao(reduand, SIZE, 42, 95, MAX['ecg'])
    print("\n\n\n Somente o conjunto de dados Capnobase \n\n\n")
    faz_matriz_confusao(reduand, SIZE, 0, 42, MAX['ecg'])


if __name__ == '__main__':
    main()
