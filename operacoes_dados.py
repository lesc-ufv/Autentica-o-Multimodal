import random
# operacoes data
from scipy import signal
from scipy import io
# data
import numpy as np
import pandas as pd

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

