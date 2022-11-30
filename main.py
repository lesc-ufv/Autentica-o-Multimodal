# operacoes data
from scipy import signal
from scipy import io
import random
import operacoes_dados as op
# data
import numpy as np
import pandas as pd
import leitura as lt
# IA
import tensorflow as tf
from keras import layers, models, losses
import models as md

modo = 'validacao'  # validacao, treino ou teste

def main():
    predicts = {}  # previsoes das redes

    # colocando parametros
    MAX = {'ppg': 32, 'ecg': 38}  # maximo do total de batimentos nos pedacos para a CNN-2
    SIZE = 3  # quantidade de batimentos por pessoa para o treino da CNN-2
    TOTAL = 107  # Total de individuos
    for sinal_avaliado in ['ecg', 'ppg']:
        cnns = {}

        # obtendo as listas de etiquetas e batimentos
        bat_test, label_test, bat_treino, label_trein, bat_vali, label_vali = lt.batimentos_e_labels(sinal_avaliado)

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
        cnns[sinal_avaliado+'CNN1'] = md.get_model_1(sinal_avaliado, bat_treino, label_trein,modo)

        if modo == 'treino':  # treinando cnn-2
            input_treino = cnns[sinal_avaliado+'CNN1'].predict(bat_treino)

            EPOCH = 32

            cnns[sinal_avaliado+'CNN2'] = md.run_model(input_treino, label_trein,
                                                    SIZE, EPOCH,
                                                    sinal_avaliado)

            cnns[sinal_avaliado+'CNN2'].save("models2/cnnmodelI"+sinal_avaliado)
        if modo == 'validacao' or modo == 'teste':  # obtendo cnn-2
            cnns[sinal_avaliado+'CNN2'] = models.load_model('models2/cnnmodelI'+sinal_avaliado)

            cnns[sinal_avaliado+'CNN2'].summary()

        if modo == 'teste':  # testa a cnn-2
            input_teste = cnns[sinal_avaliado+'CNN1'].predict(bat_test)
            predicts[sinal_avaliado] = op.testar(input_teste, label_test, cnns[sinal_avaliado+'CNN2'], SIZE,
                                              MAX[sinal_avaliado])

        if modo == 'validacao' or modo == 'treino':  # valida a cnn-2
            input_valid = cnns[sinal_avaliado+'CNN1'].predict(bat_vali)
            predicts[sinal_avaliado] = op.testar(input_valid, label_vali, cnns[sinal_avaliado+'CNN2'], SIZE,
                                              MAX[sinal_avaliado])

    for sinal_avaliado in ['ecg', 'ppg']:  # mostrando os resultados para ECG e PPG separadamente
        print("\n\n\n Somente o conjunto de dados "+sinal_avaliado+" \n\n\n")
        vetor = op.matriz_binario(predicts[sinal_avaliado], SIZE, TOTAL, MAX[sinal_avaliado])
        reduand = op.matriz_reduzida(vetor, SIZE, TOTAL, MAX[sinal_avaliado])
        op.faz_matriz_confusao(reduand, SIZE, 0, TOTAL, MAX[sinal_avaliado])

    print("\n\n\n ECG and PPG \n\n\n")  # mostra os resultados para ECG AND PPG
    vetor_and = op.and_vetores(predicts['ecg'], predicts['ppg'], SIZE, TOTAL)
    reduand = op.matriz_reduzida(vetor_and, SIZE, TOTAL, MAX['ecg'])
    op.faz_matriz_confusao(reduand, SIZE, 0, TOTAL, MAX['ecg'])

    # mostrando os resultados separados por conjuntos de dados
    print("\n\n\n Somente o conjunto de dados TROIKA \n\n\n")
    op.faz_matriz_confusao(reduand, SIZE, 95, TOTAL, MAX['ecg'])
    print("\n\n\n Somente o conjunto de dados BIDMC \n\n\n")
    op.faz_matriz_confusao(reduand, SIZE, 42, 95, MAX['ecg'])
    print("\n\n\n Somente o conjunto de dados Capnobase \n\n\n")
    op.faz_matriz_confusao(reduand, SIZE, 0, 42, MAX['ecg'])


if __name__ == '__main__':
    main()
