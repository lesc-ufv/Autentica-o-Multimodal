# data
import numpy as np
import pandas as pd
# IA
import tensorflow as tf
from keras import layers, models, losses

def get_model_1(sinal_avaliado, bat_treino, label_trein, modo):
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


