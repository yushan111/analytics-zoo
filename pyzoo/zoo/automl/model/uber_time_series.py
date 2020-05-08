#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
import pandas as pd
import matplotlib.pyplot as plt

lookback = 5
# input_dim and output_dim is the num of variates
input_dim = 1
output_dim = input_dim

# for encoder decoder
latent_dims = [128, 32]
forecast_steps = 1

# for prediction net
dense_units = [128, 64, 16]
dropout_rates = [0.2, 0.2, 0.2]


def build_encoder_decoder():
    inputs = Input(shape=(lookback, input_dim))
    encode_1, h1, c1 = LSTM(latent_dims[0], activation='relu',
                            return_state=True, return_sequences=True)(inputs)
    encode_2, h2, c2 = LSTM(latent_dims[1], activation='relu',
                            return_sequences=False, return_state=True, name="encoder_output")(encode_1)

    decoder_inputs = Input(shape=(forecast_steps, output_dim))
    decode_1 = LSTM(latent_dims[1], activation='relu', return_sequences=True)\
        (decoder_inputs, initial_state=[h2, c2])
    decode_2 = LSTM(latent_dims[0], activation='relu', return_sequences=True)\
        (decode_1, initial_state=[h1, c1])
    print(decode_2.shape)
    outputs = TimeDistributed(Dense(input_dim))(decode_2)

    encoder_decoder = Model([inputs, decoder_inputs], outputs)
    encoder = Model(inputs=encoder_decoder.input[0],
                    outputs=encoder_decoder.get_layer("encoder_output").output)

    encoder_decoder.compile(optimizer='adam', loss='mse')
    # print(encoder_decoder.summary())
    return encoder_decoder, encoder


def build_prediction(pred_input_dim, pred_output_dim):
    model = Sequential()
    model.add(Dense(dense_units[0], activation="relu", input_dim=pred_input_dim))
    model.add(Dropout(dropout_rates[0]))
    model.add(Dense(dense_units[1], activation="relu"))
    model.add(Dropout(dropout_rates[1]))
    model.add(Dense(dense_units[2], activation="relu"))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(pred_output_dim, activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(1e-3))
    return model


def get_embedding(x, y, encoder_decoder, encoder):
    y_exp = np.expand_dims(y, axis=2)
    decoder_input = x[:, -forecast_steps:, :]
    encoder_decoder.fit([x, decoder_input], y_exp, epochs=4, batch_size=5, verbose=1)
    print(x.shape, y.shape, decoder_input.shape)
    yhat = encoder_decoder.predict([x, decoder_input], verbose=0)

    print('---Predicted---')
    print(np.round(yhat, 3))
    print('---Actual---')
    print(np.round(y, 3))

    # y = np.squeeze(y)[0:100]
    # yhat = np.squeeze(yhat)[0:100]
    # x = np.arange(y.shape[0])[0:100]
    # ax = plt.plot(x, y, "r--", x, yhat, "b--")
    # plt.show()

    encoder_outputs, h_state, c_state = encoder.predict(x)
    return c_state


if True:
    input_df = pd.DataFrame((np.sin(np.arange(1, 10000)/10) + 1)/2)
    ft = TimeSequenceFeatureTransformer()
    x, y = ft._roll_train(input_df, past_seq_len=lookback, future_seq_len=forecast_steps)
    encoder_decoder, encoder = build_encoder_decoder()
    embedding = get_embedding(x, y, encoder_decoder, encoder)
    embedding_average = np.mean(embedding, axis=1, keepdims=True)
    # print(embedding_average)
    #
    # we only use embedding for prediction, therefore input dim for prediction net is 1
    prediction_model = build_prediction(pred_input_dim=1, pred_output_dim=forecast_steps)
    prediction_model.fit(embedding_average, y, epochs=50)
    prediction_result = prediction_model.predict(embedding_average)
    # print(prediction_result)
    # print(y)
    # print(prediction_result.shape)

    y = np.squeeze(y)[0:200]
    yhat = np.squeeze(prediction_result)[0:200]
    x = np.arange(y.shape[0])[0:200]
    ax = plt.plot(x, y, "r--", x, yhat, "b--")
    plt.show()