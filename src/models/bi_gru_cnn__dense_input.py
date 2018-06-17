from keras.layers import Input, Flatten, Bidirectional, Dropout, concatenate, GRU, Activation, Conv1D, MaxPooling1D, \
    BatchNormalization
from keras.layers import LSTM, Dense
from keras.models import Model
import numpy as np

def get_model(config, data):
    max_sequence_length = data.max_full_sequence_length
    input_dims = data.input_dims
    num_classes = data.num_classes

    input_dropout = config['input_dropout']
    lstm_units = config['lstm_units']
    dense_units = config['dense_units']
    dense_units2 = config['dense_units2']
    cnn_kernel_size = int(config['cnn_kernel_size']) #def 5
    pool_size = int(config['pool_size']) #def 2
    filters = int(config['cnn_filters']) #def 256

    sequence_input = Input(shape=(max_sequence_length, input_dims[0]))
    x = sequence_input
    # x = BatchNormalization()(x)
    x = Dropout(input_dropout, input_shape=(max_sequence_length,))(x)
    x = Bidirectional(GRU(lstm_units, return_sequences=True, stateful=False))(x)
    x = Conv1D(filters=filters, kernel_size=cnn_kernel_size, padding='valid', activation='relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)

    x = Flatten()(x)

    input2 = Input(shape=(input_dims[1] * 2 * 3,))
    x2 = input2
    # x2 = BatchNormalization()(x2)
    if dense_units2:
        x2 = Dropout(0.1)(x2)
        x2 = Dense(dense_units2)(x2)
        x2 = Activation('relu')(x2)


    x = Dropout(0.2)(x)

    if dense_units:
        x = Dense(dense_units)(x)
        x = Activation('relu')(x)

    x_arr = [x, x2]
    if len(x_arr) > 1:
        x = concatenate(x_arr)
    else:
        x = x_arr[0]

    activation = 'softmax'
    output_units = num_classes
    if config['binary']:
        activation = 'sigmoid'
        output_units = 1

    preds = Dense(output_units, activation=activation)(x)


    return Model([sequence_input, input2], preds)

def get_x(x):
    return x

def get_x_train(training_data):
    return [training_data['x_train'], np.array(training_data['additional_layers']['x_train'][0])]

def get_x_val(training_data):
    return [training_data['x_val'], np.array(training_data['additional_layers']['x_val'][0])]

def get_x_val_test(test_data):
    return [test_data['x_val_test'], np.array(test_data['additional_layers']['x_val_test'][0])]

def get_cv_x_test(fold_data):
    return [fold_data['x_test'], np.array(fold_data['additional_layers']['x_test'][0])]


def get_second_input_vector_sequence(seq, data):

    return data.get_candidate_vector(seq)

def get_additional_layers_conf():

    return [
        get_second_input_vector_sequence
    ]