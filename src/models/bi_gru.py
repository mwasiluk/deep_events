from keras.layers import Input, Flatten, Bidirectional, Dropout, concatenate, GRU, Activation
from keras.layers import LSTM, Dense
from keras.models import Model


# import pickle

def get_model(config, data):
    max_sequence_length = data.max_full_sequence_length
    input_dims = data.input_dims
    num_classes = data.num_classes

    input_dropout = config['input_dropout']
    lstm_units = config['lstm_units']
    dense_units = config['dense_units']

    sequence_input = Input(shape=(max_sequence_length, input_dims[0]))
    x = sequence_input
    x = Dropout(input_dropout, input_shape=(max_sequence_length,))(x)
    x = Bidirectional(GRU(lstm_units, return_sequences=True, stateful=False))(x)
    x = Flatten()(x)

    if dense_units:
        x = Dense(dense_units)(x)
        x = Activation('relu')(x)


    activation = 'softmax'
    output_units = num_classes
    if config['binary']:
        activation = 'sigmoid'
        output_units = 1

    preds = Dense(output_units, activation=activation)(x)


    return Model(sequence_input, preds)

def get_x(x):
    return x

def get_x_train(training_data):
    return training_data['x_train']

def get_x_val(training_data):
    return training_data['x_val']

def get_x_val_test(test_data):
    return test_data['x_val_test']

def get_cv_x_test(fold_data):
    return fold_data['x_test']

def get_additional_layers_conf():
    return []