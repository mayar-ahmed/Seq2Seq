import pandas as pd
import numpy as np
from keras.models import load_model
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Lambda
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import Callback
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os


##################################################
# this code contains an lstm-class fro building an training stacked stm net to predict next value
# use it on daily data,weekly data to see if one step forecast is possible

class PeriodicLogger(Callback):

    def on_train_begin(self, logs={}):
        # Initialization code
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        if self.epochs % 10== 0:
            print("epoch{}-- loss : {}".format(self.epochs,logs.get('loss')))



class Seq2Seq():
    # takes parameters needed for lstm
    def __init__(self, exp, n_input, n_forecast, lstm_cells,data,tsplit=0.8):
        """

        :param exp: experiment number
        :param n_input:number of input steps
        :param n_forecast: number of points to forecast
        :param data:pandas series/data frame containing values
        :param lstm_cells:number of cells
        :param tsplit:fraction of data to use for training/val and the rest is test

        """
        self.exp_dir = 'seq2seq_ntf_experiments/exp{}/'.format(exp)
        self.seq_in = n_input
        self.seq_out=n_forecast
        self.data = data
        self.neurons=lstm_cells
        self.tsplit=tsplit
        # number of data samples
        self.n = data.shape[0]

        print("experiment directory: ", self.exp_dir)
        print('model initialized with input sequence length : {} , output sequence length: {} , full data length : {}\n'.format(self.seq_in,self.seq_out, self.n))

        print('creating experiment directory\n')
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.file = open(self.exp_dir + 'expriment_params.txt', 'w')
        self.file.write('input sequence length : {} , output sequence length: {} , full data length : {}\n'.format(self.seq_in,self.seq_out, self.n))

        print("preparing data for lstm")
        self.prepare_data()

    # prepare data shape for lstm (split, normalize, reshape)

    def prepare_data(self):

        # split into train and validate
        num_train = int(self.tsplit* self.n)

        train_data = self.data[0:num_train]
        test_data = self.data[num_train:]

        print("training sequence length: {} , validation sequence length: {} ".format(num_train, self.n - num_train))
        print("train and validation data")
        plt.plot(train_data, 'b', label='training')
        plt.plot(test_data , 'r' , label='testing')
        plt.legend()
        plt.show()

        values = self.data.value.values.reshape(-1,1)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler= self.scaler.fit(values[:num_train])

        scaled_values=self.scaler.transform(values)


        print('time series after scaling')
        plt.plot(scaled_values)
        plt.show()

        x_timeseries = scaled_values
        y_timeseries = scaled_values.reshape(-1,1)

        #reshaping data using input and output sequence length

        self.x_train, self.y_train, self.x_test, self.y_test = self.to_supervised(x_timeseries, y_timeseries, self.seq_in, self.seq_out, split = self.tsplit)
        print('\nsize of x_train, y_train, x_test, y_test:')
        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape )


        print("")
        s1 = "x_train shape : {} , y_train shape : {} \n".format(self.x_train.shape, self.y_train.shape)
        s2 = "x_test shape : {} , y_test shape : {} \n".format(self.x_test.shape, self.y_test.shape)
        print(s1)
        print(s2)

        self.file.write(s1)
        self.file.write(s2)

    def to_supervised(self,x_timeseries, y_timeseries, n_memory_step, n_forcast_step, split = None):
        '''
         this function is made by developers of deep series library

            x_timeseries: input time series data, numpy array, (time_step, features)
            y_timeseries: target time series data,  numpy array, (time_step, features)
            n_memory_step: number of memory step in supervised learning, int
            n_forcast_step: number of forcase step in supervised learning, int
            split: portion of data to be used as train set, float, e.g. 0.8
        '''
        assert len(x_timeseries.shape) == 2, 'x_timeseries must be shape of (time_step, features)'
        assert len(y_timeseries.shape) == 2, 'y_timeseries must be shape of (time_step, features)'

        input_step, input_feature = x_timeseries.shape
        output_step, output_feature = y_timeseries.shape
        assert input_step == output_step, 'number of time_step of x_timeseries and y_timeseries are not consistent!'

        n_RNN_sample=input_step-n_forcast_step-n_memory_step+1
        RNN_x=np.zeros((n_RNN_sample,n_memory_step, input_feature))
        RNN_y=np.zeros((n_RNN_sample,n_forcast_step, output_feature))

        for n in range(n_RNN_sample):
            RNN_x[n,:,:]=x_timeseries[n:n+n_memory_step,:]
            RNN_y[n,:,:]=y_timeseries[n+n_memory_step:n+n_memory_step+n_forcast_step,:]
        if split != None:
            assert (split <=0.9) & (split >= 0.1), 'split not in reasonable range'
            return RNN_x[:int(split*len(RNN_x))], RNN_y[:int(split*len(RNN_x))], \
                   RNN_x[int(split*len(RNN_x))+1:], RNN_y[int(split*len(RNN_x))+1:]
        else:
            return RNN_x, RNN_y, None, None

    #########################################
    def build_graph(self):
        """
        build model graph


        :param neurons: number of hidden lstm cells
        :return: model object

        """
        input_shape=self.x_train.shape[1:]
        output_shape=self.y_train.shape[1:]
        encoder = LSTM(units = self.neurons, return_state = True)
        decoder = LSTM(units = self.neurons, return_sequences=True, return_state = True)
        decoder_dense = Dense(output_shape[-1])

        # data flow
        encoder_input = Input(input_shape)
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_state = [state_h, state_c]

        # initial input and state for iteration
        iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
        iter_state = encoder_state
        all_output = []

        #use prdiction as next input + use encoder output as decoder input
        for _ in range(output_shape[0]):
            # Run the decoder on one timestep, output == state_h since only one time step
            output, state_h, state_c = decoder(iter_input, initial_state=iter_state)
            output = decoder_dense(output)

            # Store the current prediction (we will concatenate all predictions later)
            all_output.append(output)

            # Reinject the outputs and state for the next loop iteration
            iter_input = output
            iter_state = [state_h, state_c]

        decoder_output = Lambda(lambda x: K.concatenate(x, axis=1))(all_output)
        model = Model(encoder_input, decoder_output)

        return model

    # build lstm architecture,
    def train(self, lr, batch_size, num_epochs):
        """
        construct stateful lstm model
        :param exp: experiment number
        :param neurons: number of neurons in lstm encoder and ecoder
        :param lr: learning rate
        :param batch_size:
        :param num_epochs:
        :return:
        """

        # save parameters to reconstruct other model
        self.file.write(
            'neurons: {} \nlr: {} \nbatch size: {} \nepochs: {}\n'.format(self.neurons, lr, batch_size, num_epochs))

        # Define model
        self.model=self.build_graph()
        print("trained model summary")
        #self.model.summary()

        #set up training
        decay_rate=lr/num_epochs

        adam = Adam(lr=lr,decay=decay_rate ,clipnorm=3.0)

        # compile model
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

        #self.model.save(self.exp_dir + 'model.h5', overwrite=True)

        # save parameters needed for predicting
        checkpoint= ModelCheckpoint(self.exp_dir+'weights', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        logger=PeriodicLogger()

        # fit model
        # give it reshaped timeseries data
        history=self.model.fit(self.x_train, self.y_train, validation_split=0.2, epochs=num_epochs, batch_size=batch_size,
                                 verbose=0, shuffle=False ,callbacks=[checkpoint ,logger])


        loss_hist=history.history['loss']

        # plot history
        plt.plot(loss_hist)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper left')
        plt.savefig(self.exp_dir + 'trainingloss.png')
        plt.show()

        #just to test reloading models
        # print("before loading model")
        # self.model.load_weights(self.exp_dir+'weights')
        #
        # p = self.model.predict(self.x_train)
        #
        # rmse =[]
        # for seq_index in range(10):
        #     # Take one sequence (part of the training set)
        #     # for trying out decoding.
        #     predicted=self.scaler.inverse_transform(p[seq_index])
        #     true= self.scaler.inverse_transform(self.y_train[seq_index])
        #
        #     s= np.sqrt(mean_squared_error(predicted,true))
        #     rmse+=[s]
        #
        #
        #     print("rmse : ", s)
        #     plt.plot(true , 'o-',label="true")
        #     plt.plot(predicted, 'o-', label='predicted')
        #     plt.legend()
        #     plt.show()
        #
        # print("avg rmse across sequences ", np.mean(np.array(rmse)))


    # build identical model and predict sequence one step at a time
    def predict(self ,n,x, y=None):

        """

        :param x: input
        :param n: number of predictions to show
        :param y: true output
        :return:
        """
        # print("recreated model summary")
        # model.summary()
        #model = load_model(self.exp_dir + "model.h5")
        model = self.build_graph()
        model.load_weights(self.exp_dir + 'weights')

        print("model and weights loaded successfully")

        ##########################################################################
        # predict validation:

        p = model.predict(x)

        rmse =[]
        for seq_index in range(n):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            predicted=self.scaler.inverse_transform(p[seq_index])
            true= self.scaler.inverse_transform(y[seq_index])

            s= np.sqrt(mean_squared_error(predicted,true))
            rmse+=[s]

            print(rmse)


            print("rmse : ", s)
            plt.plot(true , 'o-',label="true")
            plt.plot(predicted, 'o-', label='predicted')
            plt.legend()
            plt.show()

        print("avg rmse across sequences ", np.mean(np.array(rmse)))

        return p
        ###############################################


