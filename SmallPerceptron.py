#!/usr/bin/env python


import numpy as np
import struct

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GaussianNoise, GaussianDropout, AlphaDropout
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from keras.utils import np_utils
from keras.losses import mean_squared_error, mean_absolute_error, categorical_crossentropy, binary_crossentropy, logcosh
from keras.regularizers import l1,l2
import time
import pandas as pd 

def get_all_variables():
    return ["EventId", "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt",
            "Label","Weight","KaggleSet", "KaggleWeight"]

def GetRelevantIndices(VariableNames = None):
    # Input: array of Strings of the variables, for which data should be omitted with values of -999
    # output: numpy array object with the indices get_variables() corresponding to the strings
    if VariableNames is None:
        print('Did not receive any input for GetRelevantIndices!')
        return -1
    returnlist = np.array([], dtype = int)
    for name in VariableNames:
        tmp = get_all_variables().index(name)
        returnlist = np.append(returnlist, int(tmp))
    return returnlist

def get_variables():
    return ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", 
            "DER_deltaeta_jet_jet","DER_mass_jet_jet",
            "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "PRI_tau_pt", "PRI_tau_eta",
            "PRI_lep_pt", "PRI_lep_eta",
            "PRI_jet_leading_pt","PRI_jet_leading_eta",
            "PRI_jet_subleading_eta","PRI_jet_subleading_pt"]

if __name__ == '__main__':
    # Load training data
    #signal = pd.read_csv("atlas-higgs-challenge-2014_signal.csv")#.values
    #background = pd.read_csv("atlas-higgs-challenge-2014_background.csv")#.values
    #print(signal["DER_mass_MMC"])
    #train = pd.concat([signal,background], ignore_index=True)
    #test  = pd.read_csv("atlas-higgs-challenge-2014_validation.csv")#.values
    #print(train["DER_mass_MMC"])
    #print(test)
    # Convert labels from integers to one-hot vectors
    #labels = train['Label']
    #labels = np_utils.to_categorical(labels, 2)

    trainindices = GetRelevantIndices(get_variables())
    #print(trainindices)
    #train_data = train[trainindices]
    input_len=len(trainindices)
    model = Sequential()
    # current highscore: all elu activation, first layer tanh, last layer softmax
    # with l*2, l*5, l*1, l/1.5, 2 layer sizes, AMS = 0.6645 def|red|N,G,D
    # new highscore: relu,softplus,relu,relu,softplus,relu,softmax
    # with l*2,l*5,l*2.5,l,l/1.5,l/2.5,2, AMS = 0.677 def|red|N,G,D
    model.add(Dense(int(input_len*2), kernel_initializer='glorot_normal', input_dim=input_len, activation='relu', kernel_regularizer=l2(0.001)))
    #model.add(GaussianNoise(0.25))
    model.add(Dense(int(input_len*5), kernel_initializer='glorot_normal', activation='softplus', kernel_regularizer=l2(0.001)))
    #model.add(BatchNormalization())
    model.add(Dense(int(input_len*2.5), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(0.005)))
    #model.add(Dropout(0.4))
    #model.add(GaussianDropout(0.3))
    model.add(Dense(int(input_len*1), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(0.005)))
    #model.add(BatchNormalization())
    model.add(Dense(int(input_len/1.5), kernel_initializer='glorot_normal', activation='softplus', kernel_regularizer=l2(0.005)))
    #model.add(GaussianNoise(0.2))
    model.add(Dense(int(input_len/2.5), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(0.001)))
    #model.add(Dense(int(input_len*2), kernel_initializer='glorot_normal', activation='elu', kernel_regularizer=l2(0.001)))
    #model.add(Dense(int(input_len*3), kernel_initializer='glorot_normal', activation='elu', kernel_regularizer=l2(0.001)))
    #model.add(Dropout(0.25))
    #model.add(Dense(int(input_len/1.5), kernel_initializer='glorot_normal', activation='tanh', kernel_regularizer=l2(0.005)))
    #model.add(GaussianNoise(0.3))
    #model.add(Dense(int(input_len/2.5), kernel_initializer='glorot_normal', activation='tanh', kernel_regularizer=l2(0.005)))
    model.add(Dense(2, kernel_initializer='glorot_normal', activation='softmax', kernel_regularizer=l2(0.005)))
    model.compile(
            loss=binary_crossentropy,
            optimizer=Adam(),#SGD(lr = 0.1, momentum = 0.9, decay = 0.01),#Adam()
            metrics=['accuracy'])
    model.save('optimizedModel.h5')
