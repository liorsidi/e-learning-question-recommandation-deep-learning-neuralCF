import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import Constants as C


class CFModel(Sequential):
    def __init__(self, n_users, m_items,activation = 'tanh', factors=20, deep=True, layers=2, p_dropout=0.5, **kwargs):
        """
        a NCF keras implementation that support generic architecture design

        """
        P = Sequential()
        Q = Sequential()
        if factors == 0:  # relative
            user_factors = int(m_items / 10)
            items_factors = int(m_items / 10)
        else:
            user_factors = factors
            items_factors = factors
        if deep:
            P.add(Embedding(n_users, user_factors, input_length=1))
            P.add(Reshape((user_factors,)))
            Q.add(Embedding(m_items, items_factors, input_length=1))
            Q.add(Reshape((items_factors,)))
            super(CFModel, self).__init__(**kwargs)
            if layers == 0:  # regular SVD
                self.add(Merge([P, Q], mode='dot', dot_axes=1))
            else:  # deeper framework
                self.add(Merge([P, Q], mode='concat'))
                for l in range(layers):
                    self.add(Dropout(p_dropout))
                    self.add(Dense(user_factors, activation=activation))
                self.add(Dropout(p_dropout))
                self.add(Dense(1))#, activation='linear'))
        else:  # wide
            P.add(Dense(n_users,input_dim =1))
            Q.add(Dense(m_items,input_dim=1))
            super(CFModel, self).__init__(**kwargs)
            self.add(Merge([P, Q], mode='concat', dot_axes=1))
            self.add(Dense(1, activation=activation))
        self.compile(loss='mean_squared_error', optimizer='adadelta')

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]


class NCF(object):
    def __init__(self, s_id, Q_i, Q, deep, layers, p_dropout, factors,activation='tanh', epochs=10, batch_size=1024,callback=None,shuffle=False):
        """
        a NCF implementation for question sequencing
        :param s_id: student id
        :param S: all students answers
        :param questionaire: questionaires
        """
        self.s_id = s_id
        self.Q_i = Q_i
        self.Q = Q
        self.model = CFModel(1, 1)
        self.factors = factors
        self.deep = deep
        self.layers = layers
        self.p_dropout = p_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.callback = callback
        self.shuffle = shuffle
        self.params = dict(deep=self.deep,
                      layers=self.layers,
                      p_dropout=self.p_dropout,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                           activation=self.activation,
                        factors = self.factors,
                           shuffle=self.shuffle,
                           callback = self.callback)

    def rate(self):
        user = list([self.s_id] * len(self.Q_i))
        items = self.Q_i
        user = self.le_users.transform(user)
        items = self.le_items.transform(items)

        Q_ranked = pd.DataFrame(columns=[C.difficulty_col, C.student_col, C.questions_col])
        preds = self.model.predict([user, items]).flatten()

        Q_ranked[C.student_col] = [self.s_id] * len(self.Q_i)
        Q_ranked[C.questions_col] = self.Q_i
        Q_ranked[C.difficulty_col] = list(preds)
        return Q_ranked

    def update_model(self, S):
        users = self.le_users.transform(list(S[C.student_col]))
        items = self.le_items.transform(list(S[C.questions_col]))

        y = S[C.difficulty_col]
        self.model.train_on_batch([users, items], y)
        new_Q = S[C.questions_col].unique()

        self.Q_i = [x for x in self.Q_i if x not in new_Q]

    def fit(self, S):
        self.n_users = len(S[C.student_col].unique())
        self.m_items = len(set(list(S[C.questions_col].unique()) + self.Q))

        self.le_users = LabelEncoder()
        self.le_items = LabelEncoder()

        self.le_users.fit(list(set(S[C.student_col])))
        self.le_items.fit(list(set(list(S[C.questions_col]) + self.Q)))

        users = self.le_users.transform(list(S[C.student_col]))
        items = self.le_items.transform(list(S[C.questions_col]))

        y = S[C.difficulty_col]
        self.model = CFModel(self.n_users, self.m_items,self.activation, self.factors, self.deep, self.layers, self.p_dropout)
        if self.callback is None:
            self.model.fit([users, items], y, batch_size=self.batch_size, epochs=self.epochs,shuffle=self.shuffle)
        else:
            users_train, users_valid, items_train, items_valid, y_train, y_valid = train_test_split(users, items, y, test_size=0.2)
            self.model.fit([users_train,items_train], y_train, batch_size=self.batch_size, epochs=self.epochs, callbacks=self.callback,
                           validation_data=([users_valid,items_valid], y_valid),shuffle=self.shuffle)
