#!/usr/bin/env python
# coding: utf-8

# In[32]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention


# In[50]:


def make_attentive_model(tx: int, num_channels: int, ty:int, 
                         num_lstm_hidden=128, attention_width=12) -> Model:
    inputs = Input(shape=(tx, num_channels))
    X = LSTM(
        units=num_lstm_hidden, 
        return_sequences=True, 
        bias_initializer='zeros',
    )(inputs)
    X = SeqSelfAttention(
        units=32,
        attention_width=attention_width,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        return_attention=False,
        history_only=False,
        kernel_initializer='glorot_normal',
        bias_initializer='zeros',
        kernel_regularizer=l2(1e-6),
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_additive_bias=True,
        use_attention_bias=True,
        attention_activation=None,
        attention_regularizer_weight=0.0,
    )(X)
    X = Flatten()(X)
    X = Dense(
        units=ty, 
        bias_initializer='zeros',
    )(X)
#     X = Activation('linear')(X)
    model = Model(inputs=inputs, outputs=X)
    return model


# In[51]:


# def build_structured_self_attention_embedder(word_window_size, num_features,
#                                              hidden_state_size, num_layers, attention_filters1, attention_filters2,
#                                              dropout, recurrent_dropout, regularization_lambda):
#     # Input for text sequence
#     sequence_input = Input(shape=(word_window_size, num_features, ), name="sequence_input_placeholder")

# #     # Word embeddings lookup for words in sequence
# #     sequence_word_embeddings = Embedding(input_dim=vocabulary_size + 1,
# #                                          output_dim=word_embedding_size,
# #                                          embeddings_initializer='glorot_uniform',
# #                                          embeddings_regularizer=l2(regularization_lambda),
# #                                          mask_zero=True,
# #                                          name="sequence_word_embeddings")(sequence_input)

#     # Obtain hidden state of Bidirectional LSTM at each word embedding
#     hidden_states = Dense(8, kernel_initializer='glorot_uniform', use_bias=True, bias_initializer='zeros')(sequence_input)
#     for layer in range(num_layers):
#         hidden_states = Bidirectional(LSTM(units=hidden_state_size,
#                                            dropout=dropout,
#                                            recurrent_dropout=recurrent_dropout,
#                                            kernel_initializer='glorot_uniform',
#                                            recurrent_initializer='glorot_uniform',
#                                            use_bias=True,
#                                            bias_initializer='zeros',
#                                            kernel_regularizer=l2(regularization_lambda),
#                                            recurrent_regularizer=l2(regularization_lambda),
#                                            bias_regularizer=l2(regularization_lambda),
#                                            activity_regularizer=l2(regularization_lambda),
#                                            implementation=1,
#                                            return_sequences=True,
#                                            return_state=False,
#                                            unroll=True),
#                                       merge_mode='concat', name="lstm_outputs_{}".format(layer))(hidden_states)

#     # Attention mechanism
#     attention = Conv1D(filters=attention_filters1, kernel_size=1, activation='tanh', padding='same', 
#                        use_bias=True,
#                        kernel_initializer='glorot_uniform', 
#                        bias_initializer='zeros',
#                        kernel_regularizer=l2(regularization_lambda),
#                        bias_regularizer=l2(regularization_lambda), activity_regularizer=l2(regularization_lambda),
#                        name="attention_layer1")(hidden_states)
#     attention = Conv1D(filters=attention_filters2, kernel_size=1, activation='linear', padding='same', 
#                        use_bias=True,
#                        kernel_initializer='glorot_uniform', 
#                        bias_initializer='zeros',
#                        kernel_regularizer=l2(regularization_lambda),
#                        bias_regularizer=l2(regularization_lambda), activity_regularizer=l2(regularization_lambda),
#                        name="attention_layer2")(attention)
#     attention = Lambda(lambda x: softmax(x, axis=1), name="attention_vector")(attention)

#     # Apply attention weights
#     weighted_sequence_embedding = Dot(axes=[1, 1], normalize=False, name="weighted_sequence_embedding")(
#         [attention, hidden_states])

#     # Add and normalize to obtain final sequence embedding
#     sequence_embedding = Flatten()(weighted_sequence_embedding)
#     sequence_embedding = Dense(1, kernel_initializer='normal', use_bias=False, activation='linear')(sequence_embedding)

#     # Build model
#     model = Model(inputs=sequence_input, outputs=sequence_embedding, name="sequence_embedder")
#     model.summary()
#     return model


# In[52]:


import numpy as np
import pandas as pd
from crypr.build import series_to_supervised, make_single_feature, calc_target
from crypr.util import get_project_path
from os.path import join

maximum_pct_change = 3.5
tx = 72
ty = 1


data = pd.read_csv(join(get_project_path(), 'data', 'raw', 'BTC.csv'), index_col=0)
data = calc_target(data, target='close')['target']
data = series_to_supervised(data, tx, ty, dropnan=True)
data = data[data.abs() < maximum_pct_change].dropna(how='any')
p(data.shape)
p(data.columns.values)
plt.plot(data.values)
plt.show()


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.15, shuffle=False)
p(X_train.shape)
p(X_train.columns.values)
plt.plot(X_train.values)
plt.show()


# In[23]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_train = MinMaxScaler(feature_range=(-1,1)).fit_transform(X_train)
X_test = MinMaxScaler(feature_range=(-1,1)).fit_transform(X_test)
plt.plot(X_train)
plt.show()


# In[24]:


# att = build_structured_self_attention_embedder(word_window_size=12, num_features=1,
#                                              hidden_state_size=64, num_layers=1, attention_filters1=96, 
#                                                attention_filters2=96,
#                                              dropout=0.2, recurrent_dropout=0.2, regularization_lambda=1e-4)


# In[53]:


att = make_attentive_model(tx=tx, num_channels=1, ty=ty, num_lstm_hidden=64, attention_width=16)
opt = Adam(lr=1e-4)

att.compile(optimizer=opt, loss='mse', metrics=['mae'])
att.summary()


# In[55]:


att.fit(
    np.expand_dims(X_train, axis=-1), 
    y_train, 
    epochs=40, 
    batch_size=128,
    shuffle=True,
    validation_data=(np.expand_dims(X_test, -1), y_test),
)


# In[56]:


y_pred = att.predict(np.expand_dims(X_test, axis=-1))


# In[57]:


plt.subplots(figsize=(12,7))
plt.plot(y_test.values, label='actual')
plt.plot(y_pred, label='predicted')
plt.legend()
plt.show()


# In[58]:


from sklearn.metrics import mean_absolute_error
test_mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
p(test_mae)


# In[ ]:




