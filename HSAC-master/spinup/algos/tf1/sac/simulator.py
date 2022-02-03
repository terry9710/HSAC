from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import pandas as pd

train_data = pd.read_csv("D:\\spinningup-master\\data_reshape.CSV", low_memory=False)
c_state_column = ['STATE' + str(i) for i in range(14)]
c_next_state_column = ['NEXT_STATE' + str(i) for i in range(14)]

reward_column = ['REWARD']
action_c_column = ['ACTION' + str(i) for i in range(5)]
action_d_column = ['ACTION5']

column_dict = {'c_state': train_data[c_state_column],
               'c_next_state': train_data[c_next_state_column],
               'reward': train_data[reward_column],
               'action_c': train_data[action_c_column],
               'action_d': train_data[action_d_column]}

c_obs_dim = len(c_state_column)
act_c_dim = len(action_c_column)
act_d_dim = len(train_data['ACTION5'].unique())

# repla_buffer
replay_buffer = ReplayBuffer(obs_dim=c_obs_dim, act_c_dim=act_c_dim, act_d_dim=act_d_dim, size=len(train_data))
replay_buffer.load_from_csv(column_dict)

data_X, data_y = replay_buffer.sample_data()
print(data_X.shape, data_y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_y, test_size=0.2)

train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(train_X.shape, Y_train.shape, test_X.shape, Y_test.shape)
print(train_X)
# #
model = Sequential()
########### lstm_model_2.h5
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(14))
model.compile(loss='mse', optimizer='adam')

history = model.fit(train_X, Y_train, epochs=200, batch_size=2000, validation_data=(test_X, Y_test), verbose=2)


model.save('lstm_model_2.h5')


