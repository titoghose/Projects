#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, Activation
from keras.optimizers import RMSprop

import rospy
from std_msgs.msg import String


# Reading the hillary dataset into raw_text
class hillary:
    
    def __init__(self):
        self.message = ''
        self.seq_len = 50
        self.n_vocab = 64
        self.char_to_int = dict()
        self.int_to_char = dict()
        self.seed = 'interviewed by @cnn. the crooked hillary clinton'
        
        self.model = Sequential()

    def callback(self, data):
        self.seed = data.data
        msg = self.predict()
        print msg
        self.pub.publish(msg)

    def setup(self):
        self.load_dict()
        self.def_model()
        self.node()

    def node(self):
        rospy.init_node('hillary')
        rospy.Subscriber('trump_tweet', String, self.callback)
        self.pub = rospy.Publisher('hillary_tweet', String, queue_size=100)
        
        msg = self.predict()
        print msg
        self.pub.publish(msg)
        while not rospy.is_shutdown():
            rospy.spin()

    def load_dict(self):
        self.char_to_int = np.load('/home/tito/catkin_ws/src/lstm_twitter/scripts/hillary_char_to_int.npy').item()
        self.int_to_char = np.load('/home/tito/catkin_ws/src/lstm_twitter/scripts/hillary_int_to_char.npy').item()

    def def_model(self):

        ####### Setting up the model #######
        self.model.add(LSTM(256, input_shape=(self.seq_len, self.n_vocab), return_sequences=True, unroll=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_vocab))
        self.model.add(Activation('softmax'))
        rmsprop = RMSprop(lr=0.001, clipvalue=5, decay=0.)
        self.model.load_weights("/home/tito/catkin_ws/src/lstm_twitter/scripts/weights-improvement-2xLSTM256_b9000-00-1.3454.hdf5")
        self.model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self):

        def gen_index(preds, temp=1.0):
            # helper function to sample an index from a probability array
            preds = preds + 0.0001
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temp
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        self.seed = self.seed.lower()

        if len(self.seed) >= self.seq_len:
            start = len(self.seed) - self.seq_len + 1
            self.seed = self.seed[start:]
        else:
            space = self.seq_len - len(self.seed) - 1
            self.seed = ' ' * space + self.seed

        self.seed += '.'
        generated = ''

        cnt = 0
        while cnt<40:
            x = np.zeros((1, self.seq_len, self.n_vocab))

            for t, ch in enumerate(self.seed):
                x[0, t, self.char_to_int[ch]] = 1

            pred = self.model.predict(x, verbose=0)[0]
            temperature = 0.5
            next_index = gen_index(pred, temperature)
            next_char = self.int_to_char[next_index]

            generated += next_char
            self.seed = self.seed[1:] + next_char

            cnt += 1
            if next_char in ('.', '?', '!') and cnt > 40:
                break

        return generated

if __name__ == '__main__':
    t_obj = hillary()
    try:
        t_obj.setup()
    except rospy.ROSInterruptException:
        pass
