#!/usr/bin/env pythonHillary
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, Activation
from keras.optimizers import RMSprop
from keras.models import load_model

import rospy
from std_msgs.msg import String

name1 = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]
weights = sys.argv[4]
model_path = sys.argv[5]
name2 = sys.argv[6]
lstm = int(sys.argv[7])

# Reading the trump dataset into raw_text
class Trump:
    def __init__(self):
        self.cnt = 1
        self.message = ''
        self.seq_len = 50
        self.n_vocab = 65
        self.model = Sequential()
        self.char_to_int = dict()
        self.int_to_char = dict()
        self.seed = 'interviewed by @cnn. the crooked hillary clinton'
        self.msg = ''
        rospy.init_node(name1)
        self.pub = rospy.Publisher(name1 + '_tweet', String, queue_size=100)
        print("Publishing on ", name1 + '_tweet')
        self.sub = rospy.Subscriber(name2 + '_tweet', String, self.callback)
        print("Subscribing to ", name2 + '_tweet')


    def callback(self, data):
        self.seed = data.data
        #self.predict()
        self.predict()
        rospy.sleep(1)
        self.pub.publish(self.msg)
        print "Msg: ",self.cnt, "\n", self.msg
        print
        print
        self.cnt += 1

    def setup(self):
        print "\n\n------I am :", name1, "-------\n\n"
        self.load_dict()
        self.def_model()
        self.node()

    def node(self):
        flag = name1[0]

        if flag == 't':
            self.predict()
            print self.msg
            self.pub.publish(self.msg)

        rospy.spin()

    def load_dict(self):

        self.char_to_int = np.load('/home/tito/catkin_ws/src/lstm_twitter/scripts/' + file1).item()
        self.int_to_char = np.load('/home/tito/catkin_ws/src/lstm_twitter/scripts/' + file2).item()

    
    def def_model(self):

        # Setting up the model
        self.model = Sequential()
        self.model.add(LSTM(lstm, input_shape=(self.seq_len, self.n_vocab), return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(lstm))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_vocab))
        self.model.add(Activation('softmax'))
        rmsprop = RMSprop(lr=0.001, clipvalue=5, decay=0.)
        self.model.load_weights("/home/tito/catkin_ws/src/lstm_twitter/scripts/" + weights)
        self.model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    def predict(self):

        #model = self.def_model()
        #load_model('/home/tito/catkin_ws/src/lstm_twitter/scripts/' + model_path)
        #model.load_weights("/home/tito/catkin_ws/src/lstm_twitter/scripts/" + weights)
        
        def gen_index(preds, temp=1.0):
            # helper function to sample an index from a probability array
            preds += 0.0001
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
        while cnt<120 :
            x = np.zeros((1, self.seq_len, self.n_vocab))

            for t, ch in enumerate(self.seed):
                x[0, t, self.char_to_int[ch]] = 1

            pred = self.model.predict(x, verbose=0)[0]
            temperature = 0.4
            next_index = gen_index(pred, temperature)
            next_char = self.int_to_char[next_index]

            generated += next_char
            self.seed = self.seed[1:] + next_char

            cnt += 1
            if next_char in ('.', '?', '!') and cnt > 60:
                break

        self.msg = generated


if __name__ == '__main__':
    t_obj = Trump()
    try:
        t_obj.setup()
    except rospy.ROSInterruptException:
        pass
