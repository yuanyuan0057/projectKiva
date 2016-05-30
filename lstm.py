
import os
import getpass
import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.models.rnn import rnn, rnn_cell
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel
import nltk
import pandas as pd
import unicodedata
import tables
import cPickle
from copy import deepcopy
import matplotlib.pyplot as plt


class DataSet(object):
    def __init__(self, data, labels, loan_amounts):

        self._data = data
        self._labels = labels
        self._loan_amounts = loan_amounts
        self._input_dim = 25
        self._output_dim = 1
        self._num_examples = len(data)
        self.normalize_data()
        


    def quantize_labels(self, incs):
        #quantize into num labels
        print incs
        for idx, label in enumerate(self._labels):
            i = 0
            while label > incs[i]:
                i += 1
            self._labels[idx] = i - 1
        self._num_classes = len(incs) - 1


    def normalize_data(self):
        self.loan_mean = np.mean(self._loan_amounts)
        self.loan_std = np.std(self._loan_amounts)
        self._loan_amounts -= self.loan_mean
        self._loan_amounts /= self.loan_std
        #plt.hist(self._labels, 80)
        #plt.show()


    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    def lookup(self, data, wv):
        embedded_data = []
        for num in data:
            embedded_data.append(wv[num])
        return np.array(embedded_data)

    def next_batch_rnn(self, batch_size, num_steps, stride = 1):
        """Return the next `batch_size` examples from this data set for training RNN."""
        output_data = np.zeros((batch_size, self._input_dim, num_steps))
        output_label = np.zeros((batch_size, num_steps))
        output_loan_amount = np.zeros((batch_size, num_steps))
        for i in range(batch_size):
            while True:
                selected_idx = np.random.randint(0, self._num_examples - 1)
                selected_datum = self._data[selected_idx]
                if len(selected_datum) - num_steps - 1 > 1:
                    break
                #print selected_datum
        
            rnd_start = []
        
            rnd_start.append(np.random.randint(0, len(selected_datum) - num_steps - 1))
            for start in rnd_start:
                for j in range(num_steps):
                    output_data[i, :, j] = selected_datum[start + j]
                    output_label[i, j] = self._labels[selected_idx]
                output_loan_amount[i, j] = self._loan_amounts[selected_idx]
        return output_data, output_label, output_loan_amount


    def fetch_rnn_test(self, num_tests):
        data, labels, loan_amounts = [], [], []
        for i in range(num_tests):
            selected_idx = np.random.randint(0, self._num_examples - 1)
            #selected_idx = i
            size = len(self._data[selected_idx])
            data.append(np.transpose(np.array(self._data[selected_idx])))
            labels.append(np.transpose(np.array([self._labels[selected_idx] for i in range(size)])))
            loan_amounts.append(np.expand_dims(np.array(self._loan_amounts[selected_idx]), axis = 0))
        return data, labels, loan_amounts



def get_quantization_bins(labels, bins = 16):
    sorted_labels = sorted(labels)
    incs = [sorted_labels[len(labels) * i / bins] for i in range(bins)] + [float('Inf')]
    print incs
    return incs



def get_train_test_data(data, labels, loan_amounts):
    assert len(data) == len(labels)
    incs = get_quantization_bins(labels, bins = 2)
    train_num = int(len(data) * 0.8)

    perm = np.arange(len(data))
    np.random.shuffle(perm)

    train_set = DataSet(map(lambda x: data[x], perm[:train_num]), map(lambda x: labels[x], perm[:train_num]), 
        map(lambda x: loan_amounts[x], perm[:train_num]))
    test_set = DataSet(map(lambda x: data[x], perm[train_num:]), map(lambda x: labels[x], perm[train_num:]),
        map(lambda x: loan_amounts[x], perm[train_num:]))

    train_set.quantize_labels(incs)
    test_set.quantize_labels(incs)

    return train_set, test_set


wv_dim = 25
def toNum(word):
    try:
        x = word_to_num[word]
    except:
        #print 'unkownd word: ', word
        x = word_to_num['<unknown>']
    return x

data, labels, loan_amounts = [], [], []
date_format = "%Y-%m-%dT%H:%M:%SZ"
time_format = ""
id_table = {}
num_files = 50
try:
    data = cPickle.load(open('data_'+str(num_files)+'.dat', 'rb'))
    labels = cPickle.load(open('labels_'+str(num_files)+'.dat', 'rb'))
    loan_amounts = cPickle.load(open('loan_amounts_'+str(num_files)+'.dat', 'rb'))
except:
    wv = []
    word_to_num = {}
    num_to_word = []
    with open('data/glove.twitter.27B.' + str(wv_dim) + 'd.txt', 'rb') as f:
        cnt = 0
        for line in f:
            splitted = line.split()
            word = splitted[0]
            vec = map(float, splitted[1:])
            wv.append(vec)
            num_to_word.append(word)
            word_to_num[word] = cnt
            cnt += 1
    print len(num_to_word)
    print len(word_to_num.keys())

    for i in range(1, 1 + num_files):
        funded_num, unfunded_num = 0, 0
        print 'loading batch ', i
        name = 'loans/loan' + str(i)
        frame = pd.read_pickle(name)
        for index, row in frame.iterrows():
            posted_date = row['posted_date']
            funded_date = row['funded_date']
            days_funded = -1
            try:
                id_table[row['id']]
            except:
                id_table[row['id']] = 1
                days_funded = -2
            funded = -1
            if days_funded == -2 and posted_date != None and row['description_language'] == 'en' and len(row['description_texts']) > 0:
                funded = 0
                if funded_date != None:
                    funded_num += 1
                    funded = 1
                    posted_date = datetime.strptime(posted_date, date_format)
                    funded_date = datetime.strptime(funded_date, date_format)
                    days_funded = float((funded_date - posted_date).seconds)
                elif row['status'] == 'expired' or 'inactive_expired':
                    days_funded = 31.0
                    unfunded_num += 1

            if days_funded > -1:
                loan_amount = row['loan_amount']
                loan_amounts.append(loan_amount)
                words = nltk.word_tokenize(row['description_texts'].lower())
                #f = lambda x: word_to_num[x]# if x in word_to_num.keys() else word_to_num['<unknown>']
                data.append(map(lambda i: wv[i], map(toNum, words)))
                labels.append(days_funded)
        print "funded loans: ", funded_num
        print "unfunded loans:", unfunded_num
    cPickle.dump(data, open('data_'+str(num_files)+'.dat', 'wb'))
    cPickle.dump(labels, open('labels_'+str(num_files)+'.dat', 'wb'))
    cPickle.dump(loan_amounts, open('loan_amounts_'+str(num_files)+'.dat', 'wb'))





train_set, test_set = get_train_test_data(data, labels, loan_amounts)
training_iters = 1e6
batch_size = 32
display_step = 1000
training_iters = 2e5 * batch_size


class Model(object):
    # Parameters

    ''' 
    Network Parameters:
    use regression to minimize distance with respect to the real fulfill datetime
    '''
    

    def __init__(self, test = False):
        
        self.is_test = test
        self.learning_rate = 1e-3
        self.num_steps = 10
        self.dropout = 0.8
        self.batch_size = 32
        self.n_classes = 2

    '''1 layer RNN'''

    def initialize_model(self):
        self.keep_prob = tf.placeholder(tf.float32)
        sigma = 1e-3
        


        #embeddings = tf.Variable(tf.convert_to_tensor(wv, dtype=tf.float32), name="Embedding")

        self.x = tf.placeholder(tf.float32, shape = (self.batch_size, wv_dim, self.num_steps))
        self.y = tf.placeholder(tf.int32, shape = (self.batch_size, self.num_steps))
        self.loan_amounts = tf.placeholder(tf.float32, shape = (self.batch_size, self.num_steps))

        if self.num_steps > 1:
            inputs = map(tf.squeeze, tf.split(2, self.num_steps, self.x))
            loans = tf.split(1, self.num_steps, self.loan_amounts)
        else:
            inputs = [self.x[:,:,0]]
            loans = [self.loan_amounts]

        filter_number_1 = 256
        filter_number_2 = 144

        cell1 = rnn_cell.BasicLSTMCell(filter_number_1, forget_bias=1.0, input_size = wv_dim)
        cell2 = rnn_cell.BasicLSTMCell(filter_number_2, forget_bias=1.0, input_size = filter_number_1)
        cell = rnn_cell.MultiRNNCell([cell1, cell2])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        state = self.initial_state
        self.loss = 0
        rnn_outputs = []

        for idx, batch in enumerate(inputs):
            with tf.variable_scope("RNN") as scope:
                if idx > 0:
                    scope.reuse_variables()
                wc3 = tf.get_variable("wc3", (filter_number_2 + 1, self.n_classes), 
                    initializer = tf.random_normal_initializer(mean=0.0, stddev=sigma, seed=None, dtype=tf.float32))
                bc3 = tf.get_variable("bc3", (self.n_classes,),
                    initializer = tf.random_normal_initializer(mean=0.0, stddev=sigma, seed=None, dtype=tf.float32))
                output, state = cell(batch, state)
                pred = bc3 + tf.matmul(tf.concat(1, [loans[idx], output]), wc3)
                #pred = tf.matmul(output, wc3) + bc3
                rnn_outputs.append(pred)
                self.previous_state = state

        self.output = tf.argmax(rnn_outputs[-1], 1)
        for i in range(len(inputs)):
            #print rnn_outputs[i].get_shape()
            self.loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(rnn_outputs[i], self.y[:,i]))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)



    def train(self, batch_xs, batch_ys, batch_loans, sess):
        #print batch_ys
        state1 = self.initial_state.eval()

        l, state1, _ = sess.run([self.loss, self.previous_state, self.optimizer], 
            feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: self.dropout, 
            self.initial_state: state1, self.loan_amounts: batch_loans})
        return l


    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x, y) in enumerate(
            ptb_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                  self.labels_placeholder: y,
                  self.initial_state: state,
                  self.dropout_placeholder: dp}
            loss, state, _ = session.run(
              [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                  step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')



    def test(self, num_steps, batch_xs, batch_ys, batch_loans, sess):
        state1 = self.initial_state.eval()
        cost = 0
        predictions = []
        final_pred = []
        for i in range(num_steps):
            #print i
            data_in = np.expand_dims(batch_xs[:,:,i], 2)
            label_in = np.expand_dims(batch_ys[:,i], 1)
            l, outputs, state1 = sess.run([self.loss, self.output, self.previous_state], feed_dict={self.x: data_in, self.y: label_in, 
                self.loan_amounts: batch_loans, self.initial_state: state1, self.keep_prob: self.dropout})
            predictions.append(outputs)
            cost += l
        #use majority voting
        #print np.array(predictions).shape
        final_pred = (np.argmax(np.bincount(np.array(predictions)[:,0])))
        return cost, final_pred



# Launch the graph
skip_training = False

with tf.variable_scope('RNNLM') as scope:
    model = Model()
    model.initialize_model()
    train_num_steps = model.num_steps
    scope.reuse_variables()

    test_model = Model(test = True)
    test_model.dropout = 1.0
    test_model.batch_size = 1
    test_model.num_steps = 1
    test_model.initialize_model()

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    
    batch_losses = []
    train_losses = []
    t0 = time.clock()
    with tf.Session() as sess:
        decrease_lr = 0
        sess.run(init)
        if not skip_training:
            step = 1
            train_cnt = 0
            train_loss = 0
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_xs, batch_ys, batch_loans = train_set.next_batch_rnn(batch_size, train_num_steps)
                #print str(step * batch_size) + " samples trained"
                # Fit training using batch data
                #print batch_xs.shape
                #print batch_ys.shape
                #sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                l = model.train(batch_xs, batch_ys, batch_loans, sess)
                train_cnt += batch_size * train_num_steps
                train_loss += l
                if (step*batch_size) / (5000 * batch_size) - decrease_lr == 1:
                    model.learning_rate /= 1.01
                    decrease_lr += 1
                if step % display_step == 0:
                    train_losses.append(train_loss / train_cnt)
                    train_cnt = 0
                    train_loss = 0
                    print "learning rate = ", model.learning_rate
                    acc = 0
                    y_square = 0
                    # Calculate batch accuracy
                    batch_xs, batch_ys, batch_loans = test_set.fetch_rnn_test(20)
                    acc = 0
                    y_square = 0
                    total_description_length = 0
                    predictions = []
                    for idx in range(len(batch_xs)):
                        num_steps = batch_xs[idx].shape[1]
                        total_description_length += num_steps
                        #print test_model.num_steps
                        #print batch_xs[idx].shape
                        #print batch_ys[idx].shape
                        loss, pred = test_model.test(num_steps, 
                            np.expand_dims(batch_xs[idx], axis = 0), np.expand_dims(batch_ys[idx], axis = 0), 
                            np.expand_dims(batch_loans[idx], axis = 1), sess)
                        acc += loss
                        predictions.append(pred)
                        #acc += sess.run(loss, feed_dict={x: np.reshape(batch_xs[idx], (1, 1, -1))
                        #    , y: np.reshape(batch_ys[idx], (1, 1, -1)), keep_prob: 1.})
                        y_square += np.sum(batch_ys[idx]**2)
                    #acc = np.sqrt(acc / y_square)
                    #print acc.shape
                    acc /= total_description_length
                    #print 'total description: ', total_description_length
                    batch_losses.append(acc)
                    # Calculate batch loss
                    #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    print "Iter " + str(step*batch_size/training_iters) + ", Train Loss= ", (train_losses[-1])
                    print "Iter " + str(step*batch_size/training_iters) + ", Test Loss= ", (batch_losses[-1])
                    print "Iter " + str(step*batch_size/training_iters) + ", True Labels vs Predicted Labels"
                    print [batch_ys[i][0] for i in range(len(batch_ys))]
                    print predictions
                    rate = 0
                    for i in range(len(predictions)):
                        if predictions[i] == batch_ys[i][0]:
                            rate += 1.0
                    rate /= len(predictions)
                    print "accurate: ", rate
                    #print predictions
                    print "\n"

                step += 1
            print "Optimization Finished!"
            saver.save(sess, './rnn_nnet.weights')

            print "time: ", time.clock() - t0
            moving_average = np.zeros((len(batch_losses),1))
            ma = min(10, len(batch_losses) / 10)
            window = np.mean(batch_losses[:2*ma+1])
            moving_average[ma] = window
            for i in range(ma + 1, len(batch_losses) - ma):
                window = ((2*ma+1) * window + batch_losses[i+ma] - batch_losses[i-ma-1]) / (2*ma+1)
                moving_average[i] = window
            for i in range(ma):
                moving_average[i] = batch_losses[i]
                moving_average[len(batch_losses)-i-1] = moving_average[len(batch_losses)-ma-1]
            plt.plot(train_losses)
            plt.plot(batch_losses)
            plt.plot(moving_average)

            plt.show()


        saved_model = 'rnn_nnet_lstm_no_loan_amount'
        saver.restore(sess, saved_model + '.weights')
        num_samples = 3000
        batch_xs, batch_ys, batch_loans = test_set.fetch_rnn_test(num_samples)
        rate = 0
        y_square = 0
        total_description_length = 0
        predictions = []
        print "begin testing on {0} test samples.".format(num_samples)
        for idx in range(len(batch_xs)):
            num_steps = batch_xs[idx].shape[1]
            total_description_length += num_steps
            sys.stdout.write("\r" + str(idx))
            sys.stdout.flush()
            cost, pred = test_model.test(num_steps, 
                            np.expand_dims(batch_xs[idx], axis = 0), np.expand_dims(batch_ys[idx], axis = 0), 
                            np.expand_dims(batch_loans[idx], axis = 1), sess)
            predictions.append(pred)
        for i in range(len(predictions)):
            if predictions[i] == batch_ys[i][0]:
                rate += 1.0
        rate /= len(predictions)
        print "\nfinal test accuracy: ", rate









