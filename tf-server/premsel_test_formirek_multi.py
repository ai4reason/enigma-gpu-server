import gc
from xopen import xopen
import _pickle as cPickle
import pickle as pkl

import tensorflow as tf
import random as rnd
import numpy as np
#from tensorflow.contrib.layers import fully_connected
from tf_slim.layers import fully_connected
from stop_watch import StopWatch, print_times
from tf_helpers import *
#import fcoplib as cop
import os

import debug_node
from debug_node import tf_debug
from parallel_environment import GameState
from graph_placeholder import GraphPlaceholder
from graph_conv import graph_start, graph_conv
from segments import Segments, SegmentsPH
from graph_data import GraphData

tf.compat.v1.disable_eager_execution()


class NetworkConfig:
    def __init__(self):
        self.threads = 4
        self.start_shape = (4,1,4)
        self.next_shape = (32,64,32)
        #self.next_shape = (11,12,13)
        #self.res_blocks = 3
        #self.layers = 3
        self.res_blocks = 1
        self.layers = 8
        self.hidden = 128
        self.balance_loss = True

    def __str__(self):
        return "start {}, next {}, last hidden {}, {} x {} layers, bal_loss {}".format(
            self.start_shape, self.next_shape, self.hidden,
            self.layers, self.res_blocks, self.balance_loss,
        )

class Network:
    def __init__(self, config = None):
        if config is None: config = NetworkConfig()
        print(config)
        self.config = config

        graph = tf.Graph()
        graph.seed = 43

        self.session = tf.compat.v1.Session(graph = graph,
                                  config=tf.compat.v1.ConfigProto(inter_op_parallelism_threads=config.threads,
                                                        intra_op_parallelism_threads=config.threads))

        with self.session.graph.as_default():

            self.structure = GraphPlaceholder()

            x = graph_start(self.structure, config.start_shape)
            last_x = None
            for _ in range(config.res_blocks):
                for n in range(config.layers):
                    x = graph_conv(x, self.structure,
                                   output_dims = config.next_shape, use_layer_norm = False)
                    #x = tuple(map(layer_norm, x))
                if last_x is not None:
                    x = [cx+lx for cx, lx in zip(x, last_x)]
                last_x = x

            nodes, symbols, clauses = x

            prob_segments = SegmentsPH(nonzero = True)
            self.prob_segments = prob_segments

            theorems = Segments(prob_segments.data).collapse(clauses)
            conjectures = prob_segments.gather(theorems, 0)
            mask = 1 - tf.scatter_nd(
                tf.expand_dims(prob_segments.start_indices_nonzero, 1),
                tf.ones(tf.reshape(prob_segments.nonzero_num, [1]), dtype=tf.int32),
                [prob_segments.data_len],
            )
            prem_segments, premises = prob_segments.mask_data(
                theorems, mask,
                nonzero = True
            )
            self.prem_segments = prem_segments

            network_outputs = tf.concat(
                [premises, prem_segments.fill(conjectures)],
                axis = 1
            )
            hidden = fully_connected(network_outputs, config.hidden)
            premsel_logits = tf_linear_sq(hidden)

            premsel_labels = tf.compat.v1.placeholder(tf.int32, [None])
            self.premsel_labels = premsel_labels

            pos_mask = tf.cast(premsel_labels, tf.bool)
            neg_mask = tf.logical_not(pos_mask)

            premsel_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.cast(premsel_labels, tf.float32),
                logits = premsel_logits,
            )
            if config.balance_loss:
                loss_on_true = tf.boolean_mask(tensor=premsel_loss, mask=pos_mask)
                loss_on_false = tf.boolean_mask(tensor=premsel_loss, mask=neg_mask)
                self.premsel_loss = (mean_or_zero(loss_on_true) + mean_or_zero(loss_on_false))/2
            else: self.premsel_loss = tf.reduce_mean(input_tensor=premsel_loss)

            optimizer = tf.compat.v1.train.AdamOptimizer()
            self.training = optimizer.minimize(self.premsel_loss)

            premsel_predictions = tf.cast(tf.greater(premsel_logits, 0), tf.int32)
            self.premsel_predictions = premsel_predictions
            self.premsel_logits = premsel_logits
            self.premsel_num = tf.size(input=premsel_predictions)
            #self.premsel_accuracy = tf.reduce_mean(
            #    tf.cast(
            #        tf.equal(premsel_labels, premsel_predictions),
            #        tf.float32,
            #    )
            #)
            predictions_f = tf.cast(premsel_predictions, tf.float32)
            predictions_on_true = tf.boolean_mask(tensor=predictions_f, mask=pos_mask)
            predictions_on_false = tf.boolean_mask(tensor=predictions_f, mask=tf.logical_not(pos_mask))
            self.premsel_tpr = mean_or_zero(predictions_on_true)
            self.premsel_tnr = mean_or_zero(1-predictions_on_false)

            self.session.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

        self.session.graph.finalize()

    def feed(self, data, use_labels, non_destructive = True):
        graph_data, lens_labels = zip(*data)
        d = self.structure.feed(graph_data, non_destructive)
        prob_lens, labels = zip(*lens_labels)
        self.prob_segments.feed(d, prob_lens)
        if use_labels:
            d[self.premsel_labels] = np.concatenate(labels)
        return d

    def predict(self, data):
        with StopWatch("data preparation"):
            d = self.feed(data, use_labels = False)
        with StopWatch("network"):
            logits, lens = self.session.run((self.premsel_logits, self.prem_segments.lens), d)
            def slice_logits():
                i = 0
                for l in lens:
                    yield logits[i:i+l]
                    i += l
                assert i == len(logits)
            return list(slice_logits())

    def get_loss(self, data):

        with StopWatch("data preparation"):
            d = self.feed(data, use_labels = True)
        with StopWatch("network"):
            return self.session.run(
                ((self.premsel_loss, self.premsel_tpr, self.premsel_tnr),
                 self.premsel_num), d)

    def train(self, data):

        with StopWatch("data preparation"):
            d = self.feed(data, use_labels = True)
        with StopWatch("network"):
            return self.session.run(
                (self.training,
                 (self.premsel_loss, self.premsel_tpr, self.premsel_tnr),
                 self.premsel_num), d)[1:]

    def debug(self, data, labels = None):

        d = self.feed(data, use_labels = True)
        debug_node.tf_debug_print(self.session.run(
            debug_node.debug_nodes, d
        ))

    def save(self, path, step = None):
        self.saver.save(self.session, path, global_step = step, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)

def file_lines(fname):
    with open(fname) as f:
        return [ line.strip() for line in f ]

def load_data(datadir):
    fnames = os.listdir(datadir)
    rnd.shuffle(fnames)

    data_list = []
    for fname in fnames:
        data, (lens, labels, symbols) = cop.load_premsel(os.path.join(datadir, fname))
        data_list.append((GraphData(data), (lens, labels)))

    return data_list, fnames

if __name__ == "__main__":
    import traceback_utils
    import sys

    # hide entrails of Tensorflow in error messages
    sys.excepthook = traceback_utils.shadow('/home/mirek/.local/')


    with StopWatch("network construction"):

        network = Network()
        #network.load("weights/premsel_bartosz_bal_29")
        #network.debug(test_data)

#    epochs = 100 * 25 # 25 = 124 / 5
    epochs = 1
    premsel_accum = [1.0, 0.0, 0.0]
    def update_accum(accum, current):
        for i,(acc,cur) in enumerate(zip(accum, current)):
            accum[i] = np.interp(0.1, [0, 1], [acc, cur])
    def stats_str(stats):
        if len(stats) == 2:
            return "loss {:.4f}, acc {:.4f}".format(*stats)
        else: return "loss {:.4f}, acc {:.4f} ({:.4f} / {:.4f})".format(stats[0], (stats[1]+stats[2])/2, stats[1], stats[2])

    #batch_size = 100
    tdata, tfnames = load_data("enigma-2019-10/mzr02-T10/data")

    batch_size = 10

    network.load("weights_tst1/premsel_enigma_01_2020_T30_loop02_{}".format(epochs))

    with StopWatch("evaluation"):

        for i in range(0, len(tdata), batch_size):

            batch = tdata[i : i+batch_size]
            batch_logits = network.predict(batch)
            for logits, fname in zip(batch_logits, tfnames[i : i+batch_size]):
                with open('out/' + fname, 'w') as f:
                    for x in logits:
                        print(x, file=f)

    print_times()
