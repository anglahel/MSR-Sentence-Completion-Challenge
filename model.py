import tensorflow as tf
import numpy as np

 
def fully_connected(input_emb, latent_size = 128): 
        self.graph = tf.Graph() 
        batch_size = None 
        layer_size = 300 
        with self.graph.as_default(): 
 
            #self.word_input = tf.placeholder(tf.int32, shape=(None,embedding_size)) 
                 
            self.layer1 = tf.layers.dense(inputs=input_emb, units=layer_size, activation = tf.nn.tanh) 
            self.layer2 = tf.layers.dense(inputs=self.layer1, units=layer_size, activation = tf.nn.tanh) 
 
            self.ret = tf.layers.dense(inputs=self.layer2, units=latent_size, activation=tf.nn.tanh) 
 
        return self.ret 
 


class Word_FC():
    def __init__(self, dict_size, embedding_dim):
        self.graph = tf.Graph()
        batch_size = None
        with self.graph.as_default():
            
            self.word_ids = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
            
            W = tf.Variable(tf.constant(0.0, shape=[dict_size, embedding_dim]), trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[dict_size, embedding_dim])
            self.assign_op = W.assign(self.embedding_placeholder)
            embedded_words = tf.nn.embedding_lookup(W, self.word_ids)
            self.print_op = tf.Print(embedded_words, [embedded_words])



