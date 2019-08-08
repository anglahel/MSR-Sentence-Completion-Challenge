import tensorflow as tf
import numpy as np


class Word_Embedding():
    def __init__(self, dict_size, embedding_dim):
        self.graph = tf.Graph()
        self.batch_size = None
            
        self.W = tf.Variable(tf.constant(0.0, shape=[dict_size, embedding_dim]), trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[dict_size, embedding_dim])
        self.assign_op = self.W.assign(self.embedding_placeholder)
    
    def word_embedding_layer(self):
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        embedded_words = tf.nn.embedding_lookup(self.W, self.word_ids)
        self.print_op = tf.Print(embedded_words, [embedded_words])



class Model():
    def __init__(self, embedding, dict_size, embedding_dim):
        self.batch_size = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            word_embedding_class = Word_Embedding(dict_size, embedding_dim)
            word_embedding_class.word_embedding_layer()
            sess = tf.Session()
            sess.run(word_embedding_class.assign_op, feed_dict={word_embedding_class.embedding_placeholder: embedding})
            sess.run(word_embedding_class.print_op, feed_dict={word_embedding_class.word_ids: [[1]]})
    
    
    def fully_connected(self, input_embedding, latent_size=128): 
        
        layer_size = 300 
  
        layer1 = tf.layers.dense(inputs=input_embedding, units=layer_size, activation=tf.nn.tanh) 
        layer2 = tf.layers.dense(inputs=layer1, units=layer_size, activation=tf.nn.tanh) 
 
        ret = tf.layers.dense(inputs=layer2, units=latent_size, activation=tf.nn.tanh) 
 
        return ret 
 



