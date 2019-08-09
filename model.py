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
        self.embedded_words = tf.nn.embedding_lookup(self.W, self.word_ids)
        self.print_op = tf.Print(self.embedded_words, [self.embedded_words])
        return self.embedded_words



class Model():
    def __init__(self, embedding, dict_size, embedding_dim):
        self.batch_size = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.word_embedding_class = Word_Embedding(dict_size, embedding_dim)
            embedded_words = self.word_embedding_class.word_embedding_layer()
            self.word_fully_connected(embedded_words, latent_size=128)
            sess = tf.Session()
            #init = tf.global_variables_initializer()
            #sess.run(init)
            sess.run(self.word_embedding_class.assign_op, feed_dict={self.word_embedding_class.embedding_placeholder: embedding})
            #sess.run(word_embedding_class.print_op, feed_dict={word_embedding_class.word_ids: [[1]]})
            #sess.run(self.print_op, feed_dict={word_embedding_class.word_ids: [[1,2]]})
        

    def inference(self, word_ids):
        with self.graph.as_default():
           sess = tf.Session()
           init = tf.global_variables_initializer()
           sess.run(init)
           sess.run(self.print_op, feed_dict={self.word_embedding_class.word_ids: word_ids})


    
    def word_fully_connected(self, input_embedding, latent_size=128): 
        
        layer_size = 300 
  
        self.layer1 = tf.layers.dense(inputs=input_embedding, units=layer_size, activation=tf.nn.tanh) 
        self.layer2 = tf.layers.dense(inputs=self.layer1, units=layer_size, activation=tf.nn.tanh) 
 
        self.ret = tf.layers.dense(inputs=self.layer2, units=latent_size, activation=tf.nn.tanh) 
        self.print_op = tf.Print(tf.shape(self.ret), [tf.shape(self.ret)]) 
 



