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
        self.embedded_words = tf.nn.embedding_lookup(self.W, self.word_ids, name="word_embedding")
        self.print_op = tf.Print(self.embedded_words, [self.embedded_words])
        return self.embedded_words


class Sent_Embedding():
    def __init__(self, dict_size, embedding_dim):
        self.graph = tf.Graph()
        self.batch_size = None
            
        self.W = tf.Variable(tf.constant(0.0, shape=[dict_size, embedding_dim]), trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[dict_size, embedding_dim])
        self.assign_op = self.W.assign(self.embedding_placeholder)
    
    def sent_embedding_layer(self):
        self.sents = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.embedded_sents = tf.nn.embedding_lookup(self.W, self.sents, name="sent_embedding")
        self.print_op = tf.Print(self.embedded_sents, [self.embedded_sents])
        return self.embedded_sents




class Model():
    def __init__(self, embedding, dict_size, embedding_dim):
        self.batch_size = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.word_embedding_class = Word_Embedding(dict_size, embedding_dim)
            embedded_words = self.word_embedding_class.word_embedding_layer()

            self.sent_embedding_class = Sent_Embedding(dict_size, embedding_dim)
            embedded_sent = self.sent_embedding_class.sent_embedding_layer()

            self.word_fully_connected(embedded_words, latent_size=128)
            self.sent_fully_connected(embedded_sent, latent_size=128)

            self.relevance_layer()
            sess = tf.Session()
            #init = tf.global_variables_initializer()
            #sess.run(init)
            sess.run(self.word_embedding_class.assign_op, feed_dict={self.word_embedding_class.embedding_placeholder: embedding})
            sess.run(self.sent_embedding_class.assign_op, feed_dict={self.sent_embedding_class.embedding_placeholder: embedding})
            #sess.run(word_embedding_class.print_op, feed_dict={word_embedding_class.word_ids: [[1]]})
            #sess.run(self.print_op, feed_dict={word_embedding_class.word_ids: [[1,2]]})
        

    def inference(self, word_ids, sents):
        with self.graph.as_default():
           sess = tf.Session()
           init = tf.global_variables_initializer()
           sess.run(init)
           sess.run(self.print_op_word, feed_dict={self.word_embedding_class.word_ids: word_ids})
           sess.run(self.print_op_sent, feed_dict={self.sent_embedding_class.sents: sents})
           sess.run(self.print_op_cos, feed_dict={self.sent_embedding_class.sents: sents, self.word_embedding_class.word_ids: word_ids})

    def word_fully_connected(self, input_embedding, latent_size=128): 
         
        layer1 = tf.layers.dense(inputs=input_embedding, units=512, activation=tf.nn.tanh, name="word_fc1") 
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.tanh, name="word_fc2") 
 
        self.word_lat = tf.layers.dense(inputs=layer2, units=128, activation=tf.nn.tanh, name="word_fc3") 
        self.print_op_word = tf.Print(tf.shape(self.word_lat), [tf.shape(self.word_lat)]) 
 
    
    def sent_fully_connected(self, input_embedding, latent_size=128):
        
        layer1 = tf.layers.dense(inputs=input_embedding, units=512, activation=tf.nn.tanh, name="sent_fc1")
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.tanh, name="sent_fc2")
        
        self.sent_lat = tf.layers.dense(inputs=layer2, units=latent_size, activation=tf.nn.tanh, name="sent_fc3")
        self.print_op_sent = tf.Print(tf.shape(self.sent_lat), [tf.shape(self.sent_lat)])

    def relevance_layer(self):

        self.cosine = tf.losses.cosine_distance(labels=tf.nn.l2_normalize(self.sent_lat, 0), predictions=tf.nn.l2_normalize(self.word_lat, 0), axis=0)
        self.print_op_cos = tf.Print(tf.shape(self.cosine), [tf.shape(self.cosine)])




