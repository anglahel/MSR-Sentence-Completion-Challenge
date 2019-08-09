import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class Word_Lookup_Embedding():
    def __init__(self, dict_size, embedding_dim):
        self.graph = tf.Graph()
        self.batch_size = None
            
        self.W = tf.Variable(tf.constant(0.0, shape=[dict_size, embedding_dim]), trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[dict_size, embedding_dim])
        self.assign_op = self.W.assign(self.embedding_placeholder)
    
    def word_embedding_layer(self):
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.embedded_words = tf.nn.embedding_lookup(self.W, self.word_ids, name="word_embedding")


class Word_elmo_Embedding():
    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    def word_embedding_layer(self, words):
        embeddings = self.elmo(
            words,
            as_dict=True)["elmo"]
        return embeddings


class Sent_elmo_Embedding():
    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    
    def sent_embedding_layer(self, sents):
        embeddings = self.elmo(
            sents,
            as_dict=True)["default"]
        return embeddings






class Model():
    def __init__(self):
        self.batch_size = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.word_embedding_class = Word_elmo_Embedding()
            self.words = tf.placeholder(dtype=tf.string)
            self.sents = tf.placeholder(dtype=tf.string)
            self.embedded_words = self.word_embedding_class.word_embedding_layer(self.words)

            self.sent_embedding_class = Sent_elmo_Embedding()
            self.embedded_sent = self.sent_embedding_class.sent_embedding_layer(self.sents)

            self.word_lat = self.word_fully_connected(self.embedded_words, latent_size=128)
            self.sent_lat = self.sent_fully_connected(self.embedded_sent, latent_size=128)

            #self.relevance_layer()
            #sess = tf.Session()
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #sess.run(self.word_embedding_class.assign_op, feed_dict={self.word_embedding_class.embedding_placeholder: embedding})
            #sess.run(self.sent_embedding_class.assign_op, feed_dict={self.sent_embedding_class.embedding_placeholder: embedding})
            #sess.run(word_embedding_class.print_op, feed_dict={word_embedding_class.word_ids: [[1]]})
            #sess.run(self.print_op, feed_dict={word_embedding_class.word_ids: [[1,2]]})
        
    def inference(self, words, sents):
        with self.graph.as_default():
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            w = sess.run(self.word_lat, feed_dict={self.words: words})
            s = sess.run(self.sent_lat, feed_dict={self.sents: sents})
            print(w.shape)
            print(s.shape)
            #sess.run(self.print_op_cos, feed_dict={self.sent_embedding_class.sents: sents, self.word_embedding_class.word_ids: word_ids})

    def word_fully_connected(self, input_embedding, latent_size=128): 
         
        layer1 = tf.layers.dense(inputs=input_embedding, units=512, activation=tf.nn.tanh, name="word_fc1") 
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.tanh, name="word_fc2") 
 
        return tf.layers.dense(inputs=layer2, units=128, activation=tf.nn.tanh, name="word_fc3") 
 
    
    def sent_fully_connected(self, input_embedding, latent_size=128):
        
        layer1 = tf.layers.dense(inputs=input_embedding, units=512, activation=tf.nn.tanh, name="sent_fc1")
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.tanh, name="sent_fc2")
        
        return tf.layers.dense(inputs=layer2, units=latent_size, activation=tf.nn.tanh, name="sent_fc3")

    def relevance_layer(self):

        self.cosine = tf.losses.cosine_distance(labels=tf.nn.l2_normalize(self.sent_lat, 0), predictions=tf.nn.l2_normalize(self.word_lat, 0), axis=0)




