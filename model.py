import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


GAMMA = 1
LEARNING_RATE = 0.01

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
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

    def word_embedding_layer(self, words):
        embeddings = self.elmo(
            words,
            as_dict=True)["word_emb"]
        return embeddings


class Sent_elmo_Embedding():
    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    
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


            #words = tf.reshape(self.embedded_words, shape = [-1, 512])

            self.words_lat = self.word_fully_connected(self.embedded_words, latent_size=128)
            self.sent_lat = self.sent_fully_connected(self.embedded_sent, latent_size=128)
            
            #self.words_lat = tf.reshape(self.words_lat, shape = [-1, 5, 128])

            self.relevance = self.relevance_layer(self.sent_lat, self.words_lat)
            self.output = self.soft_max_layer(self.relevance)
            self.loss = self.loss_func(self.output)

            self.descent = (self.train_op(self.loss), self.loss)
            self.accuracy = self.acc(self.output)

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
            s = sess.run(self.output, feed_dict={self.sents: sents, self.words:words})
            l = sess.run(self.loss, feed_dict={self.sents:sents, self.words:words})
            print(s.shape)
            print(s)
            print(l.shape)
            print(l)
            a = sess.run(self.accuracy, feed_dict={self.sents:sents, self.words:words})
            print(a.shape)
            print(a)
            #sess.run(self.print_op_cos, feed_dict={self.sent_embedding_class.sents: sents, self.word_embedding_class.word_ids: word_ids})

    def word_fully_connected(self, input_embedding, latent_size=128): 
         
        layer1 = tf.layers.dense(inputs=input_embedding, units=512, activation=tf.nn.tanh, name="word_fc1") 
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.tanh, name="word_fc2") 
 
        return tf.layers.dense(inputs=layer2, units=128, activation=tf.nn.tanh, name="word_fc3") 
 
    
    def sent_fully_connected(self, input_embedding, latent_size=128):
        
        layer1 = tf.layers.dense(inputs=input_embedding, units=512, activation=tf.nn.tanh, name="sent_fc1")
        layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.tanh, name="sent_fc2")
        
        return tf.layers.dense(inputs=layer2, units=latent_size, activation=tf.nn.tanh, name="sent_fc3")

    def relevance_layer(self, sent_lat, words_lat):
        
        norm_sent = tf.nn.l2_normalize(tf.expand_dims(sent_lat, axis=1), axis=2)
        norm_word = tf.nn.l2_normalize(tf.transpose(words_lat, perm=[0, 2, 1]), axis=1)

        cosine = tf.squeeze(tf.linalg.matmul(norm_sent, norm_word), axis=1)
        return cosine

    """def cosine_layer(self, sent_lat, words_lat):
        a = sent_lat
        b = words_lat
        c = tf.sqrt(tf.reduce_sum(tf.multiply(a,a),axis=1))
        d = tf.sqrt(tf.reduce_sum(tf.multiply(b,b),axis=1))
        e = tf.reduce_sum(tf.multiply(a,b),axis=1)
        f = tf.multiply(c,d)
        r = tf.divide(e,f)
        return r
    """

    def soft_max_layer(self, cosine):
        output = tf.nn.softmax(GAMMA*cosine)
        return output

    def loss_func(self, output):
        loss = -tf.gather(tf.math.reduce_sum(tf.math.log(output), axis=0), 0)
        return loss
    
    def train_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        descent = optimizer.minimize(loss)
        return descent

    def acc(self, output):
        argmax = tf.math.argmax(output, axis=1)
        non_zero = tf.math.count_nonzero(argmax)
        acc = 1 - non_zero/tf.size(argmax, out_type=tf.int64)
        return acc


