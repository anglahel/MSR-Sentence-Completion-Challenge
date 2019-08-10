import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import model


class Trainer():
	
	def __init__(self,queries,epochs,batch_size,model,print_ac = True):
		self.epoch_count = epochs
		self.queries = queries
		self.batch_size = batch_size
		self.model=model
		self.print_ac = print_ac
		with model.graph.as_default():
			self.sess = tf.Session()
			init = tf.global_variables_initializer()
			self.sess.run(init)



	def train(self):
		for epoch in range(self.epoch_count):
			self.train_epoch()
			if(self.print_ac):
				print("Accuracy in epoch " + str(epoch + 1) + ": " + str(self.acc()))


	def train_epoch(self):
		#np.random.shuffle(self.queries)
		n = len(self.queries)
		n=2000
		with self.model.graph.as_default():
			ind = 0
			while(ind<n):

				j = ind + self.batch_size
				if(j>n):
					j=n

				sentences = self.queries[ind:j,0]
				words = self.queries[ind:j,1]
				#print(sentences)
				#print(words)
				self.sess.run(self.model.descent,feed_dict = {self.model.words : words , self.model.sents : sentences})
				ind = ind +self.batch_size
				print("Trained "+str(ind)+" samples\n")



	def acc(self):
		n = len(self.queries)
		ind = 0
		m = 0
		ac = 0
		d = 0
		n=2000
		with self.model.graph.as_default():
			while(ind<n):
				j = ind + self.batch_size
				if(j>n):
					j=n

				d = j-ind

				sentences = self.queries[ind:j,0]
				print(sentences)
				#sentences = ["On this occasion, wounded pride exasperated her wrath still _____."]
				words = self.queries[ind:j,1]
				for i in range(words.shape[0]):
					word = words[i]
					answ = word.split()
					#np.random.shuffle(answ)
					words[i] = answ[1] + " " + answ[0] + " " + answ[2] + " " + answ[3] + " " + answ[4]
				print(words)
				#words = ["cecidere further unfruitfully obstructively incompetently"]
				#self.model.inference(words, sentences)
				t = self.sess.run(self.model.output,feed_dict = {self.model.words : words , self.model.sents : sentences})

				#ac = (m * ac + t*d) / (m + d)
				m += d

				ind = ind +self.batch_size
				print("Validated "+str(ind)+" samples acc: " + str(t))

		return ac
