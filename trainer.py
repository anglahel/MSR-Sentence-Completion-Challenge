import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import model


class Trainer()
	
	def __init__(self,queries,epochs,batch_size,model,print_ac = True):
		self.epoch_count = epochs
		self.queries = queries
		self.batch_size = batch_size
		self.model=model
		self.print_ac = print_ac


	def train(self):
		for epoch in range(self.epoch_count):
			self.train_epoch()
			if(self.print_ac):
				print("Accuracy in epoch " + str(epoch + 1) + ": " + self.acc())


	def train_epoch(self):
		np.random.shuffle(self.queries)
		n = len(self.queries)
		with self.model.graph.as_default():
			ind = 0
			while(ind<n):

				j = ind + self.batch_size
				if(j>n):
					j=n

				self.sentences = self.queries[ind:j,0]
				self.words = self.queries[ind:j,0]

				self.session.run(self.model.descent,feed_dict = {self.model.words : self.words , self.model.sents : self.sentences})

				ind = ind +self.batch_size



	def acc(self):
		n = len(self.queries)
		ind = 0
		m = 0
		ac = 0
		d = 0
		with self.model.graph.as_default():
			while(ind<n):

				j = ind + self.batch_size
				d = j - ind
				if(j>n):
					j=n

				self.sentences = self.queries[ind:j,0]
				self.words = self.queries[ind:j,0]

				t = self.session.run(self.model.accuracy,feed_dict = {self.model.words : self.words , self.model.sents : self.sentences})

				ac = (m * ac + t) / (m + d)
				m += d

				ind = ind +self.batch_size

		return ac
