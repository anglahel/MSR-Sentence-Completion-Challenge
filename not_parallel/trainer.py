import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import model
import os


class Trainer():
	
	def __init__(self,train_data,valid_data,test_data,epochs,batch_size,model,print_ac = True):
		self.epoch_count = epochs
		self.train_data = train_data
		self.valid_data = valid_data
		self.test_data = test_data
		self.batch_size = batch_size
		self.model=model
		self.print_ac = print_ac
		self.output  = "output"
		self.model.save_graph_summary(os.path.join(self.output, 'summary'))
		self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.output, 'train'))
		self.valid_summary_writer = tf.summary.FileWriter(os.path.join(self.output, 'valid'))
		self.fl = open("acc.txt","a+")
		with model.graph.as_default():
			self.sess = tf.Session()
			init = tf.global_variables_initializer()
			self.sess.run(init)



	def train(self):
		self.epoch = 0
		for self.epoch in range(self.epoch_count):
			self.train_epoch()
			if(self.print_ac):
				#self.fl.write("Accuracy in epoch " + str(self.epoch + 1) + ": " + str(self.acc()) + ".\n")
				self.test()


	def train_epoch(self):
		np.random.shuffle(self.train_data)
		n = len(self.train_data)
		with self.model.graph.as_default():
			ind = 0
			while(ind<n):

				j = ind + self.batch_size
				if(j>n):
					j=n

				sentences = self.train_data[ind:j,0]
				words = self.train_data[ind:j,1]
				#print(sentences)
				#print(words)
				(_, loss), summary = self.sess.run((self.model.descent,self.model.summary),feed_dict = {self.model.words : words , self.model.sents : sentences})
				ind = ind +self.batch_size
				self.train_summary_writer.add_summary(summary, self.epoch)
				print("Trained "+str(j)+" samples with loss " + str(loss) + ".\n")



	def acc(self):
		n = len(self.valid_data)
		ind = 0
		m = 0
		ac = 0
		d = 0
		with self.model.graph.as_default():
			while(ind<n):
				j = ind + self.batch_size
				if(j>n):
					j=n

				d = j-ind
				sentences = self.valid_data[ind:j,0]
				#sentencesprim = self.valid_data[ind:j,0]
				#print(sentences)
				#sentences = ["On this occasion, wounded pride exasperated her wrath still _____."]
				words = self.valid_data[ind:j,1]
				#wordsprim = self.valid_data[ind:j,1]
				#changed_words = []
				#perm = [0,1,2,3,4]
				#labels = []
				#for i in range(words.shape[0]):
				#	word = words[i]
				#	answ = word.split()
				#	np.random.shuffle(perm)
				#	changed_words.append(answ[perm[0]] + " " + answ[perm[1]] + " " + answ[perm[2]] + " " + answ[perm[3]] + " " + answ[perm[4]])
				#	for k in range(5):
				#		if(perm[k]==0):
				#			labels.append(k)

				#words = np.array(changed_words)
				#print(sentences[0:5])
				#print(words[0:5])
				#print(labels[0:5])
				#words = ["cecidere further unfruitfully obstructively incompetently"]
				#self.model.inference(words, sentences)
				#sentences = ["How _____ your day yesterday evening?"]
				#words = ["runner runner runner runner runner"]
				t,summary = self.sess.run((self.model.accuracy,self.model.summary),feed_dict = {self.model.words : words , self.model.sents : sentences})
				self.train_summary_writer.add_summary(summary, self.epoch)
				ac = (m * ac + t*d) / (m + d)
				m += d
				ind = ind +self.batch_size
				#print("ACC: " + str(a))
				#print(words)
				print("Validated " + str(j) + " samples acc: " + str(t) + ".")

		return ac



	def test(self):
		n = len(self.test_data)
		ac = 0
		ind = 0
		with self.model.graph.as_default():
			while(ind<n):
				sents = self.test_data[ind:ind+1,0]
				words = self.test_data[ind:ind+1,1]
				output = self.sess.run(self.model.output,feed_dict = {self.model.words: words, self.model.sents:sents})
				cur_pred = np.argmax(output,axis=1)[0] + 1
				#print(output)
				#print(str(cur_pred)+" "+str(self.test_data[ind,2]))
				if(str(cur_pred)==str(self.test_data[ind,2])):
					ac+= 1
				#print(ac)
				ind +=1
		self.fl = open("acc.txt", "a+")
		self.fl.write("Test data accuracy: " + str(ac/n)+".\n")
		self.fl.close()
