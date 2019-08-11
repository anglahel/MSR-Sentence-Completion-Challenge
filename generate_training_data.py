import numpy as np
import nltk
import pandas as pd
import random
import os
import re

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def gen(lst,coef):
	
	lst.sort()
	n = len(lst)
	gen_lst = []
	for i in range(n-coef+1):
		if(lst[i+coef-1]==lst[i] and (i==0 or lst[i]!=lst[i-1])):
			gen_lst.append(lst[i])
	return gen_lst
#
#
# adj_file = "parts of speech word files/adjectives/28K adjectives.txt"
# adv_file = "parts of speech word files/adverbs/6K adverbs.txt"
# noun_file = "parts of speech word files/nouns/91K nouns.txt"
# verb_file = "parts of speech word files/verbs/31K verbs.txt"

adjs = []
advs = []
nouns = []
verbs = []

path = "data/Holmes_Training_Data/"
files = os.listdir(path)
files = files[0:10]
er_cnt = 0
coef = 15
for file_id in range(len(files)):
	file = files[file_id]
	try:
		f = open(path + file, "r")
		text = f.read()
		text = text.split("*END*")[-1]
		sentences = split_into_sentences(text)
		for sentence in sentences:
			words = nltk.word_tokenize(sentence)
			for word in words:
				valid_word = True
				for char in word:
					ok = False
					if(char>='a' and char<='z'):
						ok = True
					if(char=='\''):
						ok = True
					if(ok == False):
						valid_word = False
				
				if(valid_word):
					tg = nltk.pos_tag(nltk.word_tokenize(word),tagset = 'universal')
					tg = tg[0]
					if(len(tg)>1):
						if(tg[1] == 'NOUN'):
							nouns.append(tg[0])
						elif(tg[1] == 'VERB'):
							verbs.append(tg[0])
						elif(tg[1] == 'ADJ'):
							adjs.append(tg[0])
						elif(tg[1] == 'ADV'):
							advs.append(tg[0])
	except:
		er_cnt = er_cnt + 1

	print("Extracting words in file " + str(file_id + 1) + "# done!")

print("Done!")

nouns = gen(nouns,coef)
verbs = gen(verbs,coef)
adjs = gen(adjs,coef)
advs = gen(advs,coef)
print("Noun count: " + str(len(nouns)))
print("Verb count: " + str(len(verbs)))
print("Adjective count: " + str(len(adjs)))
print("Adverb count: " + str(len(advs)))


blank = "_____"
dat = []
cnt = 1
file_id = 0
for file in files:
	file_id = file_id + 1
	try:
		f = open(path + file, "r")
		text = f.read()
		text = text.split("*END*")[-1]
		sentences = split_into_sentences(text)
		for sentence in sentences:
			valid_sentence = True
			for char in sentence:
				ok = False
				if(char>='a' and char<='z'):
					ok = True
				if(char>='A' and char<='Z'):
					ok = True
				if(char == '.' or char == ',' or char == '!' or char == '?' or char == ' '):
					ok = True
				if(char=='\'' or char=='-' or char==';' or char==':'):
					ok = True
				if(ok == False):
					valid_sentence = False

			words = nltk.word_tokenize(sentence)
			if(len(words)>=5 and valid_sentence):
				modif = []
				for word in words:
					valid_word = True
					for char in word:
						ok = False
						if(char>='a' and char<='z'):
							ok = True
						if(ok == False):
							valid_word = False
					tg = nltk.pos_tag(nltk.word_tokenize(word),tagset = 'universal')
					tg = tg[0]
					if(len(tg)>1 and valid_word):
						if(tg[1] == 'NOUN' or tg[1] == 'ADJ' or tg[1] == 'ADV' or tg[1] == 'VERB'):
							modif.append(tg)
				if(len(modif)>0):
					r = np.random.randint(len(modif))
					ind = 0
					for word in words:
						if(word == modif[r][0]):
							break
						ind = ind + 1

					#syns = nltk.corpus.wordnet.synsets(modif[r][0])
					#print("Acctual word: "+modif[r][0])
					#try:
					#	print("Synonyms:")
					#	print(syns[0].lemmas()[0].name()) 
					#	print(syns[1].lemmas()[0].name()) 
					#	print(syns[2].lemmas()[0].name()) 
					#	print(syns[3].lemmas()[0].name())  
					#except:
					#	print("Synonyms don't exist!")
					stc = ""
					for i in range(len(words)):
						if(i!=ind):
							stc = stc + words[i]
						else:
							stc = stc + blank
						if(i+1 != len(words) and words[i+1]!=',' and words[i+1]!='.' and words[i+1]!='!' and words[i+1]!='?'):
							stc = stc + " "
					ans = [modif[r][0]]
					for i in range(4):
						if(modif[r][1]=='NOUN'):
							rt = np.random.randint(len(nouns))
							ans.append(nouns[rt])
						elif(modif[r][1]=='ADJ'):
							rt = np.random.randint(len(adjs))
							ans.append(adjs[rt])
						elif(modif[r][1]=='ADV'):
							rt = np.random.randint(len(advs))
							ans.append(advs[rt])
						elif(modif[r][1]=='VERB'):
							rt = np.random.randint(len(verbs))
							ans.append(verbs[rt])
					
					random.shuffle(ans)
					query = [stc]
					ind = 0
					for i in range(5):
						if(ans[i] == modif[r][0]):
							ind = i
						query.append(ans[i])
					
					if(ind == 0):
						circ = 'a'
					elif(ind == 1):
						circ = 'b'
					elif(ind == 2):
						circ = 'c'
					elif(ind == 3):
						circ = 'd'
					elif(ind == 4):
						circ = 'e'
					query.append(circ)
					dat.append(query)
		f.close()
	except:
		print("Error in file: " + file + "!")

	print("Processing queries in file " + str(file_id) + "# done!")

random.shuffle(dat)

print(str(len(dat)) + " queries found.")

que = []
ans = []
for ind in range(len(dat)):
	entry = dat[ind]
	que.append([str(ind+1)] + entry[0:-1:1])
	ans.append([str(ind+1),entry[-1]])
	#print(entry)


df = pd.DataFrame(que, columns =['id','question','a)','b)','c)','d)','e)'])
df_ans = pd.DataFrame(ans, columns =['id','answer'])
df.to_csv("training_data.csv", encoding='utf-8', index=False)
df_ans.to_csv("train_answer.csv", encoding='utf-8', index=False)
