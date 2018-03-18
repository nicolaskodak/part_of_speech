#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk.corpus import brown
from nltk.corpus import sinica_treebank
import jieba, sys, nltk

class Sent2Tag(object):
	def __init__( self, mode, language ):
		self.mode = mode
		if language.lower()=='en':
			self.tagged_sents = brown.tagged_sents(categories='news')
			self.default_tagger = nltk.DefaultTagger('NN')
		elif language.lower()=='zh':
			self.tagged_sents = sinica_treebank.tagged_sents()   # 以句为单位标
			self.default_tagger = nltk.DefaultTagger('Nab')
		else:
			print( 'only supports en or zh as language.')
			raise
		self.train_size = int(len(self.tagged_sents) * 0.9)
		self.train_sets = self.tagged_sents[:self.train_size]
		self.test_sets = self.tagged_sents[self.train_size:]
		del(self.tagged_sents)	
		if self.mode=='unigram':
			self.unigram_tagger = nltk.UnigramTagger(train=self.train_sets,backoff=self.default_tagger)
		
		if self.mode=='bigram':
			self.unigram_tagger = nltk.UnigramTagger(train=self.train_sets,backoff=self.default_tagger)
			self.bigram_tagger = nltk.BigramTagger(train=self.train_sets,backoff=self.unigram_tagger)
			
	def tag( self, sentence ):
		assert isinstance(sentence, list), 'input must be a list of strings'
		if self.mode=='naive':
			tagged_words = self.default_tagger.tag(sentence)
		elif self.mode=='unigram':
			tagged_words = self.unigram_tagger.tag(sentence)
		elif self.mode=='bigram':
			tagged_words = self.bigram_tagger.tag(sentence)
		return tagged_words
		
	def evaluate( self ):
		if self.mode=='naive':
			print( 'naive => training: {}; testing:{}'.format( self.default_tagger.evaluate(self.train_sets), self.default_tagger.evaluate(self.test_sets)) )
		elif self.mode=='unigram':
			print( 'unigram => training: {}; testing:{}'.format( self.unigram_tagger.evaluate(self.train_sets), self.unigram_tagger.evaluate(self.test_sets)) )
		elif self.mode=='bigram':
			print( 'bigram => training: {}; testing:{}'.format( self.bigram_tagger.evaluate(self.train_sets), self.bigram_tagger.evaluate(self.test_sets)) )
			
if __name__=='__main__':
	lang = sys.argv[1]
	tagger = Sent2Tag( 'bigram','zh')
	jieba.initialize()
	while True:
		text = raw_input("Input a sentence: ")
		print(text)
		print( u' '.join(list(jieba.cut(text)) ) )
		tagged_words = tagger.tag( list(jieba.cut(text)) )
		for _ in tagged_words:
		    print( u': '.join(_) )



