from __future__ import print_function

import nltk.data
import pickle
import random
import copy
import numpy as np
from scipy import spatial
import math
import time

DIR = ""

class Summarizer:

	def __init__(self, parent=None, ng=4):

		self.parent=parent

		self.ng = ng #n-gram length

		self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		self.chAllowableSet = "abcdefghijklmnopqrstuvwxyz ()"

		self.STOP = []
		self.COMMON_HIST_DICT = None

		# uncomment this to use remove stopwords and not* subtract common ngram freqs
		with open(DIR+'stopwords.txt', 'r') as f:
			self.STOP = set(f.read().lower().split())

		# uncomment this to subtract common ngram freqs (i.e. don't worry about "common knowledge")
		# * remove not to subtract stopwords when using this
		
		'''
		with open(DIR+'grams.pkl', 'r') as f:
			D = dict(pickle.load(f)[ng])

			hsum = sum([D[v] for v in D])
			for k in D:
				D[k] *= 1./hsum
				# don't forget to rescale to doc size when doing calculations

			self.COMMON_HIST_DICT = D
		'''


		return

	def summarize(self, text, random_solution=False, random_density=None, coherence_weight=0.2, independence_weight=0.8, size_weight=0.4, beam_width=3, hard_size_limit=None):



		# remove non-ascii characters
		text = ''.join([ch for ch in text if ord(ch) < 128])

		# get list of sentences
		pre_sentences = self.tokenizer.tokenize(text)
		sentences = []
		for s in pre_sentences:
			sfix = s.split('\n\n')
			sentences.extend(sfix)

		# make sentences lower case and remove stop words
		cSentences = [' '.join([w for w in sen.lower().split() if w not in self.STOP]) for sen in sentences]
		# remove non alpha-num (for optimization)
		cSentences = [''.join([ch for ch in sen if ch in self.chAllowableSet]) for sen in cSentences]
		# collapse multiple repeated spaces that may have resulted from removing unwanted chars
		cSentences = [' '.join(sen.split()) for sen in cSentences]



		#t1 = time.time()
		solutions = self.optimizeSet(cSentences, coherence_weight=coherence_weight, independence_weight=independence_weight,
								size_weight=size_weight, beam_width=beam_width, hard_size_limit=hard_size_limit)
		#t2 = time.time()

		#print("Duration: {:.3f} seconds".format(t2-t1))
		# solutions: dict of k-hots over sentences
		return text, sentences, solutions


	def optimizeSet(self, sentences, coherence_weight=0.1, independence_weight=0.2, size_weight=0.2, beam_width=5, hard_size_limit=None):
		# sentences: a list of sentences, all coverted to lower case with stopwords removed

		keys, fullHist, sentenceHists = getHistograms(sentences, commonHist = self.COMMON_HIST_DICT, ng=self.ng) # 
		# keys: the words in order (corresponding to histogram index)
		# fullHist: the histogram of n-gram freqs for the whole text body
		# sentenceHists: list of histograms -- one for each sentence

		n = len(sentences)

		#for i, sen in enumerate(sentenceHists):
		#	print(i, sum(sen), sentences[i])
		#	print()

		# at every step in optimization, need to check:
		#	- semantic score
		#	- size penalty
		#	- incoherence

		# already subtracted common knowledge hist
		current_knowledge = [0. for _ in keys]


		goal_knowledge = fullHist

		# reset precalculated sentence novelties list
		sentence_independence_hists = getSentenceIndependenceHists(sentenceHists)

		#for i in range(len(sentences)):

		#cur_solution = [0 for _ in range(n)]
		#cur_score = getSolutionScore(cur_solution, sentenceHists, goal_knowledge)

		best_solution_per_size = beamSearch(sentenceHists, goal_knowledge, sentence_independence_hists,
									coherence_weight=coherence_weight, independence_weight=independence_weight,
									size_weight=size_weight, beam_width=beam_width, hard_size_limit=hard_size_limit)

		#print()

		max_size = max(best_solution_per_size.keys())

		#for size in best_solution_per_size:
		#	print("Size:", size)
		#	print("Solution:", best_solution_per_size[size][0])
		#	print("Score:", best_solution_per_size[size][1])

		#best_solution, best_score = best_solution_per_size[max_size]

		# convert solution to one-hot encoding
		#best_solution = [1 if i in best_solution else 0 for i in range(n)]
		
		for size in best_solution_per_size:
			
			sen_list = best_solution_per_size[size][0]

			best_solution_per_size[size] = [1 if i in sen_list else 0 for i in range(n)]

		return best_solution_per_size

		#heatmap = [0.]*n	

		#return best_solution, best_score, heatmap


def beamSearch(sentenceHists, goal_knowledge, sentence_independence_hists, coherence_weight=0.1, independence_weight=0.2, size_weight=0.2, beam_width=5, hard_size_limit=None):

	n = len(sentenceHists)

	max_depth = min(2*int(n**0.5), n)

	if hard_size_limit != None:
		max_depth = hard_size_limit

	best_solution_per_size = dict()

	best_solutions = [[[], -float('inf')]]

	for depth in range(max_depth):#min(max_depth, n)):
		#print("depth = ", depth)

		sol_list_len = len(best_solutions)
		for sol in best_solutions[:sol_list_len]:
			if len(sol[0]) < depth: continue

			#print("\tsol = ", sol)

			start_index = 0 if depth == 0 else sol[0][-1]+1
			
			for i in range(start_index, n):
				cur_solution = sol[0]+[i]
				score = getSolutionScore(cur_solution, sentenceHists, goal_knowledge, sentence_independence_hists,
										coherence_weight=coherence_weight, independence_weight=independence_weight,
										size_weight=size_weight)

				best_solutions.append([cur_solution, score])

				#print('\t\tsol = ', score, cur_solution)

		# remove old solutions
		best_solutions = [sol for sol in best_solutions if len(sol[0]) == depth+1]

		if len(best_solutions) == 0:
			#print("DONE")
			#print("\tdepth = ", depth)
			break

		# sort from best to worst
		best_solutions = sorted(best_solutions, key=lambda x:-x[1])

		# take only the top beam_width # solutions
		best_solutions = best_solutions[:beam_width]

		# save the best solution at this depth
		best_solution_per_size[depth+1] = best_solutions[0]


		#print("BEST SOLUTIONS:", best_solutions)


	#best_solution, best_score = best_solutions[0]
	

	return best_solution_per_size #best_solution, best_score


def getHistograms(sentences, commonHist=None, ng=3):
	# ng: length of n-grams (letter sequences)
	# first get the histogram for each sentence
	sentenceHists = []
	for sen in sentences:
		#sen += " "*ng
		sen_freqs = dict()

		
		total = 0
		for i in range(len(sen)-ng+1):

			chars = sen[i:i+ng]
			#if ' ' in chars:
			if chars[0] == ' ':
				continue

			total += 1
			try:
				sen_freqs[chars] += 1.
			except:
				sen_freqs[chars] = 1.
		
		'''
		# use this for a word-level histogram (bag of words)
		# does seem to work as well as n-gram (still need accurate ROUGE measure)
		for w in sen.split():

			try:
				sen_freqs[w] += 1.
			except:
				sen_freqs[w] = 1.
		'''

		sentenceHists.append(sen_freqs)

	# remove "common knowledge"
	if commonHist != None:
		for sh in sentenceHists:
			sh_sum = sum([sh[k] for k in sh])

			for k in sh:
				try:
					common_freq = commonHist[k]
				except:
					common_freq = 0.
				# need to multiply by sh_sum since commonHist is normalized to sum=1
				sh[k] = max(0., sh[k] - 1.*common_freq*sh_sum)



	# now get the histogram/dict for the full (cleaned) body of text
	fullHist = joinFreqs(sentenceHists, normalize=False)

	# make sure each sentence hist has the same set of entries
	elems = fullHist.keys()
	elems.sort()

	# only use ngrams if associated count > 1
	#print(len(elems))
	elems = [e for e in elems if fullHist[e] > 1]
	#print(len(elems))

	for sh in sentenceHists:
		for el in elems:
			try:
				sh[el] += 0.
			except:
				sh[el] = 0.

	# now convert histogram dicts to vecs
	
	fullHist = [fullHist[k] for k in elems]
	sentenceHists = [[sh[k] for k in elems] for sh in sentenceHists]


	return elems, fullHist, sentenceHists

def joinFreqs(freqDicts, normalize=True):
	#input: list of dictionaries of sentence letter pair counts
	totalFreqs = dict()
	total = 0.
	for fd in freqDicts:
		for k in fd:
			total += fd[k]
			try:
				totalFreqs[k] += fd[k]
			except:
				totalFreqs[k] = fd[k]
	if normalize:
		for k in totalFreqs:
			totalFreqs[k] = 1.*totalFreqs[k]/total
	return totalFreqs


def getRandomSolution(n, p=None):
	# n: number of sentences
	
	if p == None:
		p = random.random()

	#solVec = [1 if random.random() < p else 0 for _ in range(n)]
	solVec = [0 for i in range(n)]
	indices = range(n)
	random.shuffle(indices)
	indices = indices[:int(p*n)]
	for i in indices:
		solVec[i] = 1


	return solVec




def getFreqDist(freqsA, freqsB):
	# freqs in each dictionary should be normalized

	# TODO: include different importance for each mismatch?

	dSum = 0.
	matchCount = 0
	missCount = 0
	weightSum = 0.
	l = 0
	for k in freqsA:
		vA = freqsA[k]
		weight = vA
		weightSum += vA
		l += 1
		try:
			vB = freqsB[k]
			dSum += abs(vA - vB)*weight
			matchCount += 1
		except:
			dSum += vA*weight
			missCount += 1

	dSum = 1.*l*dSum/weightSum

	return dSum



def getIncoherence(sentenceHists):
	if len(sentenceHists) <= 1: return 0.

	incoherence = 0.

	n = len(sentenceHists)

	for i in range(n-1):
		s1 = sentenceHists[i]
		s2 = sentenceHists[i+1]

		# there should be some terminology intersection between summary sentences
		# when dealing with histograms (all non-negative), cosine dist falls between 0 and 1
		d = spatial.distance.cosine(s1, s2)**2

		if math.isnan(d):
			incoherence += 1.
		else:
			incoherence += d

		#print("incoherence = ", incoherence)

	try:
		incoherence /= n-1.
	except:
		incoherence = 0.


	return incoherence

def getSemanticDistance(used_sents, fullHist):

	new_freqs = np.sum(used_sents, axis=0)

	# get cosine distance
	try:
		d = spatial.distance.cosine(new_freqs, fullHist)
		# d = 0: same
		# d = 1: completely different
	except:
		d = 1.

	return d

def getSizePenalty(used_sents, fullHist):
	# threshold: fraction of total length of text to use

	# return value between 0 and 1
	#	- piecewise linear, but derivative increases after threshold

	new_freqs = np.sum(used_sents, axis=0)

	fracUsed = 1.*np.sum(new_freqs)/np.sum(fullHist)

	size_penalty = fracUsed

	return size_penalty

def getDependencePenalty(solution, sentenceHists, sentence_independence_hists):

	# for each sentence, subtract the hist vector from all sentences after
	# (equivalent to keeping running sum and subtracting from next element)
	# - after doing this, a small hist vec indicated it is dependent on previous sentence, 
	#   and a large hist vec probably means it introduces a new term

	#sentenceDeps = []

	# want to choose sentences that have a large frac of retained info
	retained_info = []

	for i in solution:
		try:
			ri = 1.*sum(sentence_independence_hists[i])/sum(sentenceHists[i])
		except:
			# this will happen if the sentence is too small (sentence tokenization error)
			#   of if it doesn't contain and valid characters
			ri = 0. #print("ERROR:", i, sentenceHists[i])
		retained_info.append(ri)

	penalty = 1. - (1.*sum(retained_info)/len(retained_info))
	return penalty


def getSentenceIndependenceHists(sentenceHists):
	# for each sentence, subtract the hist vector from all sentences after
	# (equivalent to keeping running sum and subtracting from next element)
	# - after doing this, a small hist vec indicated it is dependent on previous sentence, 
	#   and a large hist vec probably means it introduces a new term

	sentenceDeps = []

	alpha = 1. # how effectively past information reduces current novelty
	beta = 1. # temporal decay of information

	depSum = [0. for _ in sentenceHists[0]]

	for i, sen in enumerate(sentenceHists):
		sen_p = [max(0., v - s) for v, s in zip(sen, depSum)]
		#sen_p = [v - s for v, s in zip(sen, depSum)]
		depSum = [a*alpha+b*beta for a, b in zip(sen, depSum)]
		sentenceDeps.append(sen_p)

	return sentenceDeps

def getSolutionScore(solution, sentenceHists, fullHist, sentence_independence_hists, coherence_weight=0.1, independence_weight=0.2, size_weight=0.2):
	# how to get score:
	# calculate freqs for new solution
	# calculate dist between freqs of new text and full text
	n = len(sentenceHists)

	#used_sents = [sentenceHists[i] for i in range(n) if solution[i] == 1]
	used_sents = [sentenceHists[i] for i in solution]
	new_freqs = np.sum(used_sents, axis=0)

	semantic_distance = getSemanticDistance(used_sents, fullHist)
		
	size_penalty = getSizePenalty(used_sents, fullHist)

	incoherencePenalty = getIncoherence(used_sents)

	# calculate sentence dependency graph
	# only want to include sentences if they introduce a term/subject or it has
	#  already been intoduced
	dependencePenalty = getDependencePenalty(solution, sentenceHists, sentence_independence_hists)


	# todo: include term so that the "topic" phrase/term is included near the
	#       beginning of the summary

	#print(semantic_distance, size_penalty, incoherencePenalty)
	

	return -(semantic_distance*1. + size_penalty*size_weight + incoherencePenalty*coherence_weight + dependencePenalty*independence_weight)

def printSolution(sentences, solution):
	assert(len(sentences) == len(solution))

	for i in range(len(sentences)):
		if solution[i] == 1.:
			#print("{}: {}\n".format(i, sentences[i]))
			print("{}\n".format(sentences[i]))

if __name__ == "__main__":

	# given a piece of text, calculate the letter freqs and compare to lang avg.
	# construct set of sentences that most accurately reflects the difference
	# - hopefully this set contains the most important information of the text

	print("meow")