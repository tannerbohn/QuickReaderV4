from __future__ import print_function

from Tkinter import *
from ScrolledText import ScrolledText

import random
from Summarizer import *

# todo: TextBox.articleColour() could use a careful look at...
#       it seems to sometimes miss non-alphanum characters
#       and has trouble when there is more than one exact copy
#       of a summary sentence

class TextBox:


	def __init__(self):

		self.WIDTH = 600
		self.HEIGHT = 800
		self.FONT = "helvetica"
		self.FONT_SIZE = 12

		self.bg_input = [1,1,1]
		self.fg_input = [0,0,0]

		self.bg_article = [0,0,0]
		self.fg_min_article =  [0.5, 0.5, 0.5]
		self.fg_max_article = [0.9, 0.9, 0.9]
		self.fg_solution_article = [1,1,1] #[0.3, 0.5, 1.0] #[1, 0.7, 0.4]

		invert = False
		if invert:
			self.bg_input = [1.-v for v in self.bg_input]
			self.fg_input = [1.-v for v in self.fg_input]

			self.bg_article = [1.-v for v in self.bg_article]
			self.fg_min_article = [1.-v for v in self.fg_min_article]
			self.fg_max_article = [1.-v for v in self.fg_max_article]

		self.text = "" # what is shown in the box
		self.allText = "" # the text for the entire article
		self.sentences = [] # list of sentences in article
		# dictionary mapping from size to k-hot encoding indicating
		# which sentences are in the summary
		self.solutions = [] 
		# (not used) how much weight is put on each sentence
		self.weights = []

		self.only_summary = True
		self.summary_size = 1
		self.summary_coherence = 0.0
		self.summary_independence = 0.8

		self.summarizer = Summarizer(parent=self)

		self.root = Tk()

		self.draw(init=True)

		#self.root.mainloop()

	def draw(self, init=False):

		if init:
			# show main article body
			self.tk_article = ScrolledText(self.root)

			# let user paste and enter text
			self.tk_user_input = ScrolledText(self.root)

			
			self.tk_summary_size_scale = Scale(self.root)
			self.tk_summary_size_scale_label = Label(self.root, text="Length")

			
			self.tk_summary_coherence_scale = Scale(self.root)
			self.tk_summary_coherence_scale_label = Label(self.root, text="Coherence")

			self.tk_summary_independence_scale = Scale(self.root)
			self.tk_summary_independence_scale_label = Label(self.root, text="Independence")

			
			self.tk_toggle_view = Button(self.root, text="more", command=self.handleToggleView)
			self.tk_recalculate = Button(self.root, text="Update", command=self.handleRecalculate)

			self.root.geometry("%dx%d" % (self.WIDTH, self.HEIGHT))
			self.root.title("QuickReader V4")


			self.tk_article.configure(width=25, height=6, bd=0, highlightthickness=0, wrap="word", font=self.FONT)

			self.tk_user_input.configure(width=25, height=3, bd=0,highlightthickness=0, wrap="word", font=self.FONT)

			self.tk_summary_size_scale.configure(bd=0, from_=0, to=20, orient=HORIZONTAL, sliderrelief=FLAT,
				command= lambda event: self.handleSlider(self.tk_summary_size_scale.get()))

			######
			self.tk_summary_coherence_scale.configure(bd=0, from_=0, to=1, orient=HORIZONTAL, sliderrelief=FLAT, resolution=0.05,
				command= lambda event: self.handleCoherenceSlider(self.tk_summary_coherence_scale.get()))

			self.tk_summary_coherence_scale.set(self.summary_coherence)

			######
			self.tk_summary_independence_scale.configure(bd=0, from_=0, to=1.5, orient=HORIZONTAL, sliderrelief=FLAT, resolution=0.05,
				command= lambda event: self.handleIndependenceSlider(self.tk_summary_independence_scale.get()))

			self.tk_summary_independence_scale.set(self.summary_independence)



			# set colours
			self.root.configure(background="black")

			self.tk_summary_size_scale.configure(troughcolor="#444444", fg="black", background="white", activebackground="#bbbbbb")

			self.tk_summary_coherence_scale.configure(troughcolor="#444444", fg="black", background="white", activebackground="#bbbbbb")

			self.tk_summary_independence_scale.configure(troughcolor="#444444", fg="black", background="white", activebackground="#bbbbbb")

			self.tk_article.configure(bg=toHex(self.bg_article), fg = "white", insertbackground="blue")
			self.tk_article.vbar.configure(bg="white", width=10, troughcolor="black")

			self.tk_user_input.configure(bg=toHex(self.bg_input), fg = toHex(self.fg_input), insertbackground="blue")
			self.tk_user_input.vbar.configure(bg="white", width=10, troughcolor="black")

			self.tk_user_input.focus()
			self.tk_user_input.bind("<KeyRelease-Return>",(lambda event: self.handleUserInput(self.tk_user_input.get("0.0",END))))
			self.root.bind("<Configure>", self.resize)

	def setText(self, text, redraw = False):
		self.text = text
		if redraw: self.updateArticleInfo()

	def setSentences(self, sentences, redraw=False):
		self.sentences = sentences
		if redraw: self.updateArticleInfo()

	def setSolutions(self, solutions, redraw=False):
		self.solutions = solutions
		if redraw: self.updateArticleInfo()

	def setWeights(self, weights, redraw=False):
		self.weights = weights
		if redraw: self.updateArticleInfo()

	def handleToggleView(self):

		print("View toggle!")

		self.only_summary = not self.only_summary

		if self.only_summary:
			self.tk_toggle_view.configure(text = "more")
		else:
			self.tk_toggle_view.configure(text = "less")

		self.updateSummary()

	def handleRecalculate(self):
		print("Update!")

		self.handleUserInput(self.allText)

	def handleSlider(self, value):

		print("Slider:", value)

		self.summary_size = value

		self.updateSummary()

	def handleCoherenceSlider(self, value):

		print("Coherence Slider:", value)

		self.summary_coherence = value

		#self.updateSummary()

	def handleIndependenceSlider(self, value):

		print("Independence Slider:", value)

		self.summary_independence = value

		#self.updateSummary()

	def updateSummary(self):

		l = self.summary_size

		if self.only_summary and l != 0:
			self.setText('\n\n'.join([self.sentences[i] for i in range(len(self.sentences)) if self.solutions[l][i] == 1]))
		else:
			self.setText(self.allText, redraw=False)

		self.updateArticleInfo()

		self.setWeights([0. for _ in self.sentences], redraw=True)

		self.tk_article.yview_moveto(0) #vbar.set(0, 0) #configure(jump=0)


	def handleUserInput(self, inStr):
		self.tk_user_input.delete("0.0", END)

		if inStr.strip() == "": return

		
		text = inStr

		text = ''.join([ch for ch in text if ord(ch) < 128])

		self.setText(text, redraw=False)
		self.setSolutions([], redraw=False)
		self.setWeights([], redraw=True)
		
		text, sentences, solutions = self.summarizer.summarize(text, coherence_weight=self.summary_coherence, independence_weight=self.summary_independence,
																size_weight=1., beam_width=3, hard_size_limit=None)

		self.allText = text
		self.sentences = sentences
		self.solutions = solutions

		self.solutions[0] = [1. for _ in sentences]

		# get max length for summary
		max_len = max(solutions.keys())
		set_len = min(max_len, 3)
		self.tk_summary_size_scale.configure(from_=0, to=max_len)
		self.tk_summary_size_scale.set(set_len)
		self.summary_size = set_len

		# text: all the text in one long string
		# sentences: the text split up into a list of sentences
		# solution: dictionary mapping summary size to a one-hot vector over the sentences, indicating
		#   which sentences are included in the summarization

		# the text should be the same, but update it anyways since it needs to contain the
		#  exact same stuff as the sentences
		
		self.updateSummary()

		#self.updateArticleInfo()

	def resize(self, event=[]):
		LINEH = 20.0

		pixelX=self.root.winfo_width()
		pixelY=self.root.winfo_height()

		bf = 5 # buffer size in pixels

		# update find_icon, wiki_icon, and graph_icon

		# set toggle and recalculate button
		toggleW = 50
		toggleH = 35*1
		self.tk_toggle_view.place(x=pixelX-toggleW, y=0, width = toggleW, height= toggleH)

		updateW = 50
		updateH = 35*2
		self.tk_recalculate.place(x=pixelX-updateW, y=toggleH, width = updateW, height= updateH)


		buttonH = toggleH + updateH


		labelW = 90

		# set position of size scale
		scaleW = pixelX - updateW - labelW
		scaleH = 35
		self.tk_summary_size_scale.place(x=labelW, y=0, width = scaleW, height= scaleH)

		self.tk_summary_size_scale_label.place(x=0, y=0, width=labelW, height=scaleH)

		# set position of coherence scale
		coherenceW = pixelX - updateW - labelW
		coherenceH = 35
		self.tk_summary_coherence_scale.place(x=labelW, y=scaleH, width = scaleW, height= scaleH)

		self.tk_summary_coherence_scale_label.place(x=0, y=scaleH, width=labelW, height=coherenceH)

		# set position of independence scale
		independenceW = pixelX - updateW - labelW
		independenceH = 35
		self.tk_summary_independence_scale.place(x=labelW, y=scaleH+coherenceH, width = scaleW, height= scaleH)

		self.tk_summary_independence_scale_label.place(x=0, y=scaleH+coherenceH, width=labelW, height=independenceH)

		# update user input
		inputW = pixelX
		inputH = int(3.0*LINEH)
		self.tk_user_input.place(x=0, y=pixelY-inputH, width = inputW, height = inputH)

		# update article
		articleW = pixelX
		articleH = pixelY - inputH - scaleH - coherenceH - independenceH
		self.tk_article.place(x=0, y=scaleH+coherenceH+independenceH, width = articleW, height = articleH)

	def updateArticleInfo(self):

		self.articleClear()

		self.articleCat(self.text)

		if self.weights != []:
			self.articleColour()

		self.root.update()


	def articleClear(self):
		self.tk_article.delete("1.0", END)
		self.tk_article.update()

		self.root.update()
		
		return

	def articleCat(self, inStr):

		self.tk_article.insert(END, inStr)

		self.tk_article.yview(END)


	def articleColour(self):

		'''
		solution = self.solutions[self.summary_size]

		allText = self.text #self.tk_article.get('1.0', 'end-1c')

		# make sure weights are normalised
		maxW = max(self.weights)
		minW = min(self.weights)

		weights = self.weights
		if maxW != minW:
			weights = [(v-minW)/(maxW-minW) for v in self.weights]

		for i in range(len(self.sentences)):
			if self.only_summary and solution[i] != 1.: continue

			s = self.sentences[i]
			if len(s.strip()) == 0:

				continue

			tagNameA = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10)])
			L_Size = 12 # if solution[i] == 1 else 10
			
			L_Colour = blend(self.fg_min_article, self.fg_max_article, weights[i])
			L_Colour = self.fg_solution_article if solution[i] == 1 else L_Colour

			countVar = StringVar(self.root)
			pos = self.tk_article.search(s, "1.0", stopindex="end", count=countVar)

			self.tk_article.tag_add(tagNameA, pos, "{} + {}c".format(pos, countVar.get()))

			bolding = "normal" #"bold" if self.solution[i] == 1 else "normal" #
			font = (self.FONT, L_Size, bolding)
			self.tk_article.tag_config(tagNameA, foreground=toHex(L_Colour), font=font)#self.FONT+' %s'%(L_Size))

		
		self.root.update()
		'''

		solution = self.solutions[self.summary_size]

		allText = self.text #self.tk_article.get('1.0', 'end-1c')

		#print("=========")
		for i in range(len(self.sentences)):
			if self.only_summary and solution[i] != 1.: continue

			s = self.sentences[i]
			#if len(s.strip()) == 0:
			#	continue

			#print("- ", s)

			tagNameA = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10)])
			L_Size = self.FONT_SIZE # if solution[i] == 1 else 10

			L_Colour = self.fg_solution_article if solution[i] == 1 else self.fg_min_article
			#print("\t", L_Colour)

			countVar = StringVar(self.root)
			pos = self.tk_article.search(s, "1.0", stopindex="end", count=countVar)

			self.tk_article.tag_add(tagNameA, pos, "{} + {}c".format(pos, countVar.get()))

			bolding = "normal" #"bold" if self.solution[i] == 1 else "normal" #
			font = (self.FONT, L_Size, bolding)
			self.tk_article.tag_config(tagNameA, foreground=toHex(L_Colour), font=font)#self.FONT+' %s'%(L_Size))

		
		self.root.update()



def interpolate(v1, v2, frac):
	newC = v1*(1.-frac) + v2*frac
	return newC

def blend(C1, C2, frac):
	newC = [interpolate(v1, v2, frac) for v1, v2 in zip(C1, C2)]
	return newC


def toHex(cvec):

	rgb = tuple([int(255*v) for v in cvec])

	return '#%02x%02x%02x' % rgb


