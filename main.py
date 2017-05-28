from __future__ import print_function

from TextBox import *
from Summarizer import *

def ABTest():

	'''
	how this will work:
	- load article
	- generate one optimal solution
	- generate one random solution
	- randomize order
	- present to user
	- ask which is better
	- write results to file (only view after experiment is over)

	'''

	S = Summarizer()

	f = open('input.txt', 'r')
	text = f.read()
	f.close()

	sentences, good_solution, _, _ = S.summarize(text)
	_, random_solution, _, _ = S.summarize(text, random_solution=True, random_density=1.*sum(good_solution)/len(good_solution))

	#print("RANDOM:", random_solution)
	#print("  GOOD:", good_solution)

	f = open('AB_results.txt', 'a')

	if random.random() < 0.5:
		print("OPTION A")
		printSolution(sentences, random_solution)
		print("OPTION B")
		printSolution(sentences, good_solution)

		chosen = raw_input("Which is better (A/B)?").lower()
		f.write("Correct: {}\n".format(chosen == 'b'))
	else:
		print("OPTION A")
		printSolution(sentences, good_solution)
		print("OPTION B")
		printSolution(sentences, random_solution)

		chosen = raw_input("Which is better (A/B)?").lower()
		f.write("Correct: {}\n".format(chosen == 'a'))

	f.close()
		

	return


if __name__ == "__main__":
	print("sup")
	T = TextBox()

	T.root.mainloop()
	#ABTest()
