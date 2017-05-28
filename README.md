# QuickReaderV4
A text summarisation algorithm and interface

![screenshot](https://github.com/tannerbohn/QuickReaderV4/blob/master/screenshot.png)


## Usage
To run the code, `python main.py`. When you find an article or document to summarize, copy the text, paste it into the white box at the bottom of the interface, and press enter.

Inside `TextBox.py`, you can set values associated with the colour scheme and font.

Inside `Summarizer.py` are the algorithms for the text summarization.

When using the program, the following operations are available:
* update (button): after changing the coherence or independence params (but not length), press update to recalculate the summary
* less/more (button): use this to toggle between showing only the summary or showing the entire document with the summary highlighted
* length (slider): this changes how many sentences are extracted for the summary. When set to 0, the entire document is shown.
* coherence (slider): this changes the weight of coherence (i.e. how important it is for consecutive sentences in the summary to be related)
* independence (slider): this changes the weight of independence (i.e. how important it is to include sentences in the summary that introduce new content/do not depend on previous sentences for comprehension)

## Dependencies
This project is written in Python2.7 with Tkinter for the GUI. It requires the following non-default libraries:
* nltk
* numpy, scipy
* ScrolledText
