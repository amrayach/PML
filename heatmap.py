import configparser
import itertools
import os
import utils

from collections import Counter
from random import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import squarify
import pickle
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import pdflatex


latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, colors=['red','blue'], rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	latex_str = (r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
	string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
	for idx in range(word_num):
		col = colors[0] if attention_list[idx] > 30 else colors[1]
		string += "\\colorbox{%s!%s}{"%(col, attention_list[idx])+"\\strut " + text_list[idx]+"} "
	string += "\n}}}"
	latex_str = latex_str + string + '\n'
	latex_str += (r'''\end{CJK*}
\end{document}''')

	pdfl = pdflatex.PDFLaTeX.from_binarystring(latex_str.encode('utf-8'), 'my_file')
	pdf, log, cp = pdfl.create_pdf()

	with open('./heatmaps/' + latex_file+'.pdf', 'wb') as f:
		f.write(pdf)

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()

def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list

def load_data(args, mode='train'):
    text = []
    usecols = list(map(lambda x: int(x), args.get('Data', 'usecols').split(',')))
    path = args.get('Data', 'dataset') + '/' + mode + '.csv';
    print('\n path: ' + path)
    data = pd.read_csv(path,
                       usecols=usecols,
                       encoding=args['Data'].get('encoding'),
                       sep=args['Data'].get('csv_sep'),
                       doublequote=True)
    labels = data.iloc[:, 0].tolist()
    if args.get('Data', 'dataset') == 'yelp':
        text = data.iloc[:, 1].tolist()
    elif args.get('Data', 'dataset') == 'ag_news':
        text = (data.iloc[:, 1] + ' ' + data.iloc[:, 2]).tolist()

    text = [remove_stop_words(t) for t in text]
    text = [remove_special_characters(t) for t in text]
    text = [remove_numbers(t) for t in text]
    text = [remove_single_letters(t) for t in text]
    text = [remove_single_characters(t) for t in text]
    return labels, text


if __name__ == '__main__':
    args = configparser.ConfigParser()
    args.read('./argsConfig.ini')

    if not os.path.exists('heatmaps'):
        os.makedirs('heatmaps')

    # TODO: better to get the words already preprocessed
    dummy_text = '''Mobile broadband for Public Safety is no longer a vision, it is reality. 4G is chosen as the technology for LMR replacement. In this video, Niklas Spångberg explains why CSPs play a key role in delivering mission-critical mobile broadband services.'''
    words = dummy_text.split()
    word_num = len(words)

    # TODO: replace the random generator with the word predictions
    attention = [(x + 1.) / word_num * 100 for x in range(word_num)]

    import random
    random.seed(42)
    random.shuffle(attention)

    colors = ['red', 'blue']
    generate(words, attention, "sample_heatmap", colors)

    print("you'll find the generated heatmap in heatmaps folder")
