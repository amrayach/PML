import configparser
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import squarify

from utils import remove_double_new_line, remove_special_characters, remove_stop_words, lower, remove_numbers, \
    remove_single_letters, remove_single_characters


def load_data(args, mode='train'):
    usecols = list(map(lambda x: int(x), args.get('Data', 'usecols').split(',')))
    path = args.get('Data', 'dataset') + '/'+ mode +'.csv';
    print('\n path: ' + path)
    data = pd.read_csv(path,
                         usecols=usecols,
                         encoding=args['Data'].get('encoding'),
                         sep=args['Data'].get('csv_sep'),
                         doublequote=True)
    labels  = data.iloc[:, 0].tolist()
    if args.get('Data', 'dataset') == 'yelp':
        text    = data.iloc[:, 1].tolist()
    elif args.get('Data', 'dataset') == 'ag_news':
        text = (data.iloc[:, 1] + ' ' + data.iloc[:, 2]).tolist()

    text = [remove_stop_words(t) for t in text]
    text = [remove_special_characters(t) for t in text]
    text = [remove_numbers(t) for t in text]
    text = [remove_single_letters(t) for t in text]
    text = [remove_single_characters(t) for t in text]
    return labels, text

def word_pos_neg_freq_dict(x, y):

    label_word = list(itertools.chain.from_iterable([[(word, y[i]) for word in line.split()] for i, line in enumerate(x)]))
    neg_words = dict()
    pos_words = dict()
    for i, line in enumerate(x):
        if y[i] == 1:
            for word in line.split():
                neg_words[word] = neg_words.get(word, 0) + 1
        elif y[i] == 2:
            for word in line.split():
                pos_words[word] = pos_words.get(word, 0) + 1


    return pos_words, neg_words

import operator

def most_frequent_words(text=None, n=0):
    if text == None:
        return

    text = ' '.join(text)
    words = text.split()

    counter = Counter(words)
    counter = sorted(counter.items(), key=operator.itemgetter(1),reverse=True)
    counter = dict(counter[:n])
    return counter


def most_frequent_chars(text=None, n=0):
    if text == None:
        return

    text = ' '.join(text)
    words = text.split()
    text = ''.join(words)
    letters = text.split()

    counter = Counter(letters)
    counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

    counter = dict(counter[:n])
    return counter

####################### yelp plots ################################
def plt_most_freq_words(x, y, output_num=15,outfile='plt_most_freq_words.png'):
    print("---- step 1 -----")
    counter = most_frequent_words(x, output_num)
    print(counter)


    sns.set()
    print("---- step 2 -----")
    pos_dict, neg_dict = word_pos_neg_freq_dict(x, y)
    # counter = {x for x in pos_dict}
    pos_dict = [pos_dict[k] for k in counter.keys()]
    neg_dict = [neg_dict[k] for k in counter.keys()]

    print("---- step 3 -----")
    pos_neg_tuples = list(zip(pos_dict,neg_dict))

    print("---- step 4 -----")
    df = pd.DataFrame(pos_neg_tuples, columns=['negative', 'positive'],index=counter.keys())
    df['negative'] = -df['negative']
    fig = df.plot(kind='bar', stacked=True);
    plt.show()
    fig.get_figure().savefig(outfile)

############################# ag news plots #####################
def word_class_freq_dict(x, y):

    label_word = list(itertools.chain.from_iterable([[(word, y[i]) for word in line.split()] for i, line in enumerate(x)]))

    class_1 = Counter([item[0] for item in label_word if item[1] == 1])
    class_2 = Counter([item[0] for item in label_word if item[1] == 2])
    class_3 = Counter([item[0] for item in label_word if item[1] == 3])
    class_4 = Counter([item[0] for item in label_word if item[1] == 4])

    return [class_1, class_2, class_3, class_4]

def bar_plot_vertical(x, y):
    df = pd.read_csv(args.get('Data', 'data_source') + args.get('Data', 'dataset') + '/mpg_ggplot2.csv')
    x_var = 'displ'
    groupby_var = 'class'
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]

    # Draw
    plt.figure(figsize=(16, 9), dpi=80)
    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend({group: col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 25)
    plt.xticks(ticks=bins[::3], labels=[round(b, 1) for b in bins[::3]])
    plt.show()

def bar_plot_horizontal(x, y):
    category_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    classes_freq = word_class_freq_dict(x, y)

    word_freq_tuples = most_frequent_words(x,15)

    labels = [w[0] for w in word_freq_tuples]
    results = {word : [c.get(word,0) for c in classes_freq] for word in labels}
    counts = Counter(y)
    sorted(counts.items())

    def survey(results, category_names, labels):
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))

        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
            xcenters = starts + widths / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')
        plt.xlabel('occurences')
        plt.ylabel("word")
        return fig, ax

    fig, _ = survey(results, category_names, labels)
    fig.savefig('plots/ag_news_bar_plot_horiz.png', format='png', dpi=300)
    plt.show()




def treemap_plot(x,y):
    word_freq_tuples = most_frequent_words(x,15)
    word_freq_dict = dict(word_freq_tuples)

    labels = list(word_freq_dict.keys())
    data = list(word_freq_dict.values())

    ####### uncomment the 4 lines below to change the tree plot to classes instead of words
    # counts = Counter(y)
    # sorted(counts.items())
    # labels = 'World', 'Sports', 'Business', 'Sci/Tech'
    # data =list(counts.values())

    colors = [plt.cm.Spectral(i / float(len(labels))) for i in range(len(labels))]
    fig = plt.figure(figsize=(12, 8), dpi=80)
    squarify.plot(sizes=data, label=labels, color=colors, alpha=.8, text_kwargs={'fontsize':20})

    plt.title('Treemap of AG News Class')
    plt.axis('off')
    plt.show()
    fig.savefig('plots/ag_news_treemap.png', format='png', dpi=300)

########################## generic plots ################################
def classes_pie(y, outfile='classes_pie.png', dataset=None):
    fig = plt.figure(figsize=(6, 5))
    if dataset == None:
        return
    elif dataset == 'yelp':
        labels =  'negative', 'positive'
    elif dataset == 'ag_news':
        labels = 'World', 'Sports', 'Business', 'Sci/Tech'

    counts = Counter(y)
    sorted(counts.items())

    # Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
    wedges, texts, autotexts = plt.pie(list(counts.values()), labels=labels, autopct='%1.0f%%', startangle=90,
                                       pctdistance=1.2, labeldistance=1.3, explode=[0.05]*len(labels))

    plt.axis('equal')
    if 'yelp' in outfile:
        plt.xlabel('positive to negative reviews ratio')
    elif 'ag_news' in outfile:
        plt.xlabel('reviews ratio in each class')
    plt.setp(autotexts, size=8, weight="bold")
    plt.show()
    fig.savefig(outfile, format='png', dpi=300)


def show(x, y, dataset=None):
    classes_pie(y,outfile='plots/'+dataset+'_classes_pie'+'.png' ,dataset = dataset)
    if dataset == 'yelp':
        plt_most_freq_words(x, y, outfile='plots/'+dataset+'_bar_most_freq_words.png')
    elif dataset == 'ag_news':
        treemap_plot(x,y)
        bar_plot_horizontal(x, y)


if __name__ == '__main__':
    args = configparser.ConfigParser()
    args.read('argsConfig.ini')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    y, x = load_data(args)
    show(x, y, args.get('Data', 'dataset'))