from predictExplain import ModelsDeploy
import numpy as np
import pandas as pd
import streamlit as st
from spacy.tokens import Doc, Span
from spacy_streamlit import visualize_ner

def to_rgba(hex, val):
    val = int(val) 
    val = abs(val)
    val = 255 if val > 255 else val
    hex = hex + "{:02x}".format(val)
    return hex


deploy = ModelsDeploy()

st.set_page_config(page_title='Character-Level CNN Predict & Explain:', page_icon='random', layout='wide', initial_sidebar_state='collapsed')
st.title('Character-Level CNN Predict & Explain:')
st.text('Select Model:')
model_in = st.selectbox('Models:', ['Yelp-Review-Polarity', 'AG-News-Category-Classifier'], index=0)

# widget slider for choosing number of top n important words in making decision
slider_range=50
slider_start_value=5
top_n_words = 5

sentence = st.text_input('Enter Sentence:', value="Like any Barnes & Noble, it has a nice comfy cafe, and a large selection of books.  The staff is very friendly and helpful.  They stock a decent selection, and the prices are pretty reasonable.  Obviously it's hard for them to compete with Amazon.  However since all the small shop bookstores are gone, it's nice to walk into one every once in a while.")

col_library = {'positive': '#FF0000', 'negative': '#0000FF'}

if model_in == 'Yelp-Review-Polarity':
    prediction, probs, heatmap = deploy.explain(sentence, model='yelp')
    st.text('--------------------------------')
    if prediction == 0:
        st.text("The Prediction is Negative")
    else:
        st.text("The Prediction is Positive")
    st.text('--------------------------------')
    st.text('Class Probabilities:')

    dataframe = pd.DataFrame(
        np.array([probs]),
        columns=('Negative', 'Positive'))
    st.dataframe(dataframe.style.highlight_max(axis=0))


else:
    prediction, probs, heatmap = deploy.explain(sentence, model='ag_news')

    st.text('--------------------------------')
    if prediction == 0:
        st.text("The Prediction is World")
    elif prediction == 1:
        st.text("The Prediction is Sports")

    elif prediction == 2:
        st.text("The Prediction is Business")

    else:
        st.text("The Prediction is Sci/Tech")
    st.text('--------------------------------')
    st.text('Class Probabilities:')

    dataframe = pd.DataFrame(
        np.array([probs]),
        columns=('World', 'Sports', 'Business', 'Sci/Tech'))
    st.dataframe(dataframe.style.highlight_max(axis=0))


words = [i[0] for i in heatmap]
vals = [i[1] for i in heatmap]
spaces = [True] * (len(words) - 1)
spaces.append(False)
doc = Doc(deploy.nlp.vocab, words=words, spaces=spaces)

ents = []
tags = []



for j, i in enumerate(doc):
    new_ent = Span(doc, j, j + 1, label=str(j))
    ents.append(new_ent)
    tags.append(str(j))

doc.ents = []
doc.ents = ents

col_library = {'positive': '#FF0000', 'negative': '#0000FF'}
colors = [to_rgba(col_library['positive'], x) if x >= 0 else to_rgba(col_library['negative'], x) for x in vals]

tags = tuple(list(map(lambda x: ''.join(list(map(lambda y: y.upper(), x.split('_')))), tags)))
col_dict = {}
for i in range(len(tags)):
    col_dict[tags[i]] = colors[i]

for i in range(len(heatmap)):
    heatmap[i] += (str(i),)

heatmap_neg = list(sorted(list(filter(lambda x: x[1] < 0, heatmap)), key=lambda x: x[1]))
heatmap_pos = list(sorted(list(filter(lambda x: x[1] >= 0, heatmap)), key=lambda x: x[1], reverse=True))

visualize_ner(doc, labels=tags, colors=col_dict, show_table=True, title='Character2Word Attention Heatmap:')

