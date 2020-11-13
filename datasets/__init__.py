# Expose base classses

from .base import DataLoader
# A simple Dataset object to wrap a List of Sentence
from .base import SentenceDataset

# A Dataset taking string as input and returning Sentence during iteration
from .base import StringDataset

# Reads Mongo collections. Each collection should contain one document/text per item.
from .base import MongoDataset



# Expose all document classification datasets

# A classification corpus from FastText-formatted text files.
from .document_classification import ClassificationCorpus

# Dataset for classification instantiated from a single FastText-formatted file.
from .document_classification import ClassificationDataset

# Classification corpus instantiated from CSV data files.
from .document_classification import CSVClassificationCorpus

# Dataset for text classification from CSV column formatted data.
from .document_classification import CSVClassificationDataset

"""
A very large corpus of Amazon reviews with positivity ratings. Corpus is downloaded from and documented at
https://nijianmo.github.io/amazon/index.html. We download the 5-core subset which is still tens of millions of
reviews.
"""
from .document_classification import AMAZON_REVIEWS

"""
Corpus of IMDB movie reviews labeled by sentiment (POSITIVE, NEGATIVE). Downloaded from and documented at
http://ai.stanford.edu/~amaas/data/sentiment/.
"""
from .document_classification import IMDB

"""
20 newsgroups corpus available at "http://qwone.com/~jason/20Newsgroups", classifying
news items into one of 20 categories. Each data point is a full news article so documents may be very long.
"""

"""
20 newsgroups corpus available at "http://qwone.com/~jason/20Newsgroups", classifying
news items into one of 20 categories. Each data point is a full news article so documents may be very long.
"""
from .document_classification import NEWSGROUPS


"""
Twitter sentiment corpus downloaded from and documented at http://help.sentiment140.com/for-students. Two sentiments
in train data (POSITIVE, NEGATIVE) and three sentiments in test data (POSITIVE, NEGATIVE, NEUTRAL).
"""
from .document_classification import SENTIMENT_140

"""
The customer reviews dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
NEGATIVE or POSITIVE sentiment.
"""
# from .document_classification import SENTEVAL_CR

"""
The movie reviews dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
NEGATIVE or POSITIVE sentiment.
"""
# from .document_classification import SENTEVAL_MR


"""
The subjectivity dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
SUBJECTIVE or OBJECTIVE sentiment.
"""
# from .document_classification import SENTEVAL_SUBJ


"""
The opinion-polarity dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
NEGATIVE or POSITIVE polarity.
"""
from .document_classification import SENTEVAL_MPQA

"""
The Stanford sentiment treebank dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified
into NEGATIVE or POSITIVE sentiment.
"""
# from .document_classification import SENTEVAL_SST_BINARY


"""
The Stanford sentiment treebank dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified
into 5 sentiment classes.
"""
# from .document_classification import SENTEVAL_SST_GRANULAR

"""
The TREC Question Classification Corpus, classifying questions into 50 fine-grained answer types.
"""
# from .document_classification import TREC_50

"""
The TREC Question Classification Corpus, classifying questions into 6 coarse-grained answer types
(DESC, HUM, LOC, ENTY, NUM, ABBR).
"""
# from .document_classification import TREC_6

"""
The Communicative Functions Classification Corpus. 
Classifying sentences from scientific papers into 39 communicative functions. 
"""
# from .document_classification import COMMUNICATIVE_FUNCTIONS

"""
WASSA-2017 anger emotion-intensity dataset downloaded from and documented at
https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
"""
# from .document_classification import WASSA_ANGER

"""
WASSA-2017 fear emotion-intensity dataset downloaded from and documented at
 https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
"""
# from .document_classification import WASSA_FEAR


"""
WASSA-2017 joy emotion-intensity dataset downloaded from and documented at
 https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
"""
# from .document_classification import WASSA_JOY


"""
WASSA-2017 sadness emotion-intensity dataset downloaded from and documented at
 https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
"""
# from .document_classification import WASSA_SADNESS


"""
GoEmotions dataset containing 58k Reddit comments labeled with 27 emotion categories, see. https://github.com/google-research/google-research/tree/master/goemotions
"""
from .document_classification import GO_EMOTIONS


# Expose all sequence labeling datasets
# Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.
from .sequence_labeling import ColumnCorpus

# Instantiates a column dataset (typically used for sequence labeling or word-level prediction).
from .sequence_labeling import ColumnDataset


"""
Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the eng.testa, .testb, .train
files in a folder called 'conll_03'. Then set the base_path parameter in the constructor to the path to the
parent directory where the conll_03 folder resides.
"""
from .sequence_labeling import CONLL_03
from .sequence_labeling import CONLL_03_GERMAN


"""
Initialize the wikigold corpus. The first time you call this constructor it will automatically
download the dataset.
"""
from .sequence_labeling import WIKIGOLD_NER
from .sequence_labeling import WIKINER_ENGLISH
from .sequence_labeling import WIKINER_GERMAN


"""
Initialize a dataset called twitter_ner which can be found on the following page:
https://raw.githubusercontent.com/aritter/twitter_nlp/master/data/annotated/ner.txt.
The first time you call this constructor it will automatically
download the dataset.
"""
from .sequence_labeling import TWITTER_NER








# Expose all treebanks
# If needed


# Instantiates a Corpus for text classification from CSV column formatted data
from .text_text import ParallelTextCorpus

# Instantiates a Parallel Corpus from OPUS (http://opus.nlpl.eu/)
from .text_text import ParallelTextDataset

#
from .text_text import OpusParallelCorpus