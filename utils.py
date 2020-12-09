import math
import json
import re
import numpy as np
from sklearn import metrics
import nlpaug.augmenter.word as naw

# text-preprocessing
import nltk
from nltk.corpus import stopwords


# nltk.download('averaged_perceptron_tagger')

def lower(text):
    return text.lower()


def remove_hashtags(text):
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_urls(text):
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return clean_text


def remove_double_new_line(text):
    clean_text = re.sub(r'[\r\n]{2}','\r\n', text)
    return clean_text

# Augment text with WordNet synonym semantic similarity
def augment_text_wordnet(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    clean_text = aug.augment(text)
    return text

def remove_numbers(text):
    return re.sub(r'\d+','', text)

def remove_single_letters(text):
    return re.sub(r'\s[a-z]\s',' ', text)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = nltk.word_tokenize(text)
    text = ' '.join([w for w in text if not lower(w) in stop_words])

    return text

def remove_single_characters(text):
    return re.sub(r'\b[a-zA-Z]\b', '', text)

def remove_special_characters(text):
    clean_text = re.sub(r'[!\n\\\/.,#\?\-\_\(\)\[\]\{\}§$%&\*\";:\']', ' ', text, flags=re.MULTILINE)
    return clean_text

preprocessing_setps = {
    'remove_hashtags': remove_hashtags,
    'remove_urls': remove_urls,
    'remove_user_mentions': remove_user_mentions,
    'lower': lower,
    'double_new_line': remove_double_new_line,
    'wordnet_augment_text': augment_text_wordnet,
}



def process_text(steps, text):
    #if steps is not None:
    if steps[0] != 'None':
        for step in steps:
            text = preprocessing_setps[step](text)
    return text

# metrics // model evaluations


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'f1_weighted' in list_metrics:
        output['f1_weighted'] = metrics.f1_score(y_true, y_pred, average='weighted')
    if 'f1_micro' in list_metrics:
        output['f1_micro'] = metrics.f1_score(y_true, y_pred, average='micro')
    if 'f1_macro' in list_metrics:
        output['f1_macro'] = metrics.f1_score(y_true, y_pred, average='macro')

    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_log(log_file, args, labels):
    with open(log_file, 'a+') as f:
        f.write(f'Model-Log:\n')
        f.write('=' * 50)
        f.write(f'\nModel Type: CharacterLevelCNN \n')
        f.write(f'Feature size: {args.getint("Model", "feature_num")} \n')
        f.write('=' * 50)

        f.write(f'\nData-Log:\n')
        f.write('=' * 50)
        f.write(f'\nDataset: {args.get("Data", "dataset")} \n')
        f.write(f'Encoding: {args.get("Data", "encoding")} \n')
        f.write(f'Chunk size: {args.getint("Data", "chunk_size")} \n')
        f.write(f'Max CSV Rows: {args.getint("Data", "max_csv_rows")} \n')
        f.write(f'CSV Separator: {args.get("Data", "csv_sep")} \n')
        f.write(f'CSV Columns: {args.get("Data", "usecols")} \n')
        f.write(f'Alphabet: {args.get("DataSet", "alphabet")} \n')
        f.write(f'Character Number: {args.getint("DataSet", "char_num")} \n')
        f.write(f'l0: {args.getint("DataSet", "l0")} \n')
        f.write(f'Classes: {labels}\n')
        if args.getboolean('Data', 'preprocess_data'):
            f.write(f'Preprocess Text Steps: {args.get("Data", "steps")} \n')

        f.write('=' * 50)
        f.write(f'\nTrain-Log:\n')
        f.write('=' * 50)
        f.write(f'\nMax Epochs: {args.getint("Train", "epochs")} \n')
        f.write(f'Batch Size: {args.getint("Train", "batch_size")} \n')
        f.write(f'Train Size: {args.getfloat("Train", "train_size")} \n')
        f.write(f'Dev Size: {args.getfloat("Train", "dev_size")} \n')
        f.write(f'Max-Norm Size: {args.getint("Train", "max_norm")} \n')
        f.write(f'Optimizer: {args.get("Train", "optimizer")} \n')
        f.write(f'Scheduler: {args.get("Train", "scheduler")} \n')
        f.write(f'Learning Rate: {args.getfloat("Train", "lr")} \n')
        if args.getboolean('Train', 'early_stopping'):
            f.write(f'Patience: {args.getint("Train", "patience")} \n')
        f.write(f'Learning Rate: {args.getboolean("Train", "continue_from_checkpoint")} \n')
        if args.getboolean('Train', 'continue_from_checkpoint'):
            f.write(f'Continue Train from this Model: {args.get("Log", "continue_from_model_checkpoint")} \n')

        f.write('=' * 50)
        f.write(f'\nLog-Information:\n')
        f.write('=' * 50)
        f.write(f'\nFlush History: {args.getboolean("Log", "flush_history")} \n')
        f.write(f'Log Path: {args.get("Log", "log_path")} \n')
        f.write(f'Output Path: {args.get("Log", "output")} \n')
        f.write(f'Model Name: {args.get("Log", "model_name")} \n')
        f.write(f'Log F1: {args.getboolean("Log", "log_f1")} \n')
        f.write('=' * 50)
        f.write('\n')


if __name__ == '__main__':
    text = 'what should a man do to deserve this harsh comeback'
    text = remove_double_new_line(text)
    #print("before: ", text)
    print("after: ", text)

    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)