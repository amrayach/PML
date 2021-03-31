import configparser
from model import CharacterLevelCNN
import torch
from innvestigator import InnvestigateModel
import spacy
import utils

args = configparser.ConfigParser()
args.read('argsConfig.ini')


class ModelsDeploy(object):
    def __init__(self):
        self.ag_news_model = CharacterLevelCNN(4, args)
        self.ag_news_model_checkpoint = torch.load('AgNewsModel1.pt', map_location=torch.device('cpu'))
        self.ag_news_model.load_state_dict(self.ag_news_model_checkpoint['state_dict'])
        self.ag_news_lrp = InnvestigateModel(self.ag_news_model, lrp_exponent=1, method="e-rule", beta=.5)

        self.yelp_model = CharacterLevelCNN(2, args)
        self.yelp_model_checkpoint = torch.load('YelpModel.pt', map_location=torch.device('cpu'))
        self.yelp_model.load_state_dict(self.yelp_model_checkpoint['state_dict'])
        self.yelp_lrp = InnvestigateModel(self.yelp_model, lrp_exponent=1, method="e-rule", beta=.5)
        self.alphabet = args.get('DataSet', 'alphabet')
        self.l0 = args.getint('DataSet', 'l0')
        self.nlp = spacy.load("en_core_web_sm-2.3.1")

    def oneHotEncode(self, sentence):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = sentence[:self.l0]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def generate_word_rel_vals(self, text, heatmap):

        word_rel_vals = []
        word = ''
        val = 0
        for i in range(len(text)):
            if text[i] == ' ':
                # print(' ')
                # try:
                # word_rel_vals.append((word, val / len(word)))
                #    word_rel_vals.append((word, val))
                # word_rel_vals[word] = val / len(word)
                # except:
                #    word_rel_vals.append((word, val))
                # word_rel_vals[word] = val
                word_rel_vals.append((word, val))

                word = ""
                val = 0
            else:
                word += text[i]
                val += torch.sum(heatmap[:, i]).item()
                # print(text[i], torch.sum(heatmap[:, i]).item())

        word_rel_vals.append((word, val))

        # try:
        # word_rel_vals.append((word, val / len(word)))
        #    word_rel_vals.append((word, val))

        # word_rel_vals[word] = val / len(word)
        # except:
        #    word_rel_vals.append((word, val))
        # word_rel_vals[word] = val

        return word_rel_vals

    def predict_probs(self, sentence, model='yelp'):
        input_tensor = self.oneHotEncode(sentence)
        input_tensor = torch.unsqueeze(input_tensor, 0)

        if model == 'yelp':
            with torch.no_grad():
                predictions = self.yelp_model(input_tensor)
        else:
            with torch.no_grad():
                predictions = self.ag_news_model(input_tensor)

        pred = torch.max(predictions, 1)[1].cpu().numpy().tolist()[0]
        probs = torch.exp(predictions) * 100
        probs = probs.cpu().numpy().tolist()[0]

        return pred, probs

    def explain(self, sentence, model='yelp'):

        sent1 = sentence
        sentence = utils.remove_stop_words(sentence)
        sentence = utils.remove_single_characters(sentence)
        sentence = ' '.join(utils.remove_special_characters(sentence).split())

        input_tensor = self.oneHotEncode(sentence)
        input_tensor = torch.unsqueeze(input_tensor, 0)

        if model == 'yelp':
            predictions, heatmap = self.yelp_lrp.innvestigate(in_tensor=input_tensor)
        else:
            predictions, heatmap = self.ag_news_lrp.innvestigate(in_tensor=input_tensor)

        pred = torch.max(predictions, 1)[1].cpu().numpy().tolist()[0]
        probs = torch.exp(predictions) * 100
        probs = probs.cpu().numpy().tolist()[0]

        word_rels_vals = self.generate_word_rel_vals(sentence, heatmap[0])
        word_rels_vals = list(filter(lambda x: x[0] != '', word_rels_vals))

        return pred, probs, word_rels_vals


def main():
    obj = ModelsDeploy()
    a, b, c = obj.explain(
        "Like any Barnes & Noble, it has a nice comfy cafe, and a large selection of books.  The staff is very friendly and helpful.  They stock a decent selection, and the prices are pretty reasonable.  Obviously it's hard for them to compete with Amazon.  However since all the small shop bookstores are gone, it's nice to walk into one every once in a while.")



if __name__ == '__main__':
    main()
