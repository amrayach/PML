import os
import argparse
import datetime
import sys
import errno
from data_loader import load_data, MyDataset
from model import CharacterLevelCNN
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from metric import print_f_score
import configparser


if __name__ == '__main__':

    args = configparser.ConfigParser()
    args.read('argsConfig.ini')

    # load testing data
    print("\nLoading testing data...")
    texts, labels, number_of_classes, sample_weights = load_data(args, 'test')

    test_dataset = MyDataset(texts, labels, args)
    print("Transferring testing data to iterator...")
    testing_params = {"batch_size": args.getint('Train', 'batch_size'),
                       "shuffle": False,
                       "num_workers": args.getint('Train', 'workers'),
                       "drop_last": True}
    test_loader = DataLoader(test_dataset, **testing_params)

    _, num_class_test = test_dataset.get_class_weight()
    print('\nNumber of testing samples: '+str(test_dataset.__len__()))
    for i, c in enumerate(num_class_test):
        print("\tLabel {:d}:".format(i).ljust(15)+"{:d}".format(c).rjust(8))

    #args.num_features = len(test_dataset.alphabet)
    #model = CharCNN(args)
    model = CharacterLevelCNN(number_of_classes, args)

    print("=> loading weights from '{}'".format(args.model_path))
    assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # using GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    print('\nTesting...')
    for i_batch, (data) in enumerate(test_loader):
        inputs, target = data
        target.sub_(1)
        size+=len(target)
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        inputs = Variable(inputs, volatile=True)
        target = Variable(target)
        logit = model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        accumulated_loss += F.nll_loss(logit, target, size_average=False).data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        predicates_all+=predicates.cpu().numpy().tolist()
        target_all+=target.data.cpu().numpy().tolist()
        
    avg_loss = accumulated_loss/size
    accuracy = 100.0 * corrects/size
    print('\rEvaluation - loss: {:.6f}  acc: {:.3f}%({}/{}) '.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    print_f_score(predicates_all, target_all)