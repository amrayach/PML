import os
from data_loader import load_data, MyDataset
from model import CharacterLevelCNN
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report, f1_score, plot_confusion_matrix
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import nn


import configparser
import utils

if __name__ == '__main__':

    args = configparser.ConfigParser()
    args.read('argsConfig.ini')
    log_dir = args.get('Test', 'model_log_dir')
    writer = SummaryWriter(log_dir)
    log_file = log_dir + 'log.txt'

    with open(log_file, 'a') as f:
        f.write('=' * 50)
        f.write('Testing')
        f.write('=' * 50)

    # load testing data
    print("\nLoading testing data...")
    texts, labels, number_of_classes, sample_weights = load_data(args, 'test')

    test_dataset = MyDataset(texts, labels, args)
    print("Transferring testing data to iterator...")
    testing_params = {"batch_size": args.getint('Train', 'batch_size'),
                       "shuffle": False,
                       "num_workers": args.getint('Train', 'workers'),
                       "drop_last": True}
    test_generator = DataLoader(test_dataset, **testing_params)

    print('\nNumber of testing samples: '+str(test_dataset.__len__()))
    with open(log_file, 'a') as f:
        f.write('\nNumber of testing samples: '+str(test_dataset.__len__())+'\n')

    model = CharacterLevelCNN(number_of_classes, args)

    print("=> loading weights from '{}'".format(args.get('Test', 'model_to_test')))
    #assert os.path.isfile(args.get('Test', 'model_to_test')), "=> no checkpoint found at '{}'".format(args.get('Test', 'model_to_test'))
    with open(log_file, 'a') as f:
        f.write("\n=> loading weights from '{}'".format(args.get('Test', 'model_to_test')))
    checkpoint = torch.load(args.get('Test', 'model_to_test'))
    model.load_state_dict(checkpoint['state_dict'])

    # using GPU
    if args.getboolean('Device', 'enable_gpu'):
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(test_generator)
    if args.get('Train', 'criterion') == 'nllloss':
        criterion = nn.NLLLoss()

    y_true = []
    y_pred = []

    for n_iter, batch in tqdm(enumerate(test_generator), total=num_iter_per_epoch):
        features, labels = batch
        labels.sub_(1)
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            predictions = model(features)
        loss = criterion(predictions, labels)

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(predictions, 1)[1].cpu().numpy().tolist()

        validation_metrics = utils.get_evaluation(labels.cpu().numpy(),
                                                  predictions.cpu().detach().numpy(),
                                                  list_metrics=["accuracy", "f1_weighted", "f1_micro", "f1_macro"])
        accuracy = validation_metrics['accuracy']
        f1_weighted = validation_metrics['f1_weighted']
        f1_micro = validation_metrics['f1_micro']
        f1_macro = validation_metrics["f1_macro"]

        losses.update(loss.data, features.size(0))
        accuracies.update(validation_metrics["accuracy"], features.size(0))

        writer.add_scalar('Test/Loss',
                          loss.item(),
                          n_iter)

        writer.add_scalar('Test/Accuracy',
                          accuracy,
                          n_iter)

        writer.add_scalar('Test/f1-weighted',
                          f1_weighted,
                          n_iter)

        writer.add_scalar('Test/f1-micro',
                          f1_micro,
                          n_iter)

        writer.add_scalar('Test/f1-macro',
                          f1_macro,
                          n_iter)

    f1_test_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_test_micro = f1_score(y_true, y_pred, average='micro')
    f1_test_macro = f1_score(y_true, y_pred, average='macro')

    report = classification_report(y_true, y_pred)
    #cnf_matrix_plot = plot_confusion_matrix()

    print(report)

    with open(log_file, 'a') as f:
        f.write(f'Average loss: {losses.avg.item()} \n')
        f.write(f'Average accuracy: {accuracies.avg.item()} \n')
        f.write(f'F1 Weighted score {f1_test_weighted} \n\n')
        f.write(f'F1 Micro score {f1_test_micro} \n\n')
        f.write(f'F1 Macro score {f1_test_macro} \n\n')
        f.write(report)
        f.write('=' * 50)
        f.write('\n')



# This was the continue from checkpoint training part but
        # is not compatible with the cross validation part
        # so it is temporary removed but could be adjusted in the future

    '''
    if os.path.isfile(args.get('Log', 'continue_from_model_checkpoint')):
        print("=> loading checkpoint from '{}'".format(args.get('Log', 'continue_from_model_checkpoint')))
        checkpoint = torch.load(args.get('Log', 'continue_from_model_checkpoint'))
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_f1 = checkpoint.get('best_f1', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_iter = 0
        start_epoch = 0
        best_f1 = 0
        best_epoch = 0
    '''