import configparser
import os
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from metric import print_f_score
from torch import nn

from torch.utils.data import DataLoader, WeightedRandomSampler

from data_loader import load_data, MyDataset
from model import CharacterLevelCNN
import utils
from sklearn.metrics import classification_report, f1_score


def train(model, training_generator, optimizer, criterion, epoch, writer, log_file, scheduler, class_names, args, print_every=10):
    model.train()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(training_generator)

    progress_bar = tqdm(enumerate(training_generator),
                        total=num_iter_per_epoch)

    y_true = []
    y_pred = []

    for iter, batch in progress_bar:
        features, labels = batch
        #tmp = torch.isnan(features)
        #tmp1 = torch.isinf(features)

        #print(1 in tmp)
        #print(1 in tmp1)
        #print(features)
        labels.sub_(1)
        #print(labels)

        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        predictions = model(features)

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(predictions, 1)[1].cpu().numpy().tolist()

        loss = criterion(predictions, labels)

        loss.backward()

        if args.get('Train', 'scheduler') == 'clr':
            scheduler.step()

        optimizer.step()
        training_metrics = utils.get_evaluation(labels.cpu().numpy(),
                                                predictions.cpu().detach().numpy(),
                                                list_metrics=["accuracy", "f1"])

        losses.update(loss.data, features.size(0))
        accuracies.update(training_metrics["accuracy"], features.size(0))

        f1 = training_metrics['f1']

        writer.add_scalar('Train/Loss',
                          loss.item(),
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Train/Accuracy',
                          training_metrics['accuracy'],
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Train/f1',
                          f1,
                          epoch * num_iter_per_epoch + iter)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if (iter % print_every == 0) and (iter > 0):
            print("[Training - Epoch: {}], LR: {} , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                lr,
                iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg
            ))

            if bool(args.getboolean('Log', 'log_f1')):
                intermediate_report = classification_report(
                    y_true, y_pred, output_dict=True)

                print(intermediate_report)
                #f1_by_class = 'F1 Scores by class: '
                #for class_name in class_names:
                #    f1_by_class += f"{class_name} : {np.round(intermediate_report[class_name]['f1-score'], 4)} |"
                #print(f1_by_class)

    f1_train = f1_score(y_true, y_pred, average='weighted')

    writer.add_scalar('Train/loss/epoch', losses.avg, epoch + iter)
    writer.add_scalar('Train/acc/epoch', accuracies.avg, epoch + iter)
    writer.add_scalar('Train/f1/epoch', f1_train, epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, 'a') as f:
        f.write(f'Training on Epoch {epoch} \n')
        f.write(f'Average loss: {losses.avg.item()} \n')
        f.write(f'Average accuracy: {accuracies.avg.item()} \n')
        f.write(f'F1 score: {f1_train} \n\n')
        f.write(report)
        f.write('*' * 25)
        f.write('\n')

    return losses.avg.item(), accuracies.avg.item(), f1_train


def evaluate(model, validation_generator, criterion, epoch, writer, log_file, print_every=25):
    model.eval()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(validation_generator)

    y_true = []
    y_pred = []

    for iter, batch in tqdm(enumerate(validation_generator), total=num_iter_per_epoch):
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
                                                  list_metrics=["accuracy", "f1"])
        accuracy = validation_metrics['accuracy']
        f1 = validation_metrics['f1']

        losses.update(loss.data, features.size(0))
        accuracies.update(validation_metrics["accuracy"], features.size(0))

        writer.add_scalar('Test/Loss',
                          loss.item(),
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Test/Accuracy',
                          accuracy,
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Test/f1',
                          f1,
                          epoch * num_iter_per_epoch + iter)

        if (iter % print_every == 0) and (iter > 0):
            print("[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg
            ))

    f1_test = f1_score(y_true, y_pred, average='weighted')

    writer.add_scalar('Test/loss/epoch', losses.avg, epoch + iter)
    writer.add_scalar('Test/acc/epoch', accuracies.avg, epoch + iter)
    writer.add_scalar('Test/f1/epoch', f1_test, epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, 'a') as f:
        f.write(f'Validation on Epoch {epoch} \n')
        f.write(f'Average loss: {losses.avg.item()} \n')
        f.write(f'Average accuracy: {accuracies.avg.item()} \n')
        f.write(f'F1 score {f1_test} \n\n')
        f.write(report)
        f.write('=' * 50)
        f.write('\n')

    return losses.avg.item(), accuracies.avg.item(), f1_test



def main():
    args = configparser.ConfigParser()
    args.read('argsConfig.ini')

    if args.getboolean('Log', 'flush_history') == 1:
        # todo check if log dir exists else create new one
        objects = os.listdir(args.get('Log', 'log_path'))
        for f in objects:
            if os.path.isdir(args.get('Log', 'log_path') + f):
                shutil.rmtree(args.get('Log', 'log_path') + f)

    now = datetime.now()
    logdir = args.get('Log', 'log_path') + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)
    log_file = logdir + 'log.txt'
    writer = SummaryWriter(logdir)

    texts, labels, number_of_classes, sample_weights = load_data(args, 'train')

    class_names = sorted(list(set(labels)))
    class_names = [str(class_name-1) for class_name in class_names]

    train_texts, X_dev, train_labels, y_dev_labels, train_sample_weights, _ = train_test_split(texts,
                                                                                                   labels,
                                                                                                   sample_weights,
                                                                                                   train_size=args.getfloat(
                                                                                                       'Train',
                                                                                                       'train_size'),
                                                                                                   test_size=args.getfloat(
                                                                                                       'Train',
                                                                                                       'dev_size'),
                                                                                                   random_state=42,
                                                                                                   stratify=labels)


    training_set = MyDataset(train_texts, train_labels, args)
    validation_set = MyDataset(X_dev, y_dev_labels, args)

    training_params = {"batch_size": args.getint('Train', 'batch_size'),
                       "shuffle": True,
                       "num_workers": args.getint('Train', 'workers'),
                       "drop_last": True}

    validation_params = {"batch_size": args.getint('Train', 'batch_size'),
                         "shuffle": False,
                         "num_workers": args.getint('Train', 'workers'),
                         "drop_last": True}


    if args.getboolean('Train', 'use_sampler'):
        train_sample_weights = torch.from_numpy(train_sample_weights)
        sampler = WeightedRandomSampler(train_sample_weights.type(
            'torch.DoubleTensor'), len(train_sample_weights))
        training_params['sampler'] = sampler
        training_params['shuffle'] = False

    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    model = CharacterLevelCNN(number_of_classes, args)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.NLLLoss()
    #criterion = nn.BCELoss()

    # optimization scheme
    if args.get('Train', 'optimizer') == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.getfloat('Train', 'lr'))
    elif args.get('Train', 'optimizer') == 'SGD':
        if args.get('Train', 'scheduler') == 'clr':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=1, momentum=0.9, weight_decay=0.00001
            )
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.getfloat('Train', 'lr'), momentum=0.9)
    elif args.get('Train', 'optimizer') == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=args.getfloat('Train', 'lr'))

    best_f1 = 0
    best_epoch = 0


    if args.get('Train', 'scheduler') == 'clr':
        stepsize = int(args.getint('Train', 'clr_step_size') * len(training_generator))
        clr = utils.cyclical_lr(stepsize, args.getfloat('Train', 'clr_min_lr'), args.getfloat('Train', 'clr_max_lr'))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = None



    for epoch in range(args.getint('Train', 'epochs')):
        training_loss, training_accuracy, train_f1 = train(model,
                                                           training_generator,
                                                           optimizer,
                                                           criterion,
                                                           epoch,
                                                           writer,
                                                           log_file,
                                                           scheduler,
                                                           class_names,
                                                           args)

        validation_loss, validation_accuracy, validation_f1 = evaluate(model,
                                                                       validation_generator,
                                                                       criterion,
                                                                       epoch,
                                                                       writer,
                                                                       log_file)

        print('\n[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}'.
              format(epoch + 1, args.getint('Train', 'epochs'), training_loss, training_accuracy, validation_loss, validation_accuracy))
        print("=" * 50)

        # learning rate scheduling

        if args.get('Train', 'scheduler') == 'step':
            if args.get('Train', 'optimizer') == 'SGD' and ((epoch + 1) % 3 == 0) and epoch > 0:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                current_lr /= 2
                print('Decreasing learning rate to {0}'.format(current_lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

        # model checkpoint

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch

            if args.getboolean('Log', 'checkpoint'):
                torch.save(model.state_dict(),
                           args.get('Log', 'output') + 'model_{}_epoch_{}_l0_{}_lr_{}_loss_{}_acc_{}_f1_{}.pth'.format(
                               args.get('Log', 'model_name'),
                               epoch,
                               args.getint('DataSet', 'l0'),
                               optimizer.state_dict()[
                                   'param_groups'][0]['lr'],
                               round(
                                   validation_loss, 4),
                               round(
                                   validation_accuracy, 4),
                               round(
                                   validation_f1, 4)
                               ))

        if args.getboolean('Train', 'early_stopping'):
            if epoch - best_epoch > args.getint('Train', 'patience') > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(
                    epoch, validation_loss, best_epoch))
                break



if __name__ == '__main__':
    main()
