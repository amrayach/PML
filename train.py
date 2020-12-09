import configparser
import os
import shutil
from datetime import datetime

import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torchviz import make_dot
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import utils
from data_loader import load_data, MyDataset
from model import CharacterLevelCNN


def train(model, training_generator, optimizer, criterion, epoch, start_iter, writer, log_file, scheduler, class_names,
          args, print_every=10):
    model.train()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(training_generator)

    progress_bar = tqdm(enumerate(training_generator, start=start_iter),
                        total=num_iter_per_epoch)

    y_true = []
    y_pred = []
    for iter, batch in progress_bar:
        features, labels = batch
        labels.sub_(1)

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
                                                list_metrics=["accuracy", "f1_weighted", "f1_micro", "f1_macro"])

        losses.update(loss.data, features.size(0))
        accuracies.update(training_metrics["accuracy"], features.size(0))

        f1_weighted = training_metrics['f1_weighted']
        f1_micro = training_metrics['f1_micro']
        f1_macro = training_metrics["f1_macro"]

        writer.add_scalar('Train/Loss',
                          loss.item(),
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Train/Accuracy',
                          training_metrics['accuracy'],
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Train/f1-weighted',
                          f1_weighted,
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Train/f1-micro',
                          f1_micro,
                          epoch * num_iter_per_epoch + iter)

        writer.add_scalar('Train/f1-macro',
                          f1_macro,
                          epoch * num_iter_per_epoch + iter)

        #f1_scalars = {'f1_weighted': f1_weighted, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
        #writer.add_scalars('Train/all-f1-scores',
        #                   f1_scalars,
        #                   epoch * num_iter_per_epoch + iter)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, iter)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if (iter % print_every == 0) and (iter > 0):
            print("\n[Training - Epoch: {}], LR: {} , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
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
                # f1_by_class = 'F1 Scores by class: '
                # for class_name in class_names:
                #    f1_by_class += f"{class_name} : {np.round(intermediate_report[class_name]['f1-score'], 4)} |"
                # print(f1_by_class)

    f1_train_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_train_micro = f1_score(y_true, y_pred, average='micro')
    f1_train_macro = f1_score(y_true, y_pred, average='macro')

    writer.add_scalar('Train/loss/epoch', losses.avg, epoch + iter)
    writer.add_scalar('Train/acc/epoch', accuracies.avg, epoch + iter)
    writer.add_scalar('Train/f1-weighted/epoch', f1_train_weighted, epoch + iter)
    writer.add_scalar('Train/f1-micro/epoch', f1_train_micro, epoch + iter)
    writer.add_scalar('Train/f1-macro/epoch', f1_train_macro, epoch + iter)

    #f1_scalars = {'f1_weighted': f1_train_weighted, 'f1_micro': f1_train_micro, 'f1_macro': f1_train_macro}
    #writer.add_scalars('Train/all-f1-scores/epoch',
    #                   f1_scalars,
    #                   epoch * num_iter_per_epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, 'a') as f:
        f.write(f'Training on Epoch {epoch} \n')
        f.write(f'Average loss: {losses.avg.item()} \n')
        f.write(f'Average accuracy: {accuracies.avg.item()} \n')
        f.write(f'F1 Weighted score: {f1_train_weighted} \n\n')
        f.write(f'F1 Micro score: {f1_train_micro} \n\n')
        f.write(f'F1 Macro score: {f1_train_macro} \n\n')
        f.write(report)
        f.write('*' * 25)
        f.write('\n')

    return losses.avg.item(), accuracies.avg.item(), f1_train_weighted


def evaluate(model, validation_generator, criterion, epoch, writer, log_file, print_every=25):
    model.eval()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(validation_generator)

    y_true = []
    y_pred = []

    for n_iter, batch in tqdm(enumerate(validation_generator), total=num_iter_per_epoch):
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

        writer.add_scalar('Validation/Loss',
                          loss.item(),
                          epoch * num_iter_per_epoch + n_iter)

        writer.add_scalar('Validation/Accuracy',
                          accuracy,
                          epoch * num_iter_per_epoch + n_iter)

        writer.add_scalar('Validation/f1-weighted',
                          f1_weighted,
                          epoch * num_iter_per_epoch + n_iter)

        writer.add_scalar('Validation/f1-micro',
                          f1_micro,
                          epoch * num_iter_per_epoch + n_iter)

        writer.add_scalar('Validation/f1-macro',
                          f1_macro,
                          epoch * num_iter_per_epoch + n_iter)

        #f1_scalars = {'f1_weighted': f1_weighted, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
        #writer.add_scalars('Test/all-f1-scores',
        #                   f1_scalars,
        #                   epoch * num_iter_per_epoch + iter)

        if (n_iter % print_every == 0) and (n_iter > 0):
            print("[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                n_iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg
            ))

    f1_test_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_test_micro = f1_score(y_true, y_pred, average='micro')
    f1_test_macro = f1_score(y_true, y_pred, average='macro')

    writer.add_scalar('Validation/loss/epoch', losses.avg, epoch)
    writer.add_scalar('Validation/acc/epoch', accuracies.avg, epoch)
    writer.add_scalar('Validation/f1-weighted/epoch', f1_test_weighted, epoch)
    writer.add_scalar('Validation/f1-micro/epoch', f1_test_micro, epoch)
    writer.add_scalar('Validation/f1-macro/epoch', f1_test_macro, epoch)
    #f1_scalars = {'f1_weighted': f1_test_weighted, 'f1_micro': f1_test_micro, 'f1_macro': f1_test_macro}
    #writer.add_scalars('Test/all-f1-scores/epoch',
    #                   f1_scalars,
    #                   epoch * num_iter_per_epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, 'a') as f:
        f.write(f'Validation on Epoch {epoch} \n')
        f.write(f'Average loss: {losses.avg.item()} \n')
        f.write(f'Average accuracy: {accuracies.avg.item()} \n')
        f.write(f'F1 Weighted score {f1_test_weighted} \n\n')
        f.write(f'F1 Micro score {f1_test_micro} \n\n')
        f.write(f'F1 Macro score {f1_test_macro} \n\n')
        f.write(report)
        f.write('=' * 50)
        f.write('\n')

    return losses.avg.item(), accuracies.avg.item(), f1_test_weighted


def save_checkpoint(model, state, optimizer, args, epoch, validation_loss, validation_accuracy, validation_f1):
    #model_is_cuda = next(model.parameters()).is_cuda
    #model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    os.makedirs(args.get('Log', 'output') + '/' + args.get('Log', 'model_name'), exist_ok=True)
    torch.save(state,
               args.get('Log', 'output') + args.get('Log',
                                                    'model_name') + '/model_{}_epoch_{}_l0_{}_lr_{}_loss_{}_acc_{}_f1_{}.pt'.format(
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


def main():
    args = configparser.ConfigParser()
    args.read('argsConfig.ini')


    if args.getboolean('Log', 'flush_history') == 1:
        objects = os.listdir(args.get('Log', 'log_path'))
        for f in objects:
            if os.path.isdir(args.get('Log', 'log_path') + f):
                shutil.rmtree(args.get('Log', 'log_path') + f)

    if args.getboolean('Log', 'delete_model_name_dir'):
        objects = os.listdir(args.get('Log', 'output'))
        for f in objects:
            if f == args.get('Log', 'model_name'):
                shutil.rmtree(args.get('Log', 'output') + args.get('Log', 'model_name') + '/')

    now = datetime.now()
    logdir = args.get('Log', 'log_path') + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)
    log_file = logdir + 'log.txt'
    writer = SummaryWriter(logdir)

    texts, labels, number_of_classes, sample_weights = load_data(args, 'train')

    class_names = sorted(list(set(labels)))
    class_names = [str(class_name - 1) for class_name in class_names]

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

    if args.getboolean('Model', 'visualize_model_graph'):
        x = torch.zeros((args.getint('Train', 'batch_size'),
                       args.getint('DataSet', 'char_num'),
                       args.getint('DataSet', 'l0')))
        out = model(x)
        make_dot(out).render("CharacterLevelCNN", format="png", quiet_view=True)

    if torch.cuda.is_available():
        model.cuda()

    # todo check other other loss functions for binary and multi-label problems
    if args.get('Train', 'criterion') == 'nllloss':
        criterion = nn.NLLLoss()

    # criterion = nn.BCELoss()

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

    if args.get('Train', 'scheduler') == 'clr':
        stepsize = int(args.getint('Train', 'clr_step_size') * len(training_generator))
        clr = utils.cyclical_lr(stepsize, args.getfloat('Train', 'clr_min_lr'), args.getfloat('Train', 'clr_max_lr'))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = None
        lr_half_cnt = 0

    utils.init_log(log_file=log_file, args=args, labels=class_names)
    try:
        for epoch in range(start_epoch, args.getint('Train', 'epochs')):

            training_loss, training_accuracy, train_f1 = train(model,
                                                               training_generator,
                                                               optimizer,
                                                               criterion,
                                                               epoch,
                                                               start_iter,
                                                               writer,
                                                               log_file,
                                                               scheduler,
                                                               class_names,
                                                               args,
                                                               args.getint('Log', 'print_out_every'))

            validation_loss, validation_accuracy, validation_f1 = evaluate(model,
                                                                           validation_generator,
                                                                           criterion,
                                                                           epoch,
                                                                           writer,
                                                                           log_file)


            print('\n[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}'.
                  format(epoch + 1, args.getint('Train', 'epochs'), training_loss, training_accuracy, validation_loss,
                         validation_accuracy))
            print("=" * 50)

            with open(log_file, 'a') as f:
                f.write('[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}\n'.
                  format(epoch + 1, args.getint('Train', 'epochs'), training_loss, training_accuracy, validation_loss,
                         validation_accuracy))
                f.write('=' * 50)

            # learning rate scheduling
            if args.get('Train', 'scheduler') == 'step':
                if args.get('Train', 'optimizer') == 'SGD' and ((epoch + 1) % 3 == 0) and lr_half_cnt < 10:
                    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    current_lr /= 2
                    lr_half_cnt += 1
                    print('Decreasing learning rate to {0}'.format(current_lr))
                    with open(log_file, 'a') as f:
                        f.write('Decreasing learning rate to {0}\n'.format(current_lr))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

            if args.getboolean('Log', 'checkpoint'):

                state = {'epoch': epoch, 'optimizer': optimizer.state_dict(), 'best_f1': best_f1}

                if args.getint('Log', 'save_interval') > 0 and epoch % args.getint('Log', 'save_interval') == 0:
                    save_checkpoint(model, state, optimizer, args, epoch, validation_loss, validation_accuracy,
                                    validation_f1)

                if validation_f1 > best_f1:
                    best_f1 = validation_f1
                    best_epoch = epoch
                    save_checkpoint(model, state, optimizer, args, epoch, validation_loss, validation_accuracy,
                                    validation_f1)

            if args.getboolean('Train', 'early_stopping'):
                if epoch - best_epoch > args.getint('Train', 'patience') > 0:
                    print("Early-stopping: Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(
                        epoch, validation_loss, best_epoch))
                    break
    except KeyboardInterrupt:
        print('Exit Keyboard interrupt\n')
        save_checkpoint(model, state, optimizer, args, epoch, validation_loss, validation_accuracy, validation_f1)


if __name__ == '__main__':
    main()
