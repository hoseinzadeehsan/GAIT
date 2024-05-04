
import argparse
import configargparse
import numpy as np
from os.path import join
import torch
import dgl
from torch.utils.data import DataLoader
from dataset import TableDatasetMulti
import configs
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import warnings
from gnn import GCN
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = join(configs.root_dir, 'data')
metric_dir = join(configs.root_dir, 'results', 'GCN')
if not os.path.exists(metric_dir):
    os.makedirs(metric_dir)

seed = 0
torch.manual_seed(seed)
if device == torch.device('cuda'):
    torch.cuda.manual_seed_all(seed)
label_enc = LabelEncoder()

def collate(graphs):
    graph = dgl.batch(graphs)
    return graph

def evaluation(model, graph_batch, loss_fcn):
    model.eval()
    with torch.no_grad():
        logits = model(graph_batch, graph_batch.ndata['feat'])
        target = graph_batch.ndata['label'].view(-1)
        loss = loss_fcn(logits, target)
        _, pred = torch.max(logits, dim=1)

        return pred.cpu().numpy(), target.cpu().numpy(), loss.item()


def validate(args, model, valid_dataset,  data_type, save_metric=False):
    model.eval()
    loss_fcn = torch.nn.CrossEntropyLoss()
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate, pin_memory=True)

    pred_list = []
    label_list = []
    loss_list = []

    with torch.no_grad():
        for graph_batch in valid_dataloader:
            graph_batch = graph_batch.to(device)
            pred, label, loss = evaluation(model, graph_batch, loss_fcn)
            pred_list.extend(pred)
            label_list.extend(label)
            loss_list.append(loss)

    acc = accuracy_score(pred_list, label_list)
    metric = classification_report(label_list, pred_list, output_dict=True)
    mean_loss = np.array(loss_list).mean()

    print("{}:  loss: {:.4f} | accuracy: {:.4f} | macro_f: {:.4f} | weighted_f: {:.4f}"
          .format( data_type, mean_loss, acc, metric['macro avg']['f1-score'], metric['weighted avg']['f1-score']))

    if save_metric:
        df = pd.DataFrame(metric).transpose()
        df.to_csv(join(metric_dir, 'classification report_{}.csv'.format( data_type)))
        return

    return acc, metric['macro avg']['f1-score'], metric['weighted avg']['f1-score'], mean_loss

def train(args, model, train_dataset, validation_dataset):
    cur_step = 0
    best_weighted_f1 = -1
    best_loss = 10000
    best_accuracy = -1
    best_macro_f = -1

    adr = os.path.join(configs.root_dir, 'saved_models_GCN',args.model_name + args.data_name + '_{}_{}'.format(args.num_layers, args.num_hidden))
    adr_last = os.path.join(configs.root_dir, 'saved_models_GCN', args.model_name + args.data_name + 'last_{}_{}'.format(args.num_layers,args.num_hidden))
    adr_state_dict = os.path.join(configs.root_dir, 'saved_models_GCN',args.model_name + args.data_name + '_state_dict_{}_{}'.format(args.num_layers, args.num_hidden))
    adr_state_dict_best = os.path.join(configs.root_dir, 'saved_models_GCN',args.model_name + args.data_name + '_state_dict_best_{}_{}'.format(args.num_layers, args.num_hidden))

    print("Saving model to: ", adr)

    loss_fcn = torch.nn.CrossEntropyLoss()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=5,
                                                           threshold=0.01,
                                                           threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0.00001,
                                                           eps=1e-08,
                                                           verbose=True)

    print('Current learning rate', optimizer.param_groups[0]['lr'])

    for epoch in range(args.epochs):
        model.train()
        pred_list = []
        label_list = []
        loss_list = []
        for graph_batch in train_dataloader:
            graph_batch = graph_batch.to(device)
            logits = model(graph_batch, graph_batch.ndata['feat'])
            label = graph_batch.ndata['label'].view(-1)

            loss = loss_fcn(logits, label)
            _, pred = torch.max(logits, dim=1)

            pred_list.extend(pred.cpu().numpy())
            label_list.extend(label.cpu().numpy())
            loss_list.append(loss.item())

            optimizer.zero_grad()
            if loss == 0:
                continue
            loss.backward()
            optimizer.step()

        loss_data = np.array(loss_list).mean()
        acc = accuracy_score(pred_list, label_list)
        metric = classification_report(label_list, pred_list, output_dict=True)

        print("Epoch {:05d}\n"
              "Train: loss: {:.4f} | accuracy: {:.4f} | macro_f: {:.4f} | weighted_f: {:.4f}"
              .format(epoch + 1, loss_data, acc, metric['macro avg']['f1-score'],
                      metric['weighted avg']['f1-score']))


        torch.save(model, adr_last)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_weighted_f1': best_weighted_f1,
            'best_accuracy': best_accuracy,
            'best_macro_f': best_macro_f
        }, adr_state_dict)

        scheduler.step(loss_data)

        val_acc, val_macro_f, val_weighted_f, val_loss = validate(args, model, validation_dataset, 'validation')

        # choosing best model according to best validation accuracy
        if best_macro_f < val_macro_f:
            best_weighted_f1 = val_weighted_f
            best_loss = val_loss
            best_accuracy = val_acc
            best_macro_f = val_macro_f
            cur_step = 0

            torch.save(model, adr)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_weighted_f1': best_weighted_f1,
                'best_accuracy': best_accuracy,
                'best_macro_f': best_macro_f
            }, adr_state_dict_best)

        else:
            cur_step += 1
            if cur_step == args.patience:
                print('Early stopping is activated')
                break


def main(args):
    # load and preprocess dataset
    train_dataset = TableDatasetMulti(args=args, data_name=args.data_name + '_train',  data_type='train',
                                      ratio=args.train_valid_ratio)
    if args.data_name.rsplit('_', 1)[0] == 'dataset_semtab':  # or args.data_name.rsplit('_', 2)[0] == 'dataset_semtab':
        validation_dataset = TableDatasetMulti(args=args, data_name=args.data_name + '_validation',  data_type='validation',
                                               ratio=args.train_valid_ratio)
    else:
        validation_dataset = TableDatasetMulti(args=args, data_name=args.data_name + '_train',  data_type='train',
                                               ratio=args.train_valid_ratio)
    test_dataset = TableDatasetMulti(args=args, data_name=args.data_name + '_test')
    num_feats = train_dataset.feature_size
    model = GCN(num_feats, args.num_hidden, args.classes, args.num_layers, F.relu, args.residual, False, args.dropout)

    model = model.to(device)

    if args.mode == 'train':
        train(args, model, train_dataset, validation_dataset)

    adr_best_model = os.path.join(configs.root_dir, 'saved_models_GCN',
                                  args.model_name + args.data_name + '_state_dict_best_{}_{}'.format(
                                      args.num_layers, args.num_hidden))
    best_model = torch.load(adr_best_model, map_location=device)
    model.load_state_dict(best_model['model_state_dict'])
    model = model.to(device)

    print('Final results ------------')
    validate(args, model, train_dataset, 'train', True)
    validate(args, model, validation_dataset, 'validation', True)
    validate(args, model, test_dataset, 'test', True)

if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument('-c', '--config_file', is_config_file=True, help='config file path')
    parser.add_argument("--max_col", type=int, default=6,
                        help="Max Columns in a table")
    parser.add_argument('--dropout', type=float, default=0,
                        help="dropout")
    parser.add_argument("--patience", type=int, default=100,
                        help="patience before early stopping")
    parser.add_argument("--classes", type=int, default=275,
                        help="Number of final classes")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", nargs='+', type=int, default=[275],
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--self-loop", action="store_true", default=True,
                        help="Whether to use self loop")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="batch size used for training, validation and test")
    parser.add_argument('--data-name', type=str, default='dataset_semtab_4',
                        help="for loading dataset")
    parser.add_argument('--mode', default='train',
                        help="train or eval")
    parser.add_argument('--train-valid-ratio', type=float, default=1,
                        help="ratio for dividing train data between train and validation")
    parser.add_argument('--model-name', type=str, default='tmp',
                        help="saving model")
    parser.add_argument('--num-workers', default=0, type=int)
    args = parser.parse_args()

    print(args)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        main(args)
