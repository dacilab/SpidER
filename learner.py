import argparse
from typing import Dict
import torch
from torch import optim
from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import SpidER
from regularizers import N3, Spiral3, DURA
import os,json
from loss import AutomaticWeightedLoss
import numpy as np


parser = argparse.ArgumentParser(
    description="SpidER"
)
parser.add_argument(
    '--dataset', type=str, default='ICEWS14',
    help="Dataset name"
)

parser.add_argument(
    '--model', default='SpidER', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--space', default="complex", type=str,
    help="Using space"
)

parser.add_argument(
    '--weight_static', default=0., type=float,
    help="Weight of static score"
)


args = parser.parse_args()
root = 'results/'+ args.dataset + '/' + args.model
PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/emb_reg{:.5f}/time_reg{:.5f}/'.format(args.rank, args.learning_rate, args.batch_size, args.emb_reg, args.time_reg))

def save_model(model, optimizer, args):
        '''
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''
    
        argparse_dict = vars(args)
        with open(os.path.join(PATH, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(PATH, 'checkpoint')
        )
    
        temporal_embedding = model.embeddings
        np.save(
            os.path.join(PATH, 'temporal_embedding'), 
            temporal_embedding
        )
    
        static_embedding = model.static_embeddings
        np.save(
            os.path.join(PATH, 'static_embedding'), 
            static_embedding
        )

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg
         ):


    modelname = model
    datasetname = dataset
    
    dataset = TemporalDataset(dataset)
    
    sizes = dataset.get_shape()
   
    model = {
        'SpidER': SpidER(sizes, rank, no_time_emb=args.no_time_emb, weight = args.weight_static)
        }[model]
    model = model.cuda()
    awl = AutomaticWeightedLoss(2)
    if args.dataset == 'wikidata12k':
        opt = optim.Adagrad([{'params': model.parameters(), 'weight_decay': 0}], lr=learning_rate)
    else:
        opt = optim.Adagrad([{'params': model.parameters(), 'weight_decay': 0}, {'params': awl.parameters(), 'weight_decay': 0}], lr=learning_rate)

    print("Start training process: ", modelname, "on", datasetname, "using", "space = ", args.space, "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg)

    emb_reg = N3(emb_reg)
    
    time_reg = Spiral3(time_reg)
  
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}

    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()

        if dataset.has_intervals():
            optimizer = IKBCOptimizer(
                model, emb_reg, time_reg, opt, dataset, 
            batch_size=args.batch_size,
            )
            optimizer.epoch(examples)
        else:
            optimizer = TKBCOptimizer(
                model, emb_reg, time_reg, opt, awl=awl,
                batch_size=args.batch_size,
            )
            optimizer.epoch(examples)

       
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:

            if dataset.has_intervals(): 
                valid, test, train = [
                    dataset.eval(model, split, -1 if split != 'train' else 50000)
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid)
                print("test: ", test)
                print("train: ", train)

            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results
            f = open(os.path.join(PATH, 'result.txt'), 'a+')
            f.write("\n[Epoch:{}]-VALID : ".format(epoch))
            f.write(str(valid))
            f.close()
            
            # early-stop with patience
            if args.dataset == 'wikidata12k':
                mrr_valid = valid['MRR_all']
            else:
                mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 10:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            if not dataset.has_intervals():
                curve['train'].append(train)
    
                print("\t TRAIN: ", train)
            print("\t VALID : ", valid)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    save_model(model, optimizer.optimizer, args)
    if args.dataset != 'wikidata12k':
        results = avg_both(*dataset.eval(model, 'test', -1))
        print("\n\nTEST : ", results)
        f = open(os.path.join(PATH, 'result.txt'), 'a+')
        f.write("\n\nTEST : ")
        f.write(str(results))
        f.close()

if __name__ == '__main__':
    learn()


