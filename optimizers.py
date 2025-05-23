import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, awl, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.awl = awl


    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        loss_static = nn.CrossEntropyLoss(reduction='mean')
        
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, pred_static, factors, time, time_phase = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_static = loss_static(pred_static, truth)
                
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time, time_phase)
                l_sum = self.awl(l_fit, l_static)
                l = l_sum + l_reg + l_time
                
                self.optimizer.zero_grad()
                l.backward()
                
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                #print('loss={},reg={},cont={}'.format(l_fit.item(),l_reg.item(),l_time.item()))
                bar.set_postfix(
                    loss_temp=f'{l_fit.item():.0f}',
                    loss_stru=f'{l_static.item():.2f}',
                    reg=f'{l_reg.item():.0f}',
                    loss_time_reg=f'{l_time.item():.0f}'
                )


class IKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        loss_static = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                time_range = actual_examples[b_begin:b_begin + self.batch_size].cuda()

                ## RHS Prediction loss
                sampled_time = (
                        torch.rand(time_range.shape[0]).cuda() * (time_range[:, 4] - time_range[:, 3]).float() +
                        time_range[:, 3].float()
                ).round().long()
                with_time = torch.cat((time_range[:, 0:3], sampled_time.unsqueeze(1)), 1)

                predictions, pred_static, factors, time, time_phase = self.model.forward(with_time)
                truth = with_time[:, 2]

                l_fit = loss(predictions, truth)
                l_static = loss_static(pred_static, truth)

                ## Time prediction loss (ie cross entropy over time)
                time_loss = 0.
                if self.model.has_time():
                    filtering = ~(
                        (time_range[:, 3] == 0) *
                        (time_range[:, 4] == (self.dataset.n_timestamps - 1))
                    ) # NOT no begin and no end
                    these_examples = time_range[filtering, :]
                    truth = (
                            torch.rand(these_examples.shape[0]).cuda() * (these_examples[:, 4] - these_examples[:, 3]).float() +
                            these_examples[:, 3].float()
                    ).round().long()
                    time_predictions = self.model.forward_over_time(these_examples[:, :3].cuda().long())
                    time_loss = loss(time_predictions, truth.cuda())

                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time, time_phase)
                l = l_fit + l_static + l_reg + l_time + time_loss

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(with_time.shape[0])
                bar.set_postfix(
                    loss_temp=f'{l_fit.item():.0f}',
                    loss_stru=f'{l_static.item():.2f}',
                    loss_time=f'{time_loss if type(time_loss) == float else time_loss.item() :.0f}',
                    reg=f'{l_reg.item():.0f}',
                    loss_time_reg=f'{l_time.item():.4f}'
                )

