from abc import ABC, abstractmethod

from typing import Tuple, List, Dict
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    #scores = q[0] @ rhs + 0.1 * q[1] @ self.get_rhs_static(c_begin, chunk_size)
                    #targets = self.score(these_queries)
                    scores_tem = q[0] @ rhs
                    scores_cs =  q[1] @ self.get_rhs_static(c_begin, chunk_size) 
                    targets_tem, targets_cs = self.score(these_queries)
                    #print("scores_cs:\n{}\n\ntargets_cs:\n{}\n".format(scores_cs, targets_cs))

                    #assert not torch.any(torch.isinf(scores)), "inf scores"
                    #assert not torch.any(torch.isnan(scores)), "nan scores"
                    #assert not torch.any(torch.isinf(targets)), "inf targets"
                    #assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores_tem[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores_tem[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (torch.mul(scores_tem >= targets_tem, scores_cs > targets_cs)).float(), dim=1
                        #(scores_tem >= targets_tem).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores_tem = q[0] @ rhs
                scores_cs =  q[1] @ self.get_rhs_static(c_begin, chunk_size)
                targets_tem, targets_cs = self.score(queries)
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores_tem.shape[1], f"fuck {scores_tem.shape[1]} {max_to_filter}"
                        scores_tem[i, filter_in_chunk] = -1e6
                    else:
                        scores_tem[i, filter_out] = -1e6
                ranks += torch.sum(
                    (torch.mul(scores_tem >= targets_tem, scores_cs > targets_cs)).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks

class SpidER(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,no_time_emb=False, weight=0.1, init_size: float = 1e-2):
        super(SpidER, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.rank_static = rank // 20
        self.weight = weight
        # self.W = nn.Embedding(2*rank, 1, sparse=True)
        # self.W.weight.data *= 0

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[3], sizes[1]] # without no_time_emb
            ])
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank_static, sparse=True)
            for s in [sizes[0], sizes[1]]
            ])
        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.static_embeddings[0].weight.data *= init_size     # static entity embedding 7128 * 10
        self.static_embeddings[1].weight.data *= init_size     # static relation embedding 460 * 200
        
        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

    @staticmethod
    def has_time():
        return True
    
    def score(self, x):
                
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        

        time_phase = torch.abs(self.embeddings[3](x[:, 3]))
        #time_phase = torch.sin(time_phase[:, :self.rank]), torch.cos(time_phase[:, self.rank:])
        time_phase = torch.cos(time_phase[:, :self.rank]), torch.cos(time_phase[:, self.rank:])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0] + time_phase[0], rel[1] * time[0] + time_phase[0], rel[0] * time[1] + time_phase[1], rel[1] * time[1] + time_phase[1]
        #rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1] 
        #full_rel = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = torch.exp(rt[0] - rt[3]) * torch.cos(rt[1] + rt[2]), torch.exp(rt[0] - rt[3]) * torch.sin(rt[1] + rt[2])
            
        h_static = self.static_embeddings[0](x[:, 0])
        r_static = self.static_embeddings[1](x[:, 1])
        t_static = self.static_embeddings[0](x[:, 2])
        
        h_static = h_static[:, :self.rank_static], h_static[:, self.rank_static:]
        r_static = r_static[:, :self.rank_static], r_static[:, self.rank_static:]
        t_static = t_static[:, :self.rank_static], t_static[:, self.rank_static:]
        
        return torch.sum(
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1], 1, keepdim=True
                ), torch.sum(
            (h_static[0] * r_static[0] - h_static[1] * r_static[1]) * t_static[0] +
            (h_static[1] * r_static[0] + h_static[0] * r_static[1]) * t_static[1],
            1, keepdim=True)
          
          
    def forward(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
       
        time_phase = torch.abs(self.embeddings[3](x[:, 3]))
        #time_phase = torch.sin(time_phase[:, :self.rank]), torch.cos(time_phase[:, self.rank:])
        time_phase = torch.cos(time_phase[:, :self.rank]), torch.cos(time_phase[:, self.rank:])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0] + time_phase[0], rel[1] * time[0] + time_phase[0], rel[0] * time[1] + time_phase[1], rel[1] * time[1] + time_phase[1]
        #rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1] 
        #full_rel = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = torch.exp(rt[0] - rt[3]) * torch.cos(rt[1] + rt[2]), torch.exp(rt[0] - rt[3]) * torch.sin(rt[1] + rt[2])
            
        h_static = self.static_embeddings[0](x[:, 0]) # 1000 * 5
        r_static = self.static_embeddings[1](x[:, 1]) # 1000 * 5
        t_static = self.static_embeddings[0](x[:, 2]) # 1000 * 5

        h_static = h_static[:, :self.rank_static], h_static[:, self.rank_static:]
        r_static = r_static[:, :self.rank_static], r_static[:, self.rank_static:]
        t_static = t_static[:, :self.rank_static], t_static[:, self.rank_static:]
            
        right_static = self.static_embeddings[0].weight
        right_static = right_static[:, :self.rank_static], right_static[:, self.rank_static:]

        regularizer = (
                torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
                torch.sqrt(h_static[0] ** 2 + h_static[1] ** 2),
                torch.sqrt(r_static[0] ** 2 + r_static[1] ** 2),
                torch.sqrt(t_static[0] ** 2 + t_static[1] ** 2)
            )
            
        return (
                    (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                    (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t(),
                    (h_static[0] * r_static[0] - h_static[1] * r_static[1]) @ right_static[0].t() +
                    (h_static[1] * r_static[0] + h_static[0] * r_static[1]) @ right_static[1].t(),
                    regularizer,
                    self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight,  
                    self.embeddings[3].weight[:-1] if self.no_time_emb else self.embeddings[3].weight
               )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        
        rel_no_time = self.embeddings[4](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
    
    def get_rhs_static(self, chunk_begin: int, chunk_size: int):
        return self.static_embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        time_phase = torch.abs(self.embeddings[3](queries[:, 3]))
        #time_phase = torch.sin(time_phase[:, :self.rank]), torch.cos(time_phase[:, self.rank:])
        time_phase = torch.cos(time_phase[:, :self.rank]), torch.cos(time_phase[:, self.rank:])

        rt = rel[0] * time[0] + time_phase[0], rel[1] * time[0] + time_phase[0], rel[0] * time[1] + time_phase[1], rel[1] * time[1] + time_phase[1]
        #rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1] 
        #full_rel = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = torch.exp(rt[0] - rt[3]) * torch.cos(rt[1] + rt[2]), torch.exp(rt[0] - rt[3]) * torch.sin(rt[1] + rt[2])
            
        h_static = self.static_embeddings[0](queries[:, 0])
        r_static = self.static_embeddings[1](queries[:, 1])
        
        h_static = h_static[:, :self.rank_static], h_static[:, self.rank_static:]
        r_static = r_static[:, :self.rank_static], r_static[:, self.rank_static:]

        return torch.cat([
                lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
                lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
            ], 1), torch.cat([
                h_static[0] * r_static[0] - h_static[1] * r_static[1],
                h_static[1] * r_static[0] + h_static[0] * r_static[1]
            ], 1)