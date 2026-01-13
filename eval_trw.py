import time

from joblib import Parallel, delayed
import numpy as np
import torch
from tqdm import tqdm

from fast_auc import fast_auc, fast_ap
from models.gnn_bert import RWBert, GNNEmbedding
from sampler import TRWSampler

'''
Kind of ugly, but wanted to be able to use this in parts of the code other than snapshot_finetune.py
'''

class Evaluator():
    def __init__(self, walk_len, num_eval_iters=1, workers=16, eval_bs=2048, delta=0, downsample=False, device='cpu', dataset='unsw'):
        self.WALK_LEN = walk_len
        self.NUM_EVAL_ITERS = num_eval_iters
        self.EVAL_BS = eval_bs
        self.DELTA = delta
        self.DEVICE = device
        self.workers = workers
        self.DATASET = dataset

        self.downsample = downsample

    def sample(self, tr, src,dst,ts, walk_len, edge_features=None):
        if walk_len > 1:
            rw = tr.rw(src, max_ts=ts, min_ts=(ts-self.DELTA).clamp(0), reverse=True, trim_missing=False)
        else:
            rw = src.unsqueeze(-1)

        if edge_features is not None:
            #mask = torch.tensor([GNNEmbedding.MASK], device=self.DEVICE).repeat(edge_features.size())
            #rw = torch.cat([rw, mask], dim=1)
            #dst = torch.cat([edge_features, dst.unsqueeze(-1)], dim=1).flatten()
            rw = torch.cat([rw, edge_features], dim=1)

        masks = torch.tensor([[GNNEmbedding.MASK]], device=self.DEVICE).repeat(rw.size(0),1)
        rw = torch.cat([rw,masks], dim=1)
        attn_mask = rw != GNNEmbedding.PAD

        return rw, rw==GNNEmbedding.MASK, dst, attn_mask

    @torch.no_grad()
    def parallel_eval(self, model, tr: TRWSampler, te: TRWSampler):
        preds = np.zeros(te.col.size(0))
        prog = tqdm(desc='Eval', total=(te.col.size(0) // self.EVAL_BS)* self.NUM_EVAL_ITERS)

        def thread_job(pid, idx):
            nonlocal preds

            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            samp = te._single_iter(idx, shuffled=False)
            if te.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            walk, mask, tgt, attn_mask = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)
            out = out[mask]
            pred = out[torch.arange(out.size(0)), tgt]

            # Use prob of edge features as part of probability
            if ef is not None:
                pred = pred.view(walk.size(0), -1)
                pred = pred.prod(dim=1)

            preds[idx.cpu()] += pred.squeeze().detach().to('cpu').numpy()
            prog.update()

            del out
            del pred
            torch.cuda.empty_cache()

        if not self.downsample: 
            idxs = torch.arange(te.col.size(0)).split(self.EVAL_BS)
        else: 
            to_sample = torch.rand(te.col.size(0))
            to_sample[te.label == 1] = 0
            idxs = (to_sample < self.downsample).nonzero().squeeze()

        for i in range(self.NUM_EVAL_ITERS):
            Parallel(n_jobs=self.workers, prefer='threads')(
                delayed(thread_job)(i,b)
                for i,b in enumerate(idxs)
            )


        prog.close()

        preds /= self.NUM_EVAL_ITERS
        labels = te.label.numpy()

        auc = fast_auc(labels, preds)
        ap = fast_ap(labels, preds)

        return auc,ap

    @torch.no_grad()
    def parallel_validate(self, model, tr: TRWSampler, va: TRWSampler, percent=1):
        tns = np.zeros(va.col.size(0))
        tps = np.zeros(int(va.col.size(0) * percent))

        prog = tqdm(desc='TNs', total=(va.col.size(0) // self.EVAL_BS)*self.NUM_EVAL_ITERS)

        def thread_job_tn(pid, idx):
            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            samp = va._single_iter(idx, shuffled=False)
            if va.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            walk, mask, tgt, attn_mask = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)
            out = out[mask]
            pred = out[torch.arange(out.size(0)), tgt]

            # Use prob of edge features as part of probability
            if ef is not None:
                pred = pred.view(walk.size(0), -1)
                pred = pred.prod(dim=1)

            tns[idx.cpu()] += pred.squeeze().detach().to('cpu').numpy()
            prog.update()

            del out
            del pred
            torch.cuda.empty_cache()

        if not self.downsample: 
            idxs = torch.arange(va.col.size(0)).split(self.EVAL_BS)
        else: 
            to_sample = torch.rand(va.col.size(0))
            idxs = (to_sample < self.downsample).nonzero().squeeze()

        for i in range(self.NUM_EVAL_ITERS):
            Parallel(n_jobs=self.workers, prefer='threads')(
                delayed(thread_job_tn)(i,b)
                for i,b in enumerate(idxs)
            )

        prog.close()

        prog = tqdm(desc='TPs', total=(tps.shape[0] // self.EVAL_BS)*self.NUM_EVAL_ITERS)
        def thread_job_tp(pid, src,dst,ts,ef,b):
            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            walk, mask, tgt, attn_mask = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)
            out = out[mask]
            pred = out[torch.arange(out.size(0)), tgt]

            # Use prob of edge features as part of probability
            if ef is not None:
                pred = pred.view(walk.size(0), -1)
                pred = pred.prod(dim=1)

            tps[b.cpu()] += pred.squeeze().detach().to('cpu').numpy()
            prog.update()

            del out
            del pred
            torch.cuda.empty_cache()


        src = torch.randint_like(va.col, tr.num_nodes)[:tps.shape[0]]
        dst = torch.randint_like(va.col, tr.num_nodes)[:tps.shape[0]]
        ts = torch.randint_like(va.col, tr.ts.max())[:tps.shape[0]]

        if self.DATASET == 'unsw':
            # Edges only have very specific timecodes
            ts = tr.ts.unique()
            idx = torch.randint_like(src, ts.size(0))
            ts = ts[idx]

        batches = torch.arange(src.size(0)).split(self.EVAL_BS)

        for i in range(self.NUM_EVAL_ITERS):
            if not va.edge_features:
                Parallel(n_jobs=self.workers, prefer='threads')(
                    delayed(thread_job_tp)(i,src[b],dst[b],ts[b],None, b)
                    for i,b in enumerate(batches)
                )
            else:
                efs = tr.edge_attr[torch.randint(0, tr.edge_attr.size(0), (src.size(0),))]
                efs += tr.num_nodes
                Parallel(n_jobs=self.workers, prefer='threads')(
                    delayed(thread_job_tp)(i,src[b],dst[b],ts[b],efs[b], b)
                    for i,b in enumerate(batches)
                )


        prog.close()

        tps /= self.NUM_EVAL_ITERS
        tns /= self.NUM_EVAL_ITERS

        preds = np.concatenate([tps, tns])
        labels = np.zeros(preds.shape[0])
        labels[:tps.shape[0]] = 1

        auc = fast_auc(labels, preds)
        ap = fast_ap(labels, preds)

        return auc,ap


    @torch.no_grad()
    def get_metrics(self, tr,va,te, model):
        te.to(tr.device)
        te_auc, te_ap = self.parallel_eval(model, tr, te)
        print('#'*20)
        print(f'TEST SCORES')
        print('#'*20)
        print(f"AUC: {te_auc:0.4f}, AP:  {te_ap:0.4f}")
        print('#'*20)
        print()
        te.to('cpu')

        va.to(tr.device)
        va_auc, va_ap = self.parallel_validate(model, tr, va)
        print('#'*20)
        print(f'VAL SCORES')
        print('#'*20)
        print(f"AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")
        va.to('cpu')

        return te_auc, te_ap, va_auc, va_ap
    
class CausalEvaluator(Evaluator): 
    def sample(self, tr, src,dst,ts, walk_len, edge_features=None):
        if walk_len > 1:
            rw = tr.rw(src, max_ts=ts, min_ts=(ts-self.DELTA).clamp(0), reverse=True, trim_missing=False)
        else:
            rw = src.unsqueeze(-1)

        if edge_features is not None:
            #mask = torch.tensor([GNNEmbedding.MASK], device=self.DEVICE).repeat(edge_features.size())
            #rw = torch.cat([rw, mask], dim=1)
            #dst = torch.cat([edge_features, dst.unsqueeze(-1)], dim=1).flatten()
            rw = torch.cat([rw, edge_features], dim=1)

        return rw

    @torch.no_grad()
    def parallel_eval(self, model, tr: TRWSampler, te: TRWSampler):
        preds = np.zeros(te.col.size(0))
        prog = tqdm(desc='Eval', total=(te.col.size(0) // self.EVAL_BS)* self.NUM_EVAL_ITERS)

        def thread_job(pid, idx):
            nonlocal preds

            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            samp = te._single_iter(idx, shuffled=False)
            if te.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            walk = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, return_loss=False).logits

            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = out[:, -1]
            out = out[torch.arange(out.size(0)), dst]
            out = 1 - torch.sigmoid(out)

            preds[idx.cpu()] += out.squeeze().detach().to('cpu').numpy()
            prog.update()

            del out
            torch.cuda.empty_cache()

        if not self.downsample: 
            idxs = torch.arange(te.col.size(0)).split(self.EVAL_BS)
        else: 
            to_sample = torch.rand(te.col.size(0))
            to_sample[te.label == 1] = 0
            idxs = (to_sample < self.downsample).nonzero().squeeze()

        for i in range(self.NUM_EVAL_ITERS):
            Parallel(n_jobs=self.workers, prefer='threads')(
                delayed(thread_job)(i,b)
                for i,b in enumerate(idxs)
            )


        prog.close()

        preds /= self.NUM_EVAL_ITERS
        labels = te.label.numpy()

        auc = fast_auc(labels, preds)
        ap = fast_ap(labels, preds)

        return auc,ap

    @torch.no_grad()
    def parallel_validate(self, model, tr: TRWSampler, va: TRWSampler, percent=1):
        tns = np.zeros(va.col.size(0))
        tps = np.zeros(int(va.col.size(0) * percent))

        prog = tqdm(desc='TNs', total=(va.col.size(0) // self.EVAL_BS)*self.NUM_EVAL_ITERS)

        def thread_job_tn(pid, idx):
            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            samp = va._single_iter(idx, shuffled=False)
            if va.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            walk = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, return_loss=False).logits

            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = out[:, -1]
            out = out[torch.arange(out.size(0)), dst]
            out = 1 - torch.sigmoid(out)

            tns[idx.cpu()] += out.squeeze().detach().to('cpu').numpy()
            prog.update()

            del out
            torch.cuda.empty_cache()

        if not self.downsample: 
            idxs = torch.arange(va.col.size(0)).split(self.EVAL_BS)
        else: 
            to_sample = torch.rand(va.col.size(0))
            idxs = (to_sample < self.downsample).nonzero().squeeze()

        for i in range(self.NUM_EVAL_ITERS):
            Parallel(n_jobs=self.workers, prefer='threads')(
                delayed(thread_job_tn)(i,b)
                for i,b in enumerate(idxs)
            )

        prog.close()

        prog = tqdm(desc='TPs', total=(tps.shape[0] // self.EVAL_BS)*self.NUM_EVAL_ITERS)
        def thread_job_tp(pid, src,dst,ts,ef,b):
            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            walk = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, return_loss=False).logits

            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = out[:, -1]
            out = out[torch.arange(out.size(0)), dst]
            out = 1 - torch.sigmoid(out)

            tps[b.cpu()] += out.squeeze().detach().to('cpu').numpy()
            prog.update()

            del out
            torch.cuda.empty_cache()


        src = torch.randint_like(va.col, tr.num_nodes)[:tps.shape[0]]
        dst = torch.randint_like(va.col, tr.num_nodes)[:tps.shape[0]]
        ts = torch.randint_like(va.col, tr.ts.max())[:tps.shape[0]]

        if self.DATASET == 'unsw':
            # Edges only have very specific timecodes
            ts = tr.ts.unique()
            idx = torch.randint_like(src, ts.size(0))
            ts = ts[idx]

        batches = torch.arange(src.size(0)).split(self.EVAL_BS)

        for i in range(self.NUM_EVAL_ITERS):
            if not va.edge_features:
                Parallel(n_jobs=self.workers, prefer='threads')(
                    delayed(thread_job_tp)(i,src[b],dst[b],ts[b],None, b)
                    for i,b in enumerate(batches)
                )
            else:
                efs = tr.edge_attr[torch.randint(0, tr.edge_attr.size(0), (src.size(0),))]
                efs += tr.num_nodes
                Parallel(n_jobs=self.workers, prefer='threads')(
                    delayed(thread_job_tp)(i,src[b],dst[b],ts[b],efs[b], b)
                    for i,b in enumerate(batches)
                )


        prog.close()

        tps /= self.NUM_EVAL_ITERS
        tns /= self.NUM_EVAL_ITERS

        preds = np.concatenate([tps, tns])
        labels = np.zeros(preds.shape[0])
        labels[:tps.shape[0]] = 1

        auc = fast_auc(labels, preds)
        ap = fast_ap(labels, preds)

        return auc,ap