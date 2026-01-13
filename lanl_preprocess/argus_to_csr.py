from collections import defaultdict
from math import log10

from joblib import Parallel, delayed
import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

from load_utils import id_edge_a
from load_lanl import load_partial_lanl, TIMES

# Same config as reported in paper
END = TIMES['test']
DELTA = 60**2

def build_csr(raw=False): 
    t = 0 
    jobs = []
    ea_map = dict()

    while t < END: 
        jobs.append((t,t+DELTA))
        t += DELTA 

    def get_or_add(e): 
        if (eid := ea_map.get(e)) is None: 
            eid = len(ea_map)
            ea_map[e] = eid 
        return eid 

    [get_or_add(i) for i in range(100)]

    def ea_to_token(ef): 
        if ef > 1: 
            ef = int(ef) 

            if ef < 100: 
                return get_or_add(ef)
            else: 
                return get_or_add(f'10^{int(log10(ef))}')
            
        else: 
            ef = int(ef*10)
            return get_or_add(f'0.{ef}')
        
    def fmt_ea_row(row): 
        return [ea_to_token(ea) for ea in row]

    num_nodes = 0 
    csr = defaultdict(lambda : [[],[],[],[]])
    test = 0
    for j in tqdm(jobs, desc='Reading raw'): 
        tgraph = load_partial_lanl(start=j[0], end=j[1], delta=DELTA, is_test=True, use_flows=True, ea_fn=id_edge_a) 

        if not num_nodes: 
            num_nodes = tgraph.num_nodes

        # Took 25 mins of runtime to hit this error and realize
        # this check was needed :)
        if len(tgraph.eis) == 0: 
            continue 

        ei = tgraph.eis[0]
        ea = tgraph.eas[0]
        y = tgraph.ys[0]

        for i in range(ei.size(1)):
            src,dst = ei[:,i]
            src = src.item()
            dst = dst.item()

            if not raw:
                attr = fmt_ea_row(ea[:,i])
            else: 
                attr = ea[:,i].tolist()

            csr[src][0].append(dst)
            csr[src][1].append(attr)
            csr[src][2].append(j[0])

            if y[i]: 
                csr[src][3].append(len(csr[src][0])-1)

        test += 1 
        #if test >= 10: 
        #    break 

    idxptr = [0]
    col = []
    attr = []
    label = []
    ts = []

    for i in tqdm(range(num_nodes), desc='Building torch'): 
        neigh,ef,t,y = csr[i]
        col += neigh 
        attr += ef 
        ts += t

        if y: 
            label += [y_ + idxptr[-1] for y_ in y]
        
        idxptr.append(idxptr[-1] + len(neigh))

    fname = f'../data/lanlargus_tgraph{"_raw" if raw else ""}.pt'
    torch.save(
        Data(
            x = torch.zeros((num_nodes,1)),
            idxptr = torch.tensor(idxptr), 
            col = torch.tensor(col), 
            edge_attr = torch.tensor(attr), 
            is_mal = torch.tensor(label), 
            ts = torch.tensor(ts)
        ),
        fname
    )

build_csr(raw=True)