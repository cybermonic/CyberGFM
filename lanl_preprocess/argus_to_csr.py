from collections import defaultdict
from math import log10

from joblib import Parallel, delayed
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from load_utils import id_edge_a, std_edge_a, std_edge_w
from load_lanl import load_partial_lanl, TIMES

# Same config as reported in paper
END = TIMES['test']
DELTA = 60**2

def build_csr():
    '''
    Builds the tokenized and the raw file in one pass; both share the same edge order.

    Tokenized file (lanl14argus_tgraph_csr.pt), for CyberGFM:
        edge_attr    tokens of the unscaled attributes, plus the edge weight as the last column
        edge_weight  the loader's default `standardized` weight (z-score + sigmoid, per hour)

    Raw file (lanl14argus_tgraph_raw_csr.pt), for Argus and the other baselines -- exactly what
    Argus's own loader feeds its model (load_lanl_dist defaults):
        edge_attr    std_edge_a of the attributes, per hour (the 0.5 / 0.73 / 0.88 values)
        edge_weight  std_edge_w of the per-hour event count
    '''
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
            #
            if ef < 100:
                return get_or_add(ef)
            else:
                return get_or_add(f'10^{int(log10(ef))}')
            #
        else:
            ef = int(ef*100)
            return get_or_add(f'0.{ef}')

    def fmt_ea_row(row):
        return [ea_to_token(ea) for ea in row]

    num_nodes = 0
    # dst, tokenized attr, ts, mal idx, edge weight, Argus attr, Argus edge weight
    csr = defaultdict(lambda : [[],[],[],[],[],[],[]])
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
        ew = tgraph.ews[0]
        y = tgraph.ys[0]

        # Argus's own normalisation, computed over the whole hour before any split
        std_ea = std_edge_a([ea])[0]
        std_ew = std_edge_w([tgraph.cnt[0]])[0]

        for i in range(ei.size(1)):
            src,dst = ei[:,i]
            src = src.item()
            dst = dst.item()

            csr[src][0].append(dst)
            # edge weight goes on the end as one more token column
            csr[src][1].append(fmt_ea_row(ea[:,i]) + [ea_to_token(ew[i].item())])
            csr[src][5].append(std_ea[:,i].tolist())
            csr[src][2].append(j[0])
            csr[src][4].append(ew[i].item())
            csr[src][6].append(std_ew[i].item())

            if y[i]:
                csr[src][3].append(len(csr[src][0])-1)

        test += 1
        #if test >= 10:
        #    break

    idxptr = [0]
    col = []
    attr = []
    raw_attr = []
    label = []
    ts = []
    ew = []
    raw_ew = []

    for i in tqdm(range(num_nodes), desc='Building torch'):
        neigh,ef,t,y,w,raw_ef,raw_w = csr[i]
        col += neigh
        attr += ef
        raw_attr += raw_ef
        ts += t
        ew += w
        raw_ew += raw_w

        if y:
            label += [y_ + idxptr[-1] for y_ in y]

        idxptr.append(idxptr[-1] + len(neigh))

    idxptr = torch.tensor(idxptr)
    col = torch.tensor(col)
    ew = torch.tensor(ew)
    raw_ew = torch.tensor(raw_ew)
    label = torch.tensor(label)
    ts = torch.tensor(ts)

    for suffix, ea, w in (('', attr, ew), ('_raw', raw_attr, raw_ew)):
        torch.save(
            Data(
                x = torch.zeros((num_nodes,1)),
                idxptr = idxptr,
                col = col,
                edge_attr = torch.tensor(ea),
                edge_weight = w,
                is_mal = label,
                ts = ts
            ),
            f'../data/lanl14argus_tgraph{suffix}_csr.pt'
        )

build_csr()