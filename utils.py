import torch
from tqdm import tqdm 

def reindex(src, num_nodes): 
    '''
    Input: ordered source row of edge index
    Ouptut: idxptr for csr matrix
    '''
    ptr = [0]
    idx = 0 
    prog = tqdm(total=src.size(0))

    for i in range(num_nodes): 
        if idx == src.size(0)-1: 
            ptr.append(idx)
            continue 
        
        while src[idx] <= i: 
            if idx == src.size(0)-1: 
                break
            idx += 1
            prog.update()
        
        ptr.append(idx)

    return torch.tensor(ptr)

if __name__ == '__main__': 
    tr = torch.load(f'data/unsw_tgraph_tr.pt', weights_only=False)
    perturb = torch.randperm(tr.col.size(0))
    perturb = perturb[: int(perturb.size(0) * 0.25)]

    # Need to keep everything in same order, so use mask instead of index
    to_keep = torch.zeros(tr.col.size(0), dtype=torch.bool)
    to_keep[perturb] = 1

    tr.col = tr.col[to_keep]
    tr.src = tr.src[to_keep]
    tr.ts = tr.ts[to_keep]

    print("Reindexing...")
    tr.idxptr = reindex(tr.src, tr.x.size(0))
    print(f"{tr.col.size(0)} edges")

    if 'edge_attr' in tr.keys(): 
        tr.edge_attr = tr.edge_attr[to_keep]