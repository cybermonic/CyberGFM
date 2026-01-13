from collections import defaultdict
import gzip 

import numpy as np 
import torch 
from tqdm import tqdm 

FLOWS = '/mnt/raid10/cyber_datasets/lanl/flows.txt.gz'
FOURTEEN_DAYS = (60*60*24) * 14
SNAPSHOT = 3600

def build_flow_edge_dict(): 
    prog = tqdm(desc='Reading flows.txt', total=FOURTEEN_DAYS//SNAPSHOT)
    cnt = 0 
    ef_t = defaultdict(lambda : [[],[],[]])

    with gzip.open(FLOWS, 'rt') as f:
        line = f.readline() 

        while line: 
            ts,duration,src,src_p,dst,dst_p,proto,cnt_pkts,cnt_bytes = line.split(",")
            ts = int(ts) 

            while ts >= (cnt+1) * SNAPSHOT: 
                aggr_dict = dict() 
                for key,d in ef_t.items(): 
                    features = [
                        len(d[0]),
                        np.mean(d[0]), np.std(d[0]),
                        np.mean(d[1]), np.std(d[1]),
                        np.mean(d[2]), np.std(d[2])
                    ]
                    aggr_dict[key] = features 

                torch.save(aggr_dict, f'../data/lanl_flow_data/{cnt}.pt')
                cnt += 1 
                
                ef_t = defaultdict(lambda : [[],[],[]])
                prog.update()
            
            key = (src,dst) 
            duration = int(duration)
            cnt_pkts = int(cnt_pkts)
            cnt_bytes = int(cnt_bytes)

            ef_t[key][0].append(duration)
            ef_t[key][1].append(cnt_pkts)
            ef_t[key][2].append(cnt_bytes)

            line = f.readline() 

    return efs 

if __name__ == '__main__':
    feats = build_flow_edge_dict()