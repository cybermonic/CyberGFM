from collections import defaultdict
from dateutil import parser
from tqdm import tqdm
from joblib import Parallel, delayed

HOME = '/mnt/raid10/cyber_datasets/OpTC/flow_split_uq'
OUT = '/mnt/raid10/cyber_datasets/OpTC'
NFILES = 212

# Based on timecodes in the readme at the IEEE dataset repo: 
# https://ieee-dataport.org/open-access/operationally-transparent-cyber-optc
MAL_DAYS = [
    (
        parser.parse('2019-09-23T13:00:00.000').timestamp(),
        parser.parse('2019-09-23T20:00:00.000').timestamp()
    ),
    (   # Ran overnight 
        parser.parse('2019-09-24T10:00:00.000-4:00').timestamp(),
        parser.parse('2019-09-25T13:00:00.000').timestamp()
    ),
    (
        parser.parse('2019-09-25T13:00:00.000').timestamp(),
        parser.parse('2019-09-25T18:40:00.000').timestamp()
    )
]

def get_mal_day(ts): 
    if ts < MAL_DAYS[0][0]: 
        return -1
    elif ts < MAL_DAYS[0][1]: 
        return 0 
    elif ts < MAL_DAYS[1][0]: 
        return -1
    elif ts < MAL_DAYS[1][1]:
        return 1
    # Day 3 starts immidiately after day 2
    elif ts < MAL_DAYS[2][1]: 
        return 2
    else: 
        return -1

def parse_hostname(host): 
    if host.startswith('Sys'): 
        host = host.split('.')[0]
        host = int(host[-4:])
    elif host == 'DC1': 
        host = 1000 
    elif host.endswith('.255'):
        return None 
    
    # Not sure what to do with unidentified IPs. 
    # I think they're kerb servers and other important 
    # servers on the network... but only if they're in 
    # the 142. IP range. Otherwise, sometimes they're LAN IPs
    # Leave as strings for now, then map to uuids later
    return host 

def build_one(fid, redlog): 
    try:
        f = open(f'{HOME}/{fid}.csv')
    except FileNotFoundError: 
        return []
    
    lines = []
    line = f.readline() 
    while line: 
        ts,src,sp,dst,dp,usrname = line.split(',')
        ts = float(ts) 
        src = parse_hostname(src) 
        dst = parse_hostname(dst) 

        sp = int(sp) 
        dp = int(dp)
 
        if ts > 1569346918.0 and ts < 1569346918.0 + 10: 
            pass

        if sp < dp: 
            # Src is the service that dst was accessing
            # so reverse edge direction (may not be important later)
            port = sp 
            tmp = src 
            src = dst 
            dst = tmp  
        else: 
            port = dp 

        # Sometimes happens for NetBIOS connections. 
        # Not sure who is host and who is client here, so just make
        # bidirectional 
        bi = False
        if sp == dp: 
            bi = True

        md = get_mal_day(ts)

        if md >= 0:
            pass

        red = redlog[md]
        if red[(src,dst)] <= ts or (bi and red[(dst,src)] <= ts): 
            label = 1
            if src != 1000 and dst != 1000:
                print(line)
        else:
            label = 0 
        
        edge = [ts, src, dst, port, usrname.strip(), label]
        lines.append(edge)

        line = f.readline()

    f.close()
    lines.sort(key=lambda x : x[0])
    return lines 

def build_one_compressed(fid, redlog): 
    try:
        f = open(f'{HOME}/{fid}.csv')
    except FileNotFoundError: 
        return []
    
    lines = []
    line = f.readline() 
    while line: 
        first,last,src,port,usr,img,dst,label = line.split(',')
        ts_f = float(first) 
        ts_l = float(last)
        src = parse_hostname(src) 
        dst = parse_hostname(dst) 
        label = int(label)

        # Skip system events
        if usr == '' or img == '' or src is None or dst is None: 
            line = f.readline()
            continue

        md = get_mal_day(ts_f)

        if md >= 0:
            pass

        # Catches a few labels missed in the labels from the github repo
        red = redlog[md]
        if red[(src,dst)] <= ts_l and img.lower() != 'python.exe': 
            label = 1
        
        if label:
            print(line)
        edge = [ts_f, src, dst, port, usr, img, label]
        lines.append(edge)

        line = f.readline()

    f.close()
    lines.sort(key=lambda x : x[0])
    return lines 

def build_labeled_dataset(): 
    redlog = [defaultdict(lambda : float('inf')) for _ in range(4)] 
    with open('optc_labels.csv', 'r') as f:
        log = f.read().split('\n')

    def parse_hostname(h): 
        if h == 'DC1': 
            return 1000
        return int(h)
    
    for l in log:
        attk_day,src,dst,time = l.split(',')
        src = parse_hostname(src)
        dst = parse_hostname(dst)
        # Give all times 5m of wiggle room since some events don't line up exactly w 
        # what's in the redlog (e.g. pivot from 201 to 402 happens abt 70s earlier than reported)
        redlog[int(attk_day)][(src,dst)] = parser.parse(time[:-1]).timestamp() - 60*5

    unk_map = dict()
    def get_or_add(ip): 
        if (uuid := unk_map.get(ip)) is None: 
            offset = 1001 
            if not ip.startswith('142'): 
                offset += 100 
            
            uuid = offset + len(unk_map)
            unk_map[ip] = uuid 
            print(f'{ip}: {uuid}')
        
        return uuid 

    out = open(f'{OUT}/full_graph.csv','w+')
    for i in tqdm(range(NFILES)): 
        edges = build_one(i, redlog)    
        for edge in edges: 
            if type(edge[1]) != int: 
                edge[1] = get_or_add(edge[1])
            if type(edge[2]) != int: 
                edge[2] = get_or_add(edge[2])
        
            out.write(','.join([str(e) for e in edge]) + '\n')
    out.close()

def build_labeled_dataset_compressed(): 
    redlog = [defaultdict(lambda : float('inf')) for _ in range(4)] 
    with open('optc_labels.csv', 'r') as f:
        log = f.read().split('\n')

    def parse_hostname(h): 
        if h == 'DC1': 
            return 1000
        return int(h)
    
    for l in log:
        attk_day,src,dst,time = l.split(',')
        src = parse_hostname(src)
        dst = parse_hostname(dst)
        # Give all times 5m of wiggle room since some events don't line up exactly w 
        # what's in the redlog (e.g. pivot from 201 to 402 happens abt 70s earlier than reported)
        redlog[int(attk_day)][(src,dst)] = parser.parse(time[:-1]).timestamp() - 60*15

    unk_map = dict()
    def get_or_add(ip): 
        if (uuid := unk_map.get(ip)) is None: 
            offset = 1001 
            if not ip.startswith('142'): 
                offset += 100 
            
            uuid = offset + len(unk_map)
            unk_map[ip] = uuid 
            print(f'{ip}: {uuid}')
        
        return uuid 

    out = open(f'{OUT}/full_graph.csv','w+')
    for i in tqdm(range(NFILES)): 
        edges = build_one_compressed(i, redlog)    
        for edge in edges: 
            if type(edge[1]) != int: 
                edge[1] = get_or_add(edge[1])
            if type(edge[2]) != int: 
                edge[2] = get_or_add(edge[2])
        
            out.write(','.join([str(e) for e in edge]) + '\n')
    out.close()


if __name__ == '__main__':
    build_labeled_dataset_compressed()