from collections import defaultdict
from dateutil import parser
from tqdm import tqdm
from joblib import Parallel, delayed

HOME = '/mnt/raid10/cyber_datasets/OpTC/flow_start'
OUT = '/mnt/raid10/cyber_datasets/OpTC'
N_FILES = 12672

BENIGN = (
    753,
    parser.parse('2019-09-17T08:00:00.000-04:00').timestamp(),
    parser.parse('2019-09-20T18:00:00.000-04:00').timestamp()
)

MAL_DAYS = [
    (
        9393,
        parser.parse('2019-09-23T08:00:00.000-04:00').timestamp(),
        parser.parse('2019-09-23T18:00:00.000-04:00').timestamp()
    ),
    (
        10833,
        parser.parse('2019-09-24T08:00:00.000-04:00').timestamp(),
        parser.parse('2019-09-24T18:00:00.000-04:00').timestamp()
    ),
    (
        12273,
        parser.parse('2019-09-25T08:00:00.000-04:00').timestamp(),
        parser.parse('2019-09-25T18:00:00.000-04:00').timestamp()
    )
]

def build_compressed_benign(delta=60): 
    start_f,st,en = BENIGN
    done = False
    prog = tqdm()

    out = open(f'{OUT}/benign_compressed.csv', 'w+')
    edge_set = defaultdict(lambda : 0)
    next_dump = delta 

    for i in range(start_f, N_FILES):
        try:
            f = open(f'{HOME}/{i}.csv', 'r')
        except FileNotFoundError:
            continue

        if i >= next_dump: 
            t = next_dump - delta 
            for (src,dst),wgt in edge_set.items(): 
                out.write(f'{src},{dst},{wgt},{t}\n')

            edge_set = defaultdict(lambda : 0)
            
            # Cant just add delta. May have jumps of several hours
            while i > next_dump: 
                next_dump += delta 

        line = f.readline()
        while line:
            prog.update()
            ts,src,dst,_ = line.split(',')

            ts = float(ts)
            if ts < st:
                line = f.readline()
                continue
            if ts > en:
                done = True
                break

            edge_set[(src,dst)] += 1
            line = f.readline()

        f.close()

        if done:
            break

    out.close()
    prog.close()

def build_benign():
    start_f,st,en = BENIGN
    done = False
    prog = tqdm()

    out = open(f'{OUT}/benign.csv', 'w+')
    for i in range(start_f, N_FILES):
        try:
            f = open(f'{HOME}/{i}.csv', 'r')
        except FileNotFoundError:
            continue

        line = f.readline()

        while line:
            prog.update()
            ts,src,dst,_ = line.split(',')

            ts = float(ts)
            if ts < st:
                line = f.readline()
                continue
            if ts > en:
                done = True
                break

            out.write(f'{src},{dst},{ts}\n')
            line = f.readline()

        f.close()

        if done:
            break

    out.close()
    prog.close()

def build_compressed_attack(day, delta=60): 
    start_f,st,en = MAL_DAYS[day]
    done = False
    prog = tqdm(desc=f'Day {day+1}')

    redlog = dict()
    with open('optc_labels.csv', 'r') as f:
        log = f.read().split('\n')

    for l in log:
        attk_day,host,time = l.split(',')
        if int(attk_day) == day:
            redlog[host] = parser.parse(time[:-1]).timestamp()

    edge_set = defaultdict(lambda : 0)
    red_edges = set()
    next_dump = start_f + delta 

    out = open(f'{OUT}/attack_day-{day+1}_compressed.csv', 'w+')
    for i in range(start_f, N_FILES):
        try:
            f = open(f'{HOME}/{i}.csv', 'r')
        except FileNotFoundError:
            continue

        if i >= next_dump: 
            for (src,dst),wgt in edge_set.items(): 
                t = next_dump - delta 
                label = 1 if (src,dst) in red_edges else 0
                out.write(f'{src},{dst},{wgt},{t},{label}\n')

            edge_set = defaultdict(lambda : 0)
            red_edges = set() 
            
            # Cant just add delta. May have jumps of several hours
            while i > next_dump: 
                next_dump += delta 

        line = f.readline()
        while line:
            prog.update()
            ts,src,dst,_ = line.split(',')

            ts = float(ts)
            if ts < st:
                line = f.readline()
                continue
            if ts > en:
                done = True
                break

            if redlog.get(src, float('inf')) <= ts:
                red_edges.add((src,dst))
        
            edge_set[(src,dst)] += 1
            line = f.readline()

        f.close()

        if done:
            break

    out.close()
    prog.close()

def build_attack(day):
    start_f,st,en = MAL_DAYS[day]
    done = False
    prog = tqdm(desc=f'Day {day+1}')

    redlog = dict()
    with open('optc_labels.csv', 'r') as f:
        log = f.read().split('\n')

    for l in log:
        attk_day,host,time = l.split(',')
        if int(attk_day) == day:
            redlog[host] = parser.parse(time[:-1]).timestamp()

    out = open(f'{OUT}/attack_day-{day+1}.csv', 'w+')
    for i in range(start_f, N_FILES):
        try:
            f = open(f'{HOME}/{i}.csv', 'r')
        except FileNotFoundError:
            continue

        line = f.readline()

        while line:
            prog.update()
            ts,src,dst,_ = line.split(',')

            ts = float(ts)
            if ts < st:
                line = f.readline()
                continue
            if ts > en:
                done = True
                break

            if redlog.get(src, float('inf')) <= ts:
                label = 1
            else:
                label = 0

            out.write(f'{src},{dst},{ts},{label}\n')
            line = f.readline()

        f.close()

        if done:
            break

    out.close()
    prog.close()

def build_all(i):
    if i == 0:
        build_compressed_benign()
    else:
        build_compressed_attack(i-1)

if __name__ == '__main__':
    Parallel(prefer='processes', n_jobs=4)(
        delayed(build_all)(i) for i in range(4)
    )