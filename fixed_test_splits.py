'''
Fixed-test-set training-window splits for LANL: one test set, many training windows.

contamination_splits.py moves the cutoff, so every D-day split is scored on a different test set
(different attacks, different attack rate) and AP cannot be compared across D. Here the test set
never moves -- every edge from day --test-day on -- and only the training window changes: the
last K days before the cutoff, [test_day - K, test_day), for each --windows K. K = test_day is
everything before the cutoff.

    lanl14argus-fixed{T}-clean{K}d     attack edges inside the window removed   (default arm)
    lanl14argus-fixed{T}-dirty{K}d     left in, unlabeled                       (--arms clean dirty)

Validation (--val):
    random  (default) 10% of the window's edges, from one seeded draw over the whole graph, so
            the windows nest: the K=7 training set is the K=4 one plus three earlier days.
            Training reaches the cutoff at every K, so K changes how much history the model
            sees and nothing else.
    tail    the last 10% of the window by time, as the D-day splits do; names end in -tail.
            With K = T this is lanl14argus-{clean,dirty}ts{T}d edge for edge, and the script
            says whether it matches those files when they exist.

Files are written by contamination_splits.build, so they have the D-day splits' layout (labels on
the test file only), plus one field: num_tokens, the vocabulary of the WHOLE graph. Without it
the sampler sizes the vocabulary from the training file's largest edge-feature token, and a short
window that happens to miss a rare token would index past the embedding table at test time.

Printed, and saved to data/fixed_test_report.json:
  * edges and attack edges per day of the whole dataset
  * per split: sizes, attack edges in the window and in test, and how much of the test set the
    window has already seen -- the share of benign / attack test edges whose (src, dst) pair is
    in the training file, and the share that touch a host that is not

Data processing only, no model; it holds the full CSR in memory (a few GB for LANL).

    python fixed_test_splits.py                                   # test day 12, K = 1 2 4 7 12
    python fixed_test_splits.py --windows 2 4 8 --arms clean dirty
    python fixed_test_splits.py --val tail --windows 12           # rebuild and check the 12-day split
'''
import argparse
import json
import os

import torch

from contamination_splits import DAY, build


def split_name(test_day, arm, k, val):
    return f'lanl14argus-fixed{test_day}-{arm}{k}d{"-tail" if val == "tail" else ""}'


def share(x):
    return float(x.float().mean()) if x.numel() else float('nan')


def pct(v):
    return '    -' if v != v else f'{v:5.1%}'


def print_by_day(ts, mal, test_day):
    day = (ts // DAY).long()
    edges = torch.bincount(day)
    attacks = torch.bincount(day[mal], minlength=edges.numel())
    print('  day      edges  attack edges')
    for d in range(edges.numel()):
        print(f'  {d:>3} {int(edges[d]):>10,} {int(attacks[d]):>13,}{"   <- test starts" if d == test_day else ""}')
    return {str(d): dict(edges=int(edges[d]), attack_edges=int(attacks[d])) for d in range(edges.numel())}


def seen_in_training(key, src, dst, n, tr, te, mal):
    '''How much of the test set the training file has already seen.'''
    y = mal[te]
    pair_seen = torch.isin(key[te], key[tr].unique())
    known = torch.zeros(n, dtype=torch.bool)
    known[src[tr]] = True
    known[dst[tr]] = True
    new_host = ~(known[src[te]] & known[dst[te]])
    return dict(benign_pair_seen=share(pair_seen[~y]), attack_pair_seen=share(pair_seen[y]),
                benign_new_host=share(new_host[~y]), attack_new_host=share(new_host[y]))


def check_against_dday(data_dir, g, test_day, arm, masks):
    '''--val tail with K = T is the D-day split; compare with its files if they are there.'''
    old = f'lanl14argus-{arm}ts{"" if test_day == 7 else f"{test_day}d"}'
    for split, m in masks.items():
        fn = f'{data_dir}/{old}_tgraph_{split}.pt'
        if not os.path.exists(fn):
            print(f'    check: {fn} not found, nothing to compare with')
            return
        o = torch.load(fn, weights_only=False)
        same = o.col.size(0) == int(m.sum()) and torch.equal(o.ts, g.ts[m]) and torch.equal(o.col, g.col[m])
        print(f'    check against {old} {split}: {"identical" if same else "DIFFERENT"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--test-day', type=int, default=12, help='the test set is every edge from this day on')
    ap.add_argument('--windows', type=int, nargs='+', default=[1, 2, 4, 7, 12],
                    help='training windows: the last K days before the test day')
    ap.add_argument('--arms', nargs='+', choices=['clean', 'dirty'], default=['clean'])
    ap.add_argument('--val', choices=['random', 'tail'], default='random')
    ap.add_argument('--seed', type=int, default=0, help='for the random validation draw')
    args = ap.parse_args()
    out = args.data

    g = torch.load(f'{out}/lanl14argus_tgraph_csr.pt', weights_only=False)
    n, n_edges = g.x.size(0), g.col.size(0)
    mal = torch.zeros(n_edges, dtype=torch.bool)
    mal[g.is_mal] = True
    label = mal.float()
    src = torch.arange(n).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
    key = src * n + g.col.long()
    num_tokens = int(n + g.edge_attr.max() + 1)
    span = (int(g.ts.max()) + 1) / DAY
    end = args.test_day * DAY

    print(f'LANL: {n_edges:,} edges over {span:.2f} days, {int(mal.sum())} attack edges, '
          f'{num_tokens:,} tokens ({n:,} hosts)')
    if end >= span * DAY:
        raise SystemExit(f'--test-day {args.test_day}: the data only covers {span:.2f} days')
    te = g.ts >= end
    if not bool((mal & te).any()):
        raise SystemExit(f'--test-day {args.test_day}: no attack edges after the cutoff, nothing to score')

    report_path = f'{out}/fixed_test_report.json'
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    report['by_day'] = print_by_day(g.ts, mal, args.test_day)
    print(f'\ntest set (every split): {int(te.sum()):,} edges from day {args.test_day} on, '
          f'{int((mal & te).sum())} attack edges, attack rate {share(mal[te]):.2e}')
    print(f'validation: {args.val}\n')

    # One draw for the whole graph, so an edge is a validation edge in every window that holds it.
    is_val = torch.rand(n_edges, generator=torch.Generator().manual_seed(args.seed)) < 0.1

    print(f'  {"split":<34} {"train":>10} {"val":>9} | {"attack edges":>12} | '
          f'{"test pair in train":>18} | {"test edge, new host":>19}')
    print(f'  {"":<34} {"":>10} {"":>9} | {"in window":>12} | {"benign":>8} {"attack":>9} | {"benign":>9} {"attack":>9}')
    for k in args.windows:
        if k > args.test_day or k < 1:
            print(f'  skipping K={k}: the window must be 1 to {args.test_day} days')
            continue
        start = end - k * DAY
        window = (g.ts >= start) & (g.ts < end)
        if args.val == 'random':
            tr, va = window & ~is_val, window & is_val
        else:
            cut = start + 0.9 * (end - start)
            tr, va = window & (g.ts < cut), window & (g.ts >= cut)

        for arm in args.arms:
            name = split_name(args.test_day, arm, k, args.val)
            keep = ~mal if arm == 'clean' else torch.ones_like(mal)
            masks = dict(tr=tr & keep, va=va & keep, te=te)
            for split, m in masks.items():
                d = build(g, m, label if split == 'te' else None)
                d.num_tokens = num_tokens
                torch.save(d, f'{out}/{name}_tgraph_{split}.pt')

            r = dict(train=int(masks['tr'].sum()), val=int(masks['va'].sum()), test=int(te.sum()),
                     attack_in_window=int((mal & window).sum()),
                     attack_train=int((mal & masks['tr']).sum()), attack_val=int((mal & masks['va']).sum()),
                     attack_test=int((mal & te).sum()), window_start_s=int(start), window_end_s=int(end),
                     val_mode=args.val, num_tokens=num_tokens)
            r.update(seen_in_training(key, src, g.col, n, masks['tr'], te, mal))
            report[name] = r
            print(f'  {name:<34} {r["train"]:>10,} {r["val"]:>9,} | {r["attack_in_window"]:>12,} | '
                  f'{pct(r["benign_pair_seen"]):>8} {pct(r["attack_pair_seen"]):>9} | '
                  f'{pct(r["benign_new_host"]):>9} {pct(r["attack_new_host"]):>9}')
            if args.val == 'tail' and k == args.test_day:
                check_against_dday(out, g, args.test_day, arm, masks)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=1)
    print(f'\nreport -> {report_path}')


if __name__ == '__main__':
    main()