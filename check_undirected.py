'''
Checks for the --undirected-walks experiment, and the numbers that say whether it can matter.

Per dataset, from the split files alone (no model, no walks):
  * the undirected walk graph is what it should be: every training edge once in each direction
    with its timestamp and features, rows sorted by time;
  * the share of validation / test edges that are scored with NO context, because their source
    has nothing to walk on -- for static walks, and for temporal walks inside the fine-tuning
    window (the context of an edge after the training cutoff comes from the last `delta` seconds
    of training). Shown for context drawn from the source's out-edges, from its in-edges, and
    from both (undirected).

With --walks it also samples context walks through common.context with the installed walk
kernels (do this in a job):
  * every hop must be a training edge carrying that edge's features;
  * a directed sampler must move one way only. Which way is measured, not assumed, and reported:
    "in-edges" means the walk reaches the source along real edges (x -> ... -> src), "out-edges"
    that it leaves the source and is then flipped. Nodes with nothing to walk on that way must
    get no context;
  * the undirected sampler must use both directions and give every node context;
  * how long the context actually is for training and test edges, how often it doubles back (an
    undirected walk never dead-ends: where a directed walk would stop, it can only return along
    the edge it came by), and how often the edge's own destination is already in its context.
    A training edge is part of the graph its context is walked on and a test edge is not, so a
    large train / test gap in that last number is a shortcut fine-tuning can learn and scoring
    cannot use;
  * temporal samplers, with a zero-width window, must return edges at exactly the query time and
    must always find the edge that is there (the walk library's window-bound fix).

    python check_undirected.py                                   # the four LANL splits
    python check_undirected.py --datasets lanl14argus-dirtyts8d
    python check_undirected.py --walks --device 0
'''
import argparse

import torch

from common import (DAY, HOUR, PAD, context, dataset_kind, load_split, make_sampler,
                    undirected_walk_graph, uses_edge_features)

LANL = ['lanl14argus', 'lanl14argus-ts', 'lanl14argus-dirtyts', 'lanl14argus-cleants']
WAYS = ('out-edges', 'in-edges', 'undirected')


def rows(g):
    return torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])


def arc_hash(s, d, ea=None, ts=None):
    '''One int64 per (src, dst[, ts][, features]); wraps around, which is fine for comparing.'''
    h = s.long() * 1000003 + d.long() * 7919
    if ts is not None:
        h = h * 31 + ts.long()
    if ea is not None:
        for j in range(ea.size(1)):
            h = h * 1000033 + ea[:, j].long() + 1
    return h


def check_graph(tr, und, edge_features):
    n = tr.x.size(0)
    s, d = rows(tr), tr.col
    ea = tr.edge_attr if edge_features else None
    us, ud = rows(und), und.col
    uea = und.edge_attr if edge_features else None
    rev = s != d

    assert torch.equal(us, und.src), 'src does not match idxptr'
    want = torch.cat([arc_hash(s, d, ea, tr.ts),
                      arc_hash(d[rev], s[rev], ea[rev] if ea is not None else None, tr.ts[rev])])
    got = arc_hash(us, ud, uea, und.ts)
    assert torch.equal(want.sort().values, got.sort().values), 'arcs differ from edges + reversed edges'
    same_row = us[1:] == us[:-1]
    assert bool((und.ts[1:][same_row] >= und.ts[:-1][same_row]).all()), 'a row is not sorted by time'
    print(f'  walk graph ok: {s.numel():,} edges -> {us.numel():,} arcs '
          f'({int((~rev).sum()):,} self-loops kept once), rows sorted by time')

    deg = tr.idxptr[1:] - tr.idxptr[:-1]
    indeg = torch.bincount(d, minlength=n)
    print(f'  nodes: {n:,} in the vocabulary, {int(((deg > 0) | (indeg > 0)).sum()):,} in the training graph; '
          f'{int(((deg == 0) & (indeg > 0)).sum()):,} have in-edges only, '
          f'{int(((deg > 0) & (indeg == 0)).sum()):,} out-edges only')


def check_context(name, tr, delta, used=None):
    '''
    used: {'static': way, 'temporal': way} as measured by check_walks, to mark what the code does.
    '''
    s, d, ts = rows(tr), tr.col, tr.ts.long()
    ts_max = int(ts.max())
    base = ts_max + 2
    arcs = {'out-edges': (s, ts), 'in-edges': (d, ts), 'undirected': (torch.cat([s, d]), torch.cat([ts, ts]))}
    n = tr.x.size(0)
    active = {k: torch.bincount(v[0], minlength=n) > 0 for k, v in arcs.items()}
    keys = {k: (v[0] * base + v[1]).sort().values for k, v in arcs.items()}

    def in_window(way, node, t):
        # as common.context: [t - delta, t] with t clamped to the end of training; a lower bound
        # of 0 means none (the walk kernel's convention)
        hi = t.long().clamp(max=ts_max)
        lo = (hi - delta).clamp(min=0)
        a = torch.searchsorted(keys[way], node * base + lo, right=False)
        b = torch.searchsorted(keys[way], node * base + hi, right=True)
        return b > a

    for split in ('va', 'te'):
        g = load_split(name, split)
        src = g.src.long() if 'src' in g.keys() else rows(g)
        mal = g.label.bool() if split == 'te' and 'label' in g.keys() else None
        if mal is not None and not mal.any():
            mal = None
        print(f'  {split} edges scored with no context'
              + (f' [attack edges, {int(mal.sum())}]' if mal is not None else '') + ':')
        for mode, label in (('static', 'static walks'), ('temporal', f'temporal, {delta:,} s window')):
            cells = []
            for way in WAYS:
                none = ~active[way][src] if mode == 'static' else ~in_window(way, src, g.ts)
                cell = f'{way} {none.float().mean():.1%}'
                if mal is not None:
                    cell += f' [{none[mal].float().mean():.1%}]'
                cells.append(cell)
            mark = f'   <- {mode} walks use {used[mode]}' if used else ''
            print(f'    {label:<26} ' + '   '.join(cells) + mark)


def hop_directions(rw, F, num_nodes, fwd, label, t=None):
    '''
    Checks every hop of a batch of context walks against the hashed training edges `fwd` (hashed
    with timestamps if t is given: every hop must then be an edge at exactly t[row]).
    The walk reads ... a, b ... left to right with the source last. Returns
    (#hops that are only an edge a -> b, #hops that are only an edge b -> a, has-context mask).
    '''
    nodes = rw[:, ::1 + F]
    n_fwd = n_bwd = 0
    for i in range(nodes.size(1) - 1):
        a, b = nodes[:, i], nodes[:, i + 1]
        real = a != PAD
        if not real.any():
            continue
        assert bool((b[real] != PAD).all()), f'{label}: a hop starts from PAD'
        f = None
        if F:
            f = rw[:, i * (1 + F) + 1:(i + 1) * (1 + F)][real] - num_nodes
            assert bool((f >= 0).all()), f'{label}: PAD features on a real hop'
        tt = t[real] if t is not None else None
        ab = torch.isin(arc_hash(a[real], b[real], f, tt), fwd)     # edge a -> b: the walk arrives along it
        ba = torch.isin(arc_hash(b[real], a[real], f, tt), fwd)     # edge b -> a: an out-edge, flipped
        what = 'a training edge at the query time' if t is not None else 'a training edge with its features'
        assert bool((ab | ba).all()), f'{label}: a hop is not {what}'
        n_fwd += int((ab & ~ba).sum())
        n_bwd += int((ba & ~ab).sum())
    return n_fwd, n_bwd, (nodes[:, :-1] != PAD).any(1)


def context_stats(smp, g, idx, delta, F, device):
    '''
    The context of edges g[idx] as fine-tuning / scoring draws it: mean hops, the share of two-hop
    stretches that go straight back where they came from (x, y, x), and the share of edges whose
    own destination is already a node of the context.
    '''
    src = (g.src if 'src' in g.keys() else rows(g))[idx].long().to(device)
    dst = g.col[idx].long().to(device)
    nodes = context(smp, src, g.ts[idx].to(device), delta, use_walk=True)[:, ::1 + F]
    real = nodes[:, :-1] != PAD
    two = real[:, :-1]                                       # x, y, z all real (x is the far end)
    back = two & (nodes[:, :-2] == nodes[:, 2:])
    seen = (nodes[:, :-1] == dst.unsqueeze(-1)).any(1)
    return (f'{real.float().sum(1).mean().item():.2f} hops, '
            f'{int(back.sum()) / max(1, int(two.sum())):.0%} of two-hop stretches double back, '
            f'destination already in the context for {seen.float().mean().item():.1%}')


def check_walks(name, tr_data, edge_features, device, delta, walk_len=4, n=4096):
    '''Returns {'static': way, 'temporal': way}: what a directed context walk is drawn from.'''
    F = tr_data.edge_attr.size(1) if edge_features else 0
    num_nodes = tr_data.x.size(0)
    s, d, ts = rows(tr_data), tr_data.col, tr_data.ts
    ea = tr_data.edge_attr if edge_features else None
    fwd = arc_hash(s, d, ea).unique().to(device)            # (u, v, features) of real edges
    fwd_t = arc_hash(s, d, ea, ts).unique().to(device)      # ... with the timestamp

    deg = tr_data.idxptr[1:] - tr_data.idxptr[:-1]
    indeg = torch.bincount(d, minlength=num_nodes)

    def pick(mask):
        idx = mask.nonzero().squeeze(-1)
        return idx[torch.randperm(idx.numel())[:n]]

    groups = {'out-only': pick((deg > 0) & (indeg == 0)),   # nothing to walk on via in-edges
              'in-only': pick((deg == 0) & (indeg > 0)),    # nothing to walk on via out-edges
              'both': pick((deg > 0) & (indeg > 0))}
    print('  start nodes sampled: ' + ', '.join(f'{k} {v.numel():,}' for k, v in groups.items()))
    timed = pick(ts > 0)                                    # edges for the zero-width window test
    te_data = load_split(name, 'te')
    tr_idx = torch.randperm(tr_data.col.size(0))[:n]
    te_idx = torch.randperm(te_data.col.size(0))[:n]

    used = {}
    for temporal in (False, True):
        mode = 'temporal' if temporal else 'static'
        for undirected in (False, True):
            label = f'{mode:<8} {"undirected" if undirected else "directed":<10}'
            smp = make_sampler(tr_data, temporal, edge_features, walk_len, n, device,
                               undirected_walks=undirected)
            n_fwd = n_bwd = 0
            has_ctx = {}
            for kind, start in groups.items():
                if start.numel() == 0:
                    continue
                start = start.to(device)
                t = torch.full_like(start, int(ts.max()))
                rw = context(smp, start, t, delta=10 ** 12, use_walk=True)
                assert rw.size(1) == (walk_len + 1) + walk_len * F, f'{label}: unexpected context width {rw.size(1)}'
                assert torch.equal(rw[:, -1], start), f'{label}: context does not end at the source'
                a, b, got = hop_directions(rw, F, num_nodes, fwd, label)
                n_fwd, n_bwd = n_fwd + a, n_bwd + b
                has_ctx[kind] = (int(got.sum()), got.numel())   # counts: a float mean is not exact on CUDA

            ctx = ', '.join(f'{k} {c:,}/{t:,}' for k, (c, t) in has_ctx.items())
            none_got = lambda k: has_ctx.get(k, (0, 0))[0] == 0
            all_got = lambda k: k not in has_ctx or has_ctx[k][0] == has_ctx[k][1]
            hops = f'{n_fwd:,} hops arrive along an edge, {n_bwd:,} are flipped out-edges'
            if undirected:
                assert n_fwd > 0 and n_bwd > 0, f'{label}: does not use both directions ({hops})'
                assert all(all_got(k) for k in has_ctx), f'{label}: some nodes got no context ({ctx})'
                way = 'undirected'
            else:
                assert not (n_fwd and n_bwd), f'{label}: hops go both ways on a directed graph ({hops})'
                way = 'in-edges' if n_fwd else 'out-edges'
                stuck, other = ('out-only', 'in-only') if way == 'in-edges' else ('in-only', 'out-only')
                assert none_got(stuck), f'{label}: {stuck} nodes got context from {way} ({ctx})'
                assert all_got(other), f'{label}: {other} nodes got no context from {way} ({ctx})'
                used[mode] = way
            print(f'  walks ok: {label} context from {way.upper()} ({hops}); nodes with context: {ctx}')
            print(f'    context as fine-tuning draws it (up to {walk_len} hops{f", {delta:,} s window" if temporal else ""}):')
            print(f'      train edges: {context_stats(smp, tr_data, tr_idx, delta, F, device)}')
            print(f'      test edges:  {context_stats(smp, te_data, te_idx, delta, F, device)}')

            if temporal and timed.numel():
                # Zero-width window at an edge's own timestamp, from the endpoint that edge gives
                # context to. Every hop must be an edge at exactly that time, and there is always
                # at least that one edge to find.
                e = timed.to(device)
                start = (d if way == 'in-edges' else s).to(device)[e]
                t = ts.to(device)[e]
                rw = context(smp, start, t, delta=0, use_walk=True)
                _, _, got = hop_directions(rw, F, num_nodes, fwd_t, f'{label} (zero-width window)', t=t)
                assert bool(got.all()), (f'{label}: {1 - got.float().mean().item():.1%} of zero-width windows missed the '
                                         f'edge at the query time -- is the installed graph_walks older than the '
                                         f'window-bound fix (temporal_rw 3728901)?')
                print(f'  window ok: {label} {e.numel():,} zero-width windows, every hop at the query time, none empty')
    return used


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--datasets', nargs='*', default=LANL)
    ap.add_argument('--walks', action='store_true', help='also sample walks (do this inside a job)')
    ap.add_argument('--device', type=int, default=0, help='for --walks; -1 for CPU')
    ap.add_argument('--delta', type=int, help='temporal context window in seconds '
                                              '(default: lp_finetune.py\'s -- 1 h LANL, 0 UNSW, 1 d OpTC)')
    args = ap.parse_args()
    torch.manual_seed(0)

    for name in args.datasets:
        print(name)
        edge_features = uses_edge_features(name)
        delta = args.delta if args.delta is not None else {'lanl': HOUR, 'unsw': 0, 'optc': DAY}[dataset_kind(name)]
        tr = load_split(name, 'tr')
        check_graph(tr, undirected_walk_graph(tr), edge_features)
        used = None
        if args.walks:
            used = check_walks(name, tr, edge_features, args.device if args.device >= 0 else 'cpu', delta)
        check_context(name, tr, delta, used)
    print('all checks passed')