'''
Fine-tuning speed test: where does the time go, and how much would a larger batch size save?

Runs the real fine-tuning script (lp_finetune.py, or cls_finetune.py on OpTC) for ONE epoch,
unchanged, with timers wrapped around common.evaluate and common.score_split. It measures
  startup            loading the data and the pretrained checkpoint
  training epoch     one pass over the training edges
  evaluation         split into its scoring passes: test, val, the extra "val nll" pass (LP only),
                     and the rest (scoring the validation negatives and computing the metrics)
and extrapolates to the full run the script does by default (10 epochs, with an evaluation before
training and after every epoch):
    full run = startup + first evaluation + epochs x (training epoch + evaluation)

    python ft_speedtest.py --dataset lanl14argus                  # the run's own batch sizes
    python ft_speedtest.py --dataset lanl14argus --eval-bs 8192   # same training, faster scoring
    python ft_speedtest.py --dataset lanl14argus --mini-bs 4096   # larger training batches (*)
    python ft_speedtest.py --dataset unsw --stages                # also split the training epoch
                                                                  # into sampling/forward/backward/step
    python ft_speedtest.py --report                               # table of everything measured

(*) The scripts accumulate gradients up to 1024 edges per update, so --mini-bs above 1024 means
bigger updates and fewer of them per epoch: faster, but a different training recipe, so check AP
before adopting it. --eval-bs does not change what is trained (scores move only by the randomness
of the context walks).

Walk lengths default to the paper's (4 on LANL and OpTC, 16 on UNSW). Each run appends one row to
speedtest/ft_speedtest.csv; its results CSV goes to speedtest/results/, never to results/.
--stages synchronizes the GPU after every stage, which can slow the epoch slightly, so its row is
used for the stage shares only.

Measuring trains a model: submit it with bsub (run_ft_speedtest.sh), never on the login node.
--report only reads the CSV.
'''
import argparse
import csv
import os
import runpy
import socket
import sys
import time

OUT = 'speedtest'
CSV = f'{OUT}/ft_speedtest.csv'
HEAD = {'lanl': 'lp', 'unsw': 'lp', 'optc': 'cls'}
WALK_LEN = {'lanl': 4, 'optc': 4, 'unsw': 16}
STAGES = ('samp', 'fwd', 'bwd', 'step')
FIELDS = (['dataset', 'mode', 'head', 'walk_len', 'mini_bs', 'accum', 'eval_bs', 'own_bs', 'stages', 'epochs',
           'updates_per_epoch', 'train_edges', 'val_edges', 'test_edges',
           'startup_s', 'train_epoch_s', 'eval_first_s', 'eval_s',
           'pass_test_s', 'pass_val_s', 'pass_val_nll_s', 'eval_rest_s', 'full_run_s']
          + [f'stage_{s}_s' for s in STAGES] + ['peak_gpu_gb', 'gpu', 'cores', 'host', 'when'])


# ------------------------------------------------------------------------------------- measure
def measure(args):
    import torch
    import common

    kind = common.dataset_kind(args.dataset)
    head = HEAD[kind]
    wl = args.walk_len or WALK_LEN[kind]
    cuda = torch.cuda.is_available() and args.device >= 0

    def now():
        if cuda:
            torch.cuda.synchronize()     # so queued GPU work is charged to the phase that queued it
        return time.time()

    evals = []                           # one entry per common.evaluate call
    real_evaluate, real_score_split = common.evaluate, common.score_split

    def score_split(score_fn, split, tr, bs, desc):
        t = now()
        out = real_score_split(score_fn, split, tr, bs, desc)
        if evals:
            evals[-1]['passes'][desc] = evals[-1]['passes'].get(desc, 0.0) + now() - t
        return out

    def evaluate(*a, **k):
        evals.append(dict(start=now(), passes={}))
        out = real_evaluate(*a, **k)
        evals[-1]['end'] = now()
        return out

    # The fine-tuning script does `from common import evaluate, SpeedTimer, ...` when it starts, so
    # it picks up these wrappers; evaluate() itself looks score_split up in common at call time.
    common.evaluate, common.score_split = evaluate, score_split
    if args.stages:
        class AlwaysOn(common.SpeedTimer):
            def __init__(self, enabled, device):
                super().__init__(True, device)
        common.SpeedTimer = AlwaysOn

    script = f'{head}_finetune.py'
    argv = [script, '--dataset', args.dataset, '--walk-len', str(wl), '--epochs', '1',
            '--device', str(args.device), '--pretrain-tag', args.pretrain_tag,
            '--out-dir', f'{OUT}/results', '--tag', f'_speedtest{os.getpid()}']
    if args.trw:
        argv.append('--trw')
    if args.mini_bs:
        argv += ['--mini-bs', str(args.mini_bs)]
    if args.eval_bs:
        argv += ['--eval-bs', str(args.eval_bs)]
    print('speedtest:', ' '.join(argv), flush=True)

    sys.argv = argv
    t0 = time.time()
    g = runpy.run_path(script, run_name='__main__')      # the script's globals when it finishes

    if len(evals) != 2:
        raise SystemExit(f'expected one evaluation before the epoch and one after it, saw {len(evals)}')
    first, last = evals
    dur = lambda e: e['end'] - e['start']
    epochs = args.epochs or g['ap'].get_default('epochs')
    startup, train = first['start'] - t0, last['start'] - first['end']
    p = last['passes']
    row = dict(
        dataset=args.dataset, mode='temporal' if args.trw else 'static', head=head, walk_len=wl,
        mini_bs=g['mini_bs'], accum=g['accum'], eval_bs=g['eval_bs'],
        own_bs=int(not (args.mini_bs or args.eval_bs)), stages=int(args.stages), epochs=epochs,
        updates_per_epoch=g['updates_per_epoch'], train_edges=int(g['tr'].col.size(0)),
        val_edges=int(g['va'].col.size(0)), test_edges=int(g['te'].col.size(0)),
        startup_s=startup, train_epoch_s=train, eval_first_s=dur(first), eval_s=dur(last),
        pass_test_s=p.get('test', 0.0), pass_val_s=p.get('val', 0.0), pass_val_nll_s=p.get('val nll', 0.0),
        eval_rest_s=dur(last) - sum(p.values()),
        full_run_s=startup + dur(first) + epochs * (train + dur(last)),
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0,
        gpu=torch.cuda.get_device_name(0) if cuda else 'cpu',
        cores=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
        host=socket.gethostname(), when=time.strftime('%Y-%m-%d %H:%M'))
    if args.stages:
        row.update({f'stage_{s}_s': sum(g['timer'].times[s]) for s in STAGES})

    os.makedirs(OUT, exist_ok=True)
    new = not os.path.exists(CSV)
    with open(CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in row.items()})

    print(f'\nspeedtest {args.dataset} ({row["mode"]}): mini_bs {row["mini_bs"]} x{row["accum"]}, eval_bs {row["eval_bs"]}'
          f'\n  startup {startup:.1f}s | training epoch {train:.1f}s | evaluation {dur(last):.1f}s '
          f'(first {dur(first):.1f}s) | full run of {epochs} epochs ~ {fmt(row["full_run_s"])}'
          f'\n  -> {CSV}', flush=True)


# -------------------------------------------------------------------------------------- report
def fmt(s):
    return f'{s / 60:.1f} min' if s >= 120 else f'{s:.0f} s' if s >= 10 else f'{s:.1f} s'


def report():
    if not os.path.exists(CSV):
        raise SystemExit(f'no {CSV} yet')
    latest = {}                                           # a rerun of a setting replaces its row
    for r in csv.DictReader(open(CSV)):
        latest[(r['dataset'], r['mode'], r['mini_bs'], r['eval_bs'], r['stages'])] = r
    groups = {}
    for r in latest.values():
        groups.setdefault((r['dataset'], r['mode']), []).append(r)
    F = lambda r, k: float(r[k] or 0)

    for (ds, mode), rows in groups.items():
        timed = [r for r in rows if r['stages'] == '0']
        staged = [r for r in rows if r['stages'] == '1']
        if not timed:
            continue
        b = next((r for r in timed if r.get('own_bs') == '1'), timed[0])   # the run's own batch sizes
        timed = [b] + [r for r in timed if r is not b]
        E = int(b['epochs'])
        print(f'\n== {ds} ({mode}, {b["head"]} head, walk length {b["walk_len"]}) on {b["gpu"]}, {b["cores"]} cores')
        print(f'   {int(b["train_edges"]):,} training edges | each evaluation scores {int(b["test_edges"]):,} test and '
              f'{int(b["val_edges"]):,} validation edges | full run = {E} epochs + {E + 1} evaluations')
        print(f'   {"train batch":>12} {"updates/ep":>10} {"eval batch":>10} | {"train epoch":>11} {"evaluation":>10} '
              f'{"(test":>8} {"val":>7} {"val nll":>7} {"rest)":>7} | {"full run":>9} {"speedup":>8} | {"training only":>13} {"GPU GB":>6}')
        for r in timed:
            batch = int(r['mini_bs']) * int(r['accum'])
            flag = '*' if batch != int(b['mini_bs']) * int(b['accum']) else ' '
            print(f'   {r["mini_bs"] + " x" + r["accum"] + flag:>12} {int(r["updates_per_epoch"]):>10,} {r["eval_bs"]:>10} | '
                  f'{fmt(F(r, "train_epoch_s")):>11} {fmt(F(r, "eval_s")):>10} '
                  f'{fmt(F(r, "pass_test_s")):>8} {fmt(F(r, "pass_val_s")):>7} {fmt(F(r, "pass_val_nll_s")):>7} {fmt(F(r, "eval_rest_s")):>7} | '
                  f'{fmt(F(r, "full_run_s")):>9} {F(b, "full_run_s") / F(r, "full_run_s"):>7.2f}x | '
                  f'{fmt(E * F(r, "train_epoch_s")):>13} {F(r, "peak_gpu_gb"):>6.1f}')
        if any(int(r['mini_bs']) * int(r['accum']) != int(b['mini_bs']) * int(b['accum']) for r in timed):
            print('   * a different number of edges per update than the first row: a different training recipe')
        ev, tr_ = (E + 1) * F(b, 'eval_s'), E * F(b, 'train_epoch_s')
        print(f'   first row: training {fmt(tr_)} ({tr_ / F(b, "full_run_s"):.0%} of the run), evaluation {fmt(ev)} '
              f'({ev / F(b, "full_run_s"):.0%}), startup {fmt(F(b, "startup_s"))}')
        nll = (E + 1) * F(b, 'pass_val_nll_s')
        if nll > 0:
            print(f'   the "val nll" pass is only printed, never saved: without it the first row\'s full run would be '
                  f'{fmt(F(b, "full_run_s") - nll)} ({fmt(nll)} less)')
        for r in staged:
            tot = sum(F(r, f'stage_{s}_s') for s in STAGES)
            if tot > 0:
                names = dict(samp='sampling + building inputs', fwd='forward', bwd='backward', step='optimizer step')
                print(f'   training epoch by stage (mini_bs {r["mini_bs"]}): '
                      + ', '.join(f'{names[s]} {F(r, f"stage_{s}_s") / tot:.0%}' for s in STAGES))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--report', action='store_true', help='print the table from the CSV and exit')
    ap.add_argument('--dataset')
    ap.add_argument('--trw', action='store_true', help='temporal walks')
    ap.add_argument('--walk-len', type=int, help="default: the paper's (4 on LANL and OpTC, 16 on UNSW)")
    ap.add_argument('--mini-bs', type=int, help="training batch size; default: the script's own")
    ap.add_argument('--eval-bs', type=int, help="scoring batch size; default: the script's own")
    ap.add_argument('--stages', action='store_true', help='also time sampling / forward / backward / step')
    ap.add_argument('--epochs', type=int, help="epochs to extrapolate to; default: the script's default")
    ap.add_argument('--pretrain-tag', default='_ps1', help='which pretrained checkpoint to fine-tune')
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()
    if args.report:
        report()
    elif not args.dataset:
        ap.error('--dataset is required unless --report')
    else:
        measure(args)