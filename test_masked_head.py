'''
Check that applying the MLM head only at the labelled positions (GraphBertForMaskedLM.MASKED_ONLY)
gives the same numbers as applying it at every position, and time the two.

On a randomly initialised RWBert and random inputs, for a link-prediction batch (one [MASK] per
row) and a pretraining batch (about 30% of the tokens masked), it compares
    the loss                       modified_fwd(walk, mask, targets, attn)
    the gradient of every weight
    the scores                     masked_logits(...)  vs  modified_fwd(..., return_loss=False).logits[mask]
between MASKED_ONLY = True and False, then times a training step (forward + backward) both ways
at a LANL-like size. Differences should be at float32 rounding level (around 1e-6 or smaller).

No dataset or checkpoint is needed, and it takes a few seconds, but it runs a model, so submit it:

    echo 'source ~/miniconda3/bin/activate gfm; python test_masked_head.py' | \\
        bsub -q normal -gpu "num=1:mode=exclusive_process" -n 4 -W 00:15 -o logs/test_masked_head.out

    python test_masked_head.py --vocab 17000 --seq-len 38 --batch 1024    # the timing shape (defaults)
'''
import argparse
import time

import torch
from transformers import BertConfig

from models.gnn_bert import GNNEmbedding, RWBert

PAD, MASK = GNNEmbedding.PAD, GNNEmbedding.MASK


def make_model(vocab, hidden, layers, device):
    cfg = BertConfig(vocab_size=vocab + GNNEmbedding.OFFSET, hidden_size=hidden, num_hidden_layers=layers,
                     num_attention_heads=hidden // 64, intermediate_size=hidden * 4, num_nodes=vocab,
                     max_position_embeddings=1024)        # as common.bert_config builds it
    return RWBert(cfg).to(device)


def make_batch(kind, batch, seq_len, vocab, device, gen):
    '''Rows of random tokens, PAD-filled at the end as left_align leaves them.'''
    ids = torch.randint(0, vocab, (batch, seq_len), generator=gen)
    length = torch.randint(3, seq_len + 1, (batch,), generator=gen)      # real tokens per row
    pos = torch.arange(seq_len).expand(batch, seq_len)
    real = pos < length[:, None]
    if kind == 'lp':                 # one [MASK], in the last real position
        mask = pos == (length - 1)[:, None]
    else:                            # pretraining: about 30% of the real tokens, at least one per row
        mask = (torch.rand(batch, seq_len, generator=gen) < 0.3) & real
        mask[:, 0] |= ~mask.any(dim=1)
    targets = ids[mask]
    walk = ids.masked_fill(~real, PAD).masked_fill(mask, MASK)
    return walk.to(device), mask.to(device), targets.to(device), (walk != PAD).to(device)


def loss_and_grads(model, masked_only, walk, mask, targets, attn):
    model.MASKED_ONLY = masked_only
    loss = model.modified_fwd(walk.clone(), mask, targets, attn)     # clone: modified_fwd edits walk in place
    params = [p for p in model.parameters() if p.requires_grad]
    return loss.detach(), torch.autograd.grad(loss, params, allow_unused=True)


def compare(model, kind, batch, tol):
    walk, mask, targets, attn = batch
    model.eval()                     # no dropout, so both paths see the same hidden states
    new_loss, new_g = loss_and_grads(model, True, walk, mask, targets, attn)
    old_loss, old_g = loss_and_grads(model, False, walk, mask, targets, attn)
    pairs = [(a, b) for a, b in zip(new_g, old_g) if a is not None and b is not None]
    assert len(pairs) and all((a is None) == (b is None) for a, b in zip(new_g, old_g))
    g_diff = max(float((a - b).abs().max()) for a, b in pairs)
    g_scale = max(float(b.abs().max()) for _, b in pairs)

    with torch.no_grad():
        model.MASKED_ONLY = True
        new_s = model.masked_logits(walk.clone(), mask, attn)
        model.MASKED_ONLY = False
        old_s = model.modified_fwd(walk.clone(), mask, targets, attn, return_loss=False).logits[mask]
    model.MASKED_ONLY = True
    assert new_s.shape == old_s.shape, (new_s.shape, old_s.shape)
    s_diff, s_scale = float((new_s - old_s).abs().max()), float(old_s.abs().max())

    l_rel = abs(float(new_loss) - float(old_loss)) / abs(float(old_loss))
    ok = l_rel < tol and g_diff / g_scale < tol and s_diff / s_scale < tol
    print(f'{kind:>12}: {int(mask.sum()):,} labelled of {mask.numel():,} positions'
          f'\n              loss      {float(old_loss):.7f} (every position) vs {float(new_loss):.7f} (labelled only), relative difference {l_rel:.1e}'
          f'\n              gradients largest difference {g_diff:.1e} (largest gradient {g_scale:.1e})'
          f'\n              scores    largest difference {s_diff:.1e} (largest score {s_scale:.1e})'
          f'\n              {"SAME" if ok else "DIFFERENT"} to within {tol:g} relative', flush=True)
    return ok


def step_time(model, masked_only, batch, iters, cuda):
    walk, mask, targets, attn = batch
    model.MASKED_ONLY = masked_only
    model.train()

    def step():
        model.zero_grad(set_to_none=True)
        model.modified_fwd(walk.clone(), mask, targets, attn).backward()

    step()                           # warm-up
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(iters):
        step()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t) / iters, (torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else float('nan'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--vocab', type=int, default=17000, help='tokens, for the timing (LANL-like)')
    ap.add_argument('--seq-len', type=int, default=38, help='tokens per input, for the timing')
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--size', nargs=2, type=int, default=[128, 2], metavar=('HIDDEN', 'LAYERS'))
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--tol', type=float, default=1e-4)
    args = ap.parse_args()

    cuda = args.device >= 0 and torch.cuda.is_available()
    device = args.device if cuda else 'cpu'
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    print('== equivalence (random model, 3,000 tokens, 64 x 24 inputs) ==')
    model = make_model(3000, *args.size, device)
    ok = all([compare(model, 'link pred.', make_batch('lp', 64, 24, 3000, device, gen), args.tol),
              compare(model, 'pretraining', make_batch('mlm', 64, 24, 3000, device, gen), args.tol)])

    print(f'\n== one training step, {args.batch} x {args.seq_len} tokens, vocabulary {args.vocab:,}, '
          f'hidden {args.size[0]} x {args.size[1]} layers, on {torch.cuda.get_device_name(args.device) if cuda else "cpu"} ==')
    model = make_model(args.vocab, *args.size, device)
    for kind, name in (('lp', 'link pred.'), ('mlm', 'pretraining')):
        batch = make_batch(kind, args.batch, args.seq_len, args.vocab, device, gen)
        new_t, new_m = step_time(model, True, batch, args.iters, cuda)
        try:
            old_t, old_m = step_time(model, False, batch, args.iters, cuda)
            print(f'{name:>12}: every position {old_t * 1000:7.1f} ms, {old_m:5.1f} GB | labelled only '
                  f'{new_t * 1000:7.1f} ms, {new_m:5.1f} GB | {old_t / new_t:.1f}x faster', flush=True)
        except RuntimeError as e:    # the original computation may not fit at this size
            if cuda:
                torch.cuda.empty_cache()
            print(f'{name:>12}: labelled only {new_t * 1000:7.1f} ms, {new_m:5.1f} GB | every position failed: '
                  f'{str(e).splitlines()[0]}', flush=True)

    print('\nRESULT:', 'the two computations agree' if ok else 'MISMATCH -- do not use MASKED_ONLY until this is understood')
    raise SystemExit(0 if ok else 1)