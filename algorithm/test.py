import numpy as np
import time

from models import BaseNMF, StandardNMF, L21NMF
from processing import Processing
from evaluation import Evaluation

"""
All tests go here.
"""

SEED = 42
rng = np.random.default_rng(SEED)
V = rng.random((300, 10)) @ rng.random((10, 100))

rank = 10
TOL = 1e-5
MAX_ITER = 3000

def test_model(model : BaseNMF):
    """Correctness checks on a simple non-negative matrix."""
    W, H, history = model.W, model.H, np.array(model.loss_history)
    assert(W is not None and H is not None)

    # 1. Shapes follow the agreed interface.
    assert W.shape == (300, 10), W.shape
    assert H.shape == (10, 100), H.shape

    # 2. Multiplicative updates never create negative entries.
    assert W.min() >= 0 and H.min() >= 0
    assert np.all(np.isfinite(W)) and np.all(np.isfinite(H))

    # 3. Lee and Seung: the objective is non-increasing. Allow a tiny slack
    #    for floating point round-off introduced by EPS.
    increases = np.diff(history)
    assert np.all(increases <= 1e-9 * history[:-1]), increases.max()

    # 4. The fit actually improves on the random start.
    assert history[-1] < 0.5 * history[0]

    print('[random] all checks passed: loss {:.1f} -> {:.1f} in {} iters'.format(
        history[0], history[-1], model.n_iter_))
    print('Model is working as intended.')

def test_standard():
    standard_model = StandardNMF(rank=rank, max_iter=MAX_ITER, tol=TOL, seed=SEED).fit(V)
    test_model(standard_model)

    # Model determinism under same seed
    same_std = StandardNMF(rank=rank, max_iter=MAX_ITER, tol=TOL, seed=SEED).fit(V)
    different_std = StandardNMF(rank=rank, max_iter=MAX_ITER, tol=TOL, seed=SEED+1).fit(V)
    assert standard_model.W is not None and same_std.W is not None and different_std.W is not None
    assert np.array_equal(standard_model.W, same_std.W)
    assert not np.array_equal(standard_model.W, different_std.W)

def test_l21():
    l21_model = L21NMF(rank=rank, max_iter=MAX_ITER, tol=TOL, seed=SEED).fit(V)
    test_model(l21_model)

    # Model determinism under same seed
    same_l21 = L21NMF(rank=rank, max_iter=MAX_ITER, tol=TOL, seed=SEED).fit(V)
    diff_l21 = L21NMF(rank=rank, max_iter=MAX_ITER, tol=TOL, seed=SEED+1).fit(V)
    assert l21_model.W is not None and same_l21.W is not None and diff_l21.W is not None
    assert np.array_equal(l21_model.W, same_l21.W)
    assert not np.array_equal(l21_model.W, diff_l21.W)

def test_evaluation():
    """Simple correctness checks on evaluation metrics."""
    
    # Dummy H and Y to test accuracy and NMI.
    H = np.concatenate([np.ones((10, 50)), np.zeros((10, 50))], axis=1)
    Y = np.concatenate([np.ones((50)), np.zeros((50))])
    eval = Evaluation(V=V, Y=Y)
    
    # RRE should be zero between two equal V.
    assert eval.relative_reconstruction_error(reconstruction=V) == 0

    # Accuracy and NMI should be one for an easily separable H.
    pred = eval.assign_label(transformed=H)
    assert eval.average_accuracy(pred=pred) == 1
    assert eval.normalized_mutual_information(pred=pred) == 1
    print('Evaluation is working as intended.')

def test_processing():
    """Ensure data processing class works, on the ORL dataset at data/ORL."""
    try:
        data = Processing()
        V, Y, image_shape = data.load(dataset='orl', data_dir='data', reduce=3)
        n_classes = data.n_classes()
        V_corrupt, mask, indices = data.corrupt(block_size=6, seed=SEED)
        print('Processing is working as intended.')
        return V, Y, n_classes, V_corrupt
    except Exception:
        print('Could not locate data.')
        return None

def test_pipeline():
    """Runtime and evaluation of both models on real faces, for both the clean and corrupted datasets."""
    val = test_processing()
    if val:
        V, Y, n_classes, V_corrupt = val
    else:
        return
    standard_model = StandardNMF(rank=n_classes, max_iter=MAX_ITER, tol=TOL, seed=SEED)
    l21_model = L21NMF(rank=n_classes, max_iter=MAX_ITER, tol=TOL, seed=SEED)
    for type, model in (('STANDARD', standard_model), ('L21', l21_model)):
        for name, V_in in (('clean', V), ('b=6', V_corrupt)):
            t0 = time.time()
            model.fit(V_in)
            secs = time.time() - t0

            eval = Evaluation(V=V, Y=Y)
            rre, acc, nmi = eval.evaluate(model)
            print('[{:<8}][ORL {:>5}] iters={:4d} time={:5.1f}s rre={:.4f} acc={:.4f} nmi={:.4f}'
                  .format(type, name, model.n_iter_, secs, rre, acc, nmi))
    print('Pipeline is working as intended.')

def report_table():
    """Reproduce Table 1 of the Methods section.

    For each block size the final residual is split into occluded and
    non-occluded pixels, showing what share of the objective a small number of
    corrupted entries accounts for.
    """
    try:
        data = Processing()
        V, Y, image_shape = data.load(dataset='orl', data_dir='data', reduce=3)
        n_classes = data.n_classes()
    except Exception:
        print('Could not locate data.')
        return None

    print('\n[Table 1] Standard NMF on ORL, one block per image, seed 42')
    print('{:>3}  {:>10}  {:>12}  {:>14}  {:>12}  {:>7}'.format(
        'b', 'occluded', 'share of J', 'RMS occluded', 'RMS face', 'iters'))

    for b in (0, 3, 6, 9):
        if b:
            V_corrupt, mask, indices = data.corrupt(block_size=b, num_blocks=1, seed=SEED)
        else:
            V_corrupt, mask = V, np.zeros_like(V, dtype=bool)

        model = StandardNMF(rank=n_classes, max_iter=MAX_ITER, tol=TOL, seed=SEED).fit(V_corrupt)
        R = V_corrupt - model.reconstruct()
        share = (R[mask] ** 2).sum() / (R ** 2).sum() if mask.any() else 0.0
        rms_occ = np.sqrt((R[mask] ** 2).mean()) if mask.any() else float('nan')
        rms_face = np.sqrt((R[~mask] ** 2).mean())

        print('{:>3}  {:>9.1%}  {:>11.1%}  {:>14.3f}  {:>12.3f}  {:>7d}'.format(
            b, mask.mean(), share, rms_occ, rms_face, model.n_iter_))


if __name__ == '__main__':
    test_standard()
    test_l21()
    test_evaluation()
    test_pipeline() # also tests processing
    report_table()