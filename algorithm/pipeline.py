"""
Contains the pipeline method that streamlines the process of conducting a single run.
Also contains an interactive CLI program for testing individual runs using pipeline.
"""

import time
import argparse

import numpy as np

from processing import Processing
from models import BaseNMF, ALGORITHMS
from evaluation import Evaluation
from visualisation import Run, Visualisation


def pipeline(
        label : str,
        data : Processing, 
        evaluation : Evaluation,
        model_type : str, 
        rank : int,
        max_iter : int = 1500,
        tol : float = 1e-5,
        block_size : int = 1, 
        num_blocks : int = 100, 
        fraction : float = 0.1,
        recorrupt : bool = True,
        W_init : np.ndarray | None = None,
        verbose : bool = False,
        seed : int = 42
) -> tuple[Run, BaseNMF]:
    """
    Full pipeline for a single run (corrupt data -> train -> evaluate).
    Returns results in the correct format for visualization.
    Also returns all indices for corrupted and clean images.

    Parameters
    ----------
    label : str
        Label used to identify the run.
    data : Processing
        The dataset which has already been loaded (Processing.load()).
    evaluation : Evaluation
        Holds the clean data and labels for the entire experiment.
    model_type : str
        Type of NMF model to be used ('standard' or 'l21').
    rank : int
        Factorization rank.
    max_iter : int, default = 1500
        Maximum number of optimization iterations.
    tol : float, default = 1e-5
        Early stoppage based on loss difference. 
    block_size : int, default = 1
        Size of occlusion blocks (in pixels).
    num_blocks : int, default = 100
        Number of occlusion blocks introduced per image.
    fraction : float, default = 0.1
        Fraction of images in the dataset to be corrupted in training.
    W_init : np.ndarray | None, default = None
        Trained weights used to initialize a test model.
    recorrupt : bool, default = True
        Set to 'False' if corrupted data exists and can be reused.
    verbose : bool, default = False
        Set to 'True' for terminal logging.
    seed : int, default = 42
        Random state used for data corruption and model initialization.
    """
    
    if model_type not in ALGORITHMS.keys(): raise RuntimeError("Invalid NMF algorithm.")
    V, Y, image_shape = data.get_loaded()
    V_corrupt, masks, corrupt_idx = data.get_corrupted() if not recorrupt else data.corrupt(
        seed=seed, block_size=block_size, num_blocks=num_blocks, fraction=fraction)
    clean_idx = list(np.setdiff1d(np.arange(V.shape[1]), corrupt_idx))
    n_classes = data.n_classes()

    print(f"Running {model_type} algorithm...")
    start_time = time.perf_counter()
    if W_init is None:
        model = ALGORITHMS[model_type](seed=seed, rank=rank, max_iter=max_iter, tol=tol).fit(V_corrupt)
    else:
        model = ALGORITHMS[model_type](seed=seed, rank=rank, max_iter=max_iter, tol=tol, W_init=W_init).fit(V_corrupt, freeze=True)
    elapsed_time = time.perf_counter() - start_time
    print("Run completed.")

    reconstruction = model.reconstruct()
    loss_history = model.get_history()
    rre, acc, nmi = evaluation.evaluate(model=model)

    if verbose:
        print(f"V shape: {V.shape}")
        print(f"Number of subjects: {n_classes}")
        print(f"Resized image shape: {image_shape}")
        print(f"Factorisation rank: {rank}")
        print(f"Corrupted block size: {block_size}")
        print(f"Number of corrupted blocks: {num_blocks}")
        print(f"Corrupted image fraction: {fraction:.4f}")
        print(f"Iterations: {len(loss_history) - 1}")
        print(f"Runtime: {elapsed_time:.3f} seconds")
        print(f"RRE against clean data: {rre:.6f}")
        print(f"Accuracy against clean data: {acc:.6f}")
        print(f"NMI against clean data: {nmi:.6f}")
    print("\n")

    params = {
        'model_type' : model_type,
        'rank' : rank,
        'max_iter' : max_iter,
        'tol' : tol,
        'block_size' : block_size,
        'num_blocks' : num_blocks,
        'fraction' : fraction,
    }

    results = {
        'runtime' : elapsed_time,
        'rre' : rre,
        'acc' : acc,
        'nmi' : nmi
    }

    images = {
        'V' : V,
        'V_corrupt' : V_corrupt,
        'reconstruction' : reconstruction
    }

    return Run(label=label, params=params, results=results, images=images, 
               history=loss_history, corrupt_idx=corrupt_idx, clean_idx=clean_idx), model


#---------- for testing the pipeline ----------


def _ask(prompt, default = None, cast : type = str, validate = None, allow_none = False):
    """Prompt until the user gives a valid value. Enter accepts the default."""
    shown = "auto" if default is None else default
    while True:
        raw = input(f"{prompt} [{shown}]: ").strip()
        if raw == "":
            return default
        if allow_none and raw.lower() in ("none", "auto"):
            return None
        try:
            value = cast(raw)
        except ValueError:
            print(f"  Invalid input: expected {cast.__name__}.")
            continue
        if validate and not validate(value):
            print("  Value out of allowed range.")
            continue
        return value


def _ask_choice(prompt, choices, default):
    """Prompt for one of a fixed set of choices (by name or number)."""
    options = ", ".join(f"{i}) {c}" for i, c in enumerate(choices, 1))
    while True:
        raw = input(f"{prompt} ({options}) [{default}]: ").strip().lower()
        if raw == "":
            return default
        if raw in choices:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  Please choose one of: {', '.join(choices)}")


def parse_arguments():
    """Prompts user for model settings."""
    print("=== NMF configuration (press Enter to accept defaults) ===\n")

    model_type = _ask_choice("Model type", ("standard", "l21"), "l21")
    dataset = _ask_choice("Dataset", ("orl", "yaleb"), "orl")

    data_dir = _ask(
        "Data directory (containing ORL and CroppedYaleB folders)",
        default=None, allow_none=True,
    )

    reduce_default = 3 if dataset == "orl" else 4
    reduce = _ask(
        "Image downsampling factor", reduce_default, int, lambda v: v > 0
    )

    rank = _ask(
        "Factorisation rank (Enter = number of subjects)",
        default=None, cast=int, validate=lambda v: v > 0, allow_none=True,
    )

    block_size = _ask("Block size", 1, int, lambda v: v > 0)
    num_blocks = _ask("Number of blocks", 100, int, lambda v: v > 0)
    fraction = _ask("Fraction", 0.1, float, lambda v: 0 < v <= 1)
    max_iter = _ask("Max iterations", 3000, int, lambda v: v > 0)
    tol = _ask("Tolerance", 1e-7, float, lambda v: v > 0)
    seed = _ask("Random seed", 42, int)
    output_dir = _ask("Output directory", "results/test_run", str)

    args = argparse.Namespace(
        model_type=model_type,
        dataset=dataset,
        data_dir=data_dir,
        reduce=reduce,
        rank=rank,
        block_size=block_size,
        num_blocks=num_blocks,
        fraction=fraction,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        output_dir=output_dir,
    )

    print("\nConfiguration:")
    for k, v in vars(args).items():
        print(f"  {k:12s} = {v}")
    if input("\nProceed? [Y/n]: ").strip().lower() in ("n", "no"):
        raise SystemExit("Aborted.")
    return args


def test_run():
    args = parse_arguments()

    data = Processing()
    V, Y, image_shape = data.load(args.dataset, args.data_dir, args.reduce)
    evaluation = Evaluation(V, Y, args.seed)
    rank = data.n_classes() if args.rank is None else args.rank

    run, _ = pipeline(
        label='test_run',
        data=data,
        evaluation=evaluation,
        model_type=args.model_type,
        rank=rank,
        max_iter=args.max_iter,
        tol=args.tol,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        fraction=args.fraction,
        recorrupt=True,
        verbose=True,
        seed=args.seed
    )
    experiments = {'test_run' : [[run]]}

    visuals = Visualisation(image_shape=image_shape, output_dir=args.output_dir)
    visuals.plot_loss(experiments)
    visuals.save_details(experiments)
    visuals.save_images(experiments)

if __name__ == '__main__':
    test_run()