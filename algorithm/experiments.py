"""
All experiments are designed and ran here.
Details can be found within the report.
"""

from collections import defaultdict

from sklearn.model_selection import train_test_split

from processing import Processing
from evaluation import Evaluation
from pipeline import pipeline
from visualisation import Visualisation


# sequence of seeds to use for multiple runs
SEEDS = [4, 5, 3, 2, 8]


# Compare rank settings when all images are clean
def compare_rank_clean():
    data = Processing()
    V, Y, image_shape = data.load('orl', 'data')

    experiments = defaultdict(list)
    for r in range(20, 100, 10):
        standard_runs = []
        l21_runs = []

        for seed in SEEDS:
            evaluation = Evaluation(V, Y, seed=seed)

            run, _ = pipeline(
                label=f'rank={r}',
                data=data,
                evaluation=evaluation,
                model_type='standard',
                fraction=0,
                rank=r,
                seed=seed
            )
            standard_runs.append(run)

            run, _ = pipeline(
                label=f'rank={r}',
                data=data,
                evaluation=evaluation,
                model_type='l21',
                fraction=0,
                rank=r,
                recorrupt=False,
                seed=seed
            )
            l21_runs.append(run)

            experiments['standard'].append(standard_runs)
            experiments['l21'].append(l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_rank_clean')
    visuals.plot_loss(experiments)
    visuals.plot_metrics(experiments, 'rank')
    visuals.save_details(experiments)
    visuals.save_images(experiments)


# Compare rank settings when 10% of images are corrupted 
def compare_rank_corrupt():
    data = Processing()
    V, Y, image_shape = data.load('orl', 'data')

    experiments = defaultdict(list)
    for r in range(20, 100, 10):
        standard_runs = []
        l21_runs = []

        for seed in SEEDS:
            evaluation = Evaluation(V, Y, seed=seed)

            run, _ = pipeline(
                label=f'rank={r}',
                data=data,
                evaluation=evaluation,
                model_type='standard',
                rank=r,
                seed=seed
            )
            standard_runs.append(run)

            run, _ = pipeline(
                label=f'rank={r}',
                data=data,
                evaluation=evaluation,
                model_type='l21',
                rank=r,
                recorrupt=False,
                seed=seed
            )
            l21_runs.append(run)

            experiments['standard'].append(standard_runs)
            experiments['l21'].append(l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_rank_corrupt')
    visuals.plot_loss(experiments)
    visuals.plot_metrics(experiments, 'rank')
    visuals.save_details(experiments)
    visuals.save_images(experiments)


# Compare different levels of corruption (num_blocks) at 10% corruption rate
def compare_num_blocks():
    data = Processing()
    V, Y, image_shape = data.load('orl', 'data')
    rank = data.n_classes()

    experiments = defaultdict(list)
    for n in range(0, 400, 50):
        standard_runs = []
        l21_runs = []

        for seed in SEEDS:
            evaluation = Evaluation(V, Y, seed=seed)

            run, _ = pipeline(
                label=f'num_blocks={n}',
                data=data,
                evaluation=evaluation,
                model_type='standard',
                rank=rank,
                num_blocks=n,
                seed=seed
            )
            standard_runs.append(run)

            run, _ = pipeline(
                label=f'num_blocks={n}',
                data=data,
                evaluation=evaluation,
                model_type='l21',
                rank=rank,
                num_blocks=n,
                recorrupt=False,
                seed=seed
            )
            l21_runs.append(run)

            experiments['standard'].append(standard_runs)
            experiments['l21'].append(l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_num_blocks')
    visuals.plot_loss(experiments)
    visuals.plot_metrics(experiments, 'num_blocks')
    visuals.save_details(experiments)
    visuals.save_images(experiments)


# Compare different levels of corruption (block_size) at 10% corruption rate
def compare_block_size():
    data = Processing()
    V, Y, image_shape = data.load('orl', 'data')
    rank = data.n_classes()

    experiments = defaultdict(list)
    for b in range(0, 16, 2):
        standard_runs = []
        l21_runs = []

        for seed in SEEDS:
            evaluation = Evaluation(V, Y, seed=seed)

            run, _ = pipeline(
                label=f'block_size={b}',
                data=data,
                evaluation=evaluation,
                model_type='standard',
                rank=rank,
                num_blocks=1,
                block_size=b,
                seed=seed
            )
            standard_runs.append(run)

            run, _ = pipeline(
                label=f'block_size={b}',
                data=data,
                evaluation=evaluation,
                model_type='l21',
                rank=rank,
                num_blocks=1,
                block_size=b,
                recorrupt=False,
                seed=seed
            )
            l21_runs.append(run)

            experiments['standard'].append(standard_runs)
            experiments['l21'].append(l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_block_size')
    visuals.plot_loss(experiments)
    visuals.plot_metrics(experiments, 'block_size')
    visuals.save_details(experiments)
    visuals.save_images(experiments)


# Compare different (num_blocks, block_size) settings at the same per image corruption level
def compare_bsize_nblocks():
    data = Processing()
    V, Y, image_shape = data.load('orl', 'data')
    rank = data.n_classes()

    # corrupted pixels = 256 = (b ** 2) * n
    b = [1, 2, 4, 8, 16]
    n = [256, 64, 16, 4, 1]

    experiments = defaultdict(list)
    for i in range(5):
        standard_runs = []
        l21_runs = []

        for seed in SEEDS:
            evaluation = Evaluation(V, Y, seed=seed)

            run, _ = pipeline(
                label=f'bsize={b[i]} nblocks={n[i]}',
                data=data,
                evaluation=evaluation,
                model_type='standard',
                rank=rank,
                block_size=b[i],
                num_blocks=n[i],
                seed=seed
            )
            standard_runs.append(run)

            run, _ = pipeline(
                label=f'bsize={b[i]} nblocks={n[i]}',
                data=data,
                evaluation=evaluation,
                model_type='l21',
                rank=rank,
                block_size=b[i],
                num_blocks=n[i],
                recorrupt=False,
                seed=seed
            )
            l21_runs.append(run)

            experiments['standard'].append(standard_runs)
            experiments['l21'].append(l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_bsize_nblocks')
    visuals.plot_loss(experiments)
    visuals.plot_metrics(experiments, 'block_size')
    visuals.save_details(experiments)
    visuals.save_images(experiments)


# Compare different fractions of corrupted images (from 0 to 1)
def compare_corrupted_fraction():
    data = Processing()
    V, Y, image_shape = data.load('orl', 'data')
    rank = data.n_classes()

    experiments = defaultdict(list)
    for f in range(0, 110, 10):
        f = f / 100
        standard_runs = []
        l21_runs = []

        for seed in SEEDS:
            evaluation = Evaluation(V, Y, seed=seed)

            run, _ = pipeline(
                label=f'fraction={f}',
                data=data,
                evaluation=evaluation,
                model_type='standard',
                rank=rank,
                fraction=f,
                seed=seed
            )
            standard_runs.append(run)

            run, _ = pipeline(
                label=f'fraction={f}',
                data=data,
                evaluation=evaluation,
                model_type='l21',
                rank=rank,
                fraction=f,
                recorrupt=False,
                seed=seed
            )
            l21_runs.append(run)

            experiments['standard'].append(standard_runs)
            experiments['l21'].append(l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_corrupted_fraction')
    visuals.plot_loss(experiments)
    visuals.plot_metrics(experiments, 'fraction')
    visuals.save_details(experiments)
    visuals.save_images(experiments)


# Compare performance using a split of corrupt train data and clean test data
def corrupt_clean_split():
    data_train = Processing()
    data_test = Processing()
    V, Y, image_shape = data_train.load('yaleb', 'data')
    rank = data_train.n_classes()

    train_experiments = defaultdict(list)
    test_experiments = defaultdict(list)
    for f in range(0, 40, 5):
        f = f / 100
        train_standard_runs = []
        train_l21_runs = []
        test_standard_runs = []
        test_l21_runs = []

        for seed in SEEDS:
            V_train, V_test, Y_train, Y_test = train_test_split(V.T, Y, test_size=0.3, random_state=seed)
            data_train.load_data(V_train.T, Y_train, image_shape)
            data_test.load_data(V_test.T, Y_test, image_shape)
            evaluation_train = Evaluation(V_train.T, Y_train, seed=seed)
            evaluation_test = Evaluation(V_test.T, Y_test, seed=seed)

            run, standard_model = pipeline(
                label=f'fraction={f}',
                data=data_train,
                evaluation=evaluation_train,
                model_type='standard',
                rank=rank,
                max_iter=900,
                fraction=f,
                seed=seed
            )
            train_standard_runs.append(run)

            run, l21_model = pipeline(
                label=f'fraction={f}',
                data=data_train,
                evaluation=evaluation_train,
                model_type='l21',
                rank=rank,
                max_iter=900,
                fraction=f,
                recorrupt=False,
                seed=seed
            )
            train_l21_runs.append(run)

            # Lock the corruption to 0
            data_test.corrupt(fraction=0, seed=seed)
            standard_W, _ = standard_model.transform()
            l21_W, _ = l21_model.transform()

            run, _ = pipeline(
                label=f'fraction={f}',
                data=data_test,
                evaluation=evaluation_test,
                model_type='standard',
                rank=rank,
                W_init=standard_W,
                fraction=f,
                recorrupt=False,
                seed=seed
            )
            test_standard_runs.append(run)

            run, _ = pipeline(
                label=f'fraction={f}',
                data=data_test,
                evaluation=evaluation_test,
                model_type='l21',
                rank=rank,
                W_init=l21_W,
                fraction=f,
                recorrupt=False,
                seed=seed
            )
            test_l21_runs.append(run)

            train_experiments['standard'].append(train_standard_runs)
            train_experiments['l21'].append(train_l21_runs)
            test_experiments['standard'].append(test_standard_runs)
            test_experiments['l21'].append(test_l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/corrupt_clean_split_train')
    visuals.plot_loss(train_experiments)
    visuals.plot_metrics(train_experiments, 'fraction')
    visuals.save_details(train_experiments)
    visuals.save_images(train_experiments)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/corrupt_clean_split_test')
    visuals.plot_loss(test_experiments)
    visuals.plot_metrics(test_experiments, 'fraction')
    visuals.save_details(test_experiments)
    visuals.save_images(test_experiments)


# Compare performance using a split of clean train data and corrupt test data
def clean_corrupt_split():
    data_train = Processing()
    data_test = Processing()
    V, Y, image_shape = data_train.load('yaleb', 'data')
    rank = data_train.n_classes()

    train_experiments = defaultdict(list)
    train_standard_runs = []
    train_l21_runs = []
    
    standard_models = []
    l21_models = []

    for seed in SEEDS:
        V_train, V_test, Y_train, Y_test = train_test_split(V.T, Y, test_size=0.3, random_state=seed)
        data_train.load_data(V_train.T, Y_train, image_shape)
        evaluation = Evaluation(V_train.T, Y_train, seed=seed)

        run, model = pipeline(
            label=f'seed={seed}',
            data=data_train,
            evaluation=evaluation,
            model_type='standard',
            rank=rank,
            fraction=0,
            seed=seed
        )
        standard_models.append(model)
        train_standard_runs.append(run)

        run, model = pipeline(
            label=f'seed={seed}',
            data=data_train,
            evaluation=evaluation,
            model_type='l21',
            rank=rank,
            fraction=0,
            recorrupt=False,
            seed=seed
        )
        l21_models.append(model)
        train_l21_runs.append(run)

        train_experiments['standard'].append(train_standard_runs)
        train_experiments['l21'].append(train_l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/clean_corrupt_split_train')
    visuals.save_details(train_experiments)
    visuals.plot_loss(train_experiments)

    test_experiments = defaultdict(list)
    for f in range(0, 40, 5):
        f = f / 100
        test_standard_runs = []
        test_l21_runs = []

        for i, seed in enumerate(SEEDS):
            V_train, V_test, Y_train, Y_test = train_test_split(V.T, Y, test_size=0.3, random_state=seed)
            data_test.load_data(V_test.T, Y_test, image_shape)
            evaluation = Evaluation(V_test.T, Y_test, seed=seed)
            standard_W, _ = standard_models[i].transform()
            l21_W, _ = l21_models[i].transform()

            run, _ = pipeline(
                label=f'fraction={f}',
                data=data_test,
                evaluation=evaluation,
                model_type='standard',
                rank=rank,
                W_init=standard_W,
                fraction=f,
                seed=seed
            )
            test_standard_runs.append(run)

            run, _ = pipeline(
                label=f'fraction={f}',
                data=data_test,
                evaluation=evaluation,
                model_type='l21',
                rank=rank,
                W_init=l21_W,
                fraction=f,
                recorrupt=False,
                seed=seed
            )
            test_l21_runs.append(run)

            test_experiments['standard'].append(test_standard_runs)
            test_experiments['l21'].append(test_l21_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/clean_corrupt_split_test')
    visuals.plot_loss(test_experiments)
    visuals.plot_metrics(test_experiments, 'fraction')
    visuals.save_details(test_experiments)
    visuals.save_images(test_experiments)
    

# Compare baseline performance of clustering (acc and nmi) on both datasets
def compare_kmeans_clustering():
    data = Processing()
    orl_acc = 0
    orl_nmi = 0
    yaleb_acc = 0
    yaleb_nmi = 0

    V, Y, image_shape = data.load('orl', 'data')
    for seed in SEEDS:
        evaluation = Evaluation(V, Y, seed=seed)
        pred = evaluation.assign_label(V)
        orl_acc += evaluation.average_accuracy(pred)
        orl_nmi += evaluation.normalized_mutual_information(pred)

    V, Y, image_shape = data.load('yaleb', 'data')
    for seed in SEEDS:
        evaluation = Evaluation(V, Y, seed=seed)
        pred = evaluation.assign_label(V)
        yaleb_acc += evaluation.average_accuracy(pred)
        yaleb_nmi += evaluation.normalized_mutual_information(pred)
    
    print(f'ORL baseline acc: {orl_acc / len(SEEDS)}')
    print(f'ORL baseline nmi: {orl_nmi / len(SEEDS)}')
    print(f'YaleB baseline acc: {yaleb_acc / len(SEEDS)}')
    print(f'YaleB baseline nmi: {yaleb_nmi / len(SEEDS)}')


# Compare performance of clean basis on unseen clean vs corrupt
def compare_outlier_error():
    data_train = Processing()
    data_test = Processing()
    V, Y, image_shape = data_train.load('orl', 'data')
    rank = data_train.n_classes()

    models = []

    for seed in SEEDS:
        V_train, V_test, Y_train, Y_test = train_test_split(V.T, Y, test_size=0.1, random_state=seed)
        data_train.load_data(V_train.T, Y_train, image_shape)
        evaluation = Evaluation(V_train.T, Y_train, seed=seed)

        run, model = pipeline(
            label=f'seed={seed}',
            data=data_train,
            evaluation=evaluation,
            model_type='standard',
            rank=rank,
            fraction=0,
            seed=seed
        )
        models.append(model)

    experiments = defaultdict(list)
    clean_runs = []
    corrupt_runs = []

    for i, seed in enumerate(SEEDS):
        V_train, V_test, Y_train, Y_test = train_test_split(V.T, Y, test_size=0.1, random_state=seed)
        data_test.load_data(V_test.T, Y_test, image_shape)
        evaluation = Evaluation(V_test.T, Y_test, seed=seed)
        W, _ = models[i].transform()

        run, model = pipeline(
            label=f'clean',
            data=data_test,
            evaluation=evaluation,
            model_type='standard',
            rank=rank,
            W_init=W,
            fraction=0,
            seed=seed
        )
        clean_runs.append(run)

        run, model = pipeline(
            label=f'corrupt',
            data=data_test,
            evaluation=evaluation,
            model_type='standard',
            rank=rank,
            W_init=W,
            fraction=1,
            seed=seed
        )
        corrupt_runs.append(run)

        experiments['clean'].append(clean_runs)
        experiments['corrupt'].append(corrupt_runs)

    visuals = Visualisation(image_shape=image_shape, output_dir='results/compare_outlier_error')
    visuals.save_details(experiments)
    visuals.plot_loss(experiments)


if __name__ == "__main__":
    try:
        compare_kmeans_clustering()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_outlier_error()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_rank_clean()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_rank_corrupt()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_num_blocks()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_block_size()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_bsize_nblocks()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        compare_corrupted_fraction()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        corrupt_clean_split()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

    try:
        clean_corrupt_split()
    except RuntimeError as e:
        print(e)
        print("Continuing")
        pass

