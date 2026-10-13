from pathlib import Path

import numpy as np
from PIL import Image

class Processing:
    """
    Handles loading of data and preprocessing steps, including data corruption.
    Acts as a container for the clean dataset and its most recently corrupted version.
    """

    _DATASET_FOLDERS = {"orl": "ORL", "yaleb": "CroppedYaleB"}
    _DEFAULT_REDUCE = {"orl": 3, "yaleb": 5}

    def __init__(self):

        # Populated by load().
        self.V : np.ndarray | None = None
        self.labels : np.ndarray | None = None
        self.image_shape : tuple[int, int] | None = None

        # Populated by corrupt().
        self.V_corrupt : np.ndarray | None = None
        self.masks : np.ndarray | None = None
        self.corrupt_idx : list[int] | None = None

    def locate(self, dataset_folder: str, data_dir: str) -> Path:
        """Find the dataset directory beside the script, one level above it, or in the CWD."""
        if data_dir:
            root = Path(data_dir).expanduser().resolve() / dataset_folder
            if root.is_dir():
                return root
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        try:
            script_dir = Path(__file__).resolve().parent
        except NameError:  # e.g. running in a notebook/REPL
            script_dir = Path.cwd()

        candidates = [
            script_dir / "data" / dataset_folder,
            script_dir.parent / "data" / dataset_folder,
            Path.cwd() / "data" / dataset_folder,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()

        searched = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"Could not find {dataset_folder}. Searched:\n{searched}")

    def load(
        self, dataset: str = 'orl', data_dir: str = '', reduce: int = 0, split: float = 0.8, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """
        Locate and load the dataset into V, labels, image_shape.
        
        Parameters
        ----------
        dataset : str, default = 'orl'
            Name of the dataset ('orl' or 'yaleb'), to be mapped to the expected folder name
        data_dir : str, default = ''
            Relative path to the folder containing the dataset, from the current working directory
        reduce : int, defualt = 0
            Image downsampling factor
        split : float, default = 0.8
            Uses a random split of the original dataset
        seed : int, default = 42
            Seed used for deciding the split
        """

        if dataset not in self._DATASET_FOLDERS:
            raise ValueError(f"dataset must be one of {list(self._DATASET_FOLDERS)}, got {dataset!r}.")
        dataset_folder = self._DATASET_FOLDERS[dataset]
        reduce = self._DEFAULT_REDUCE[dataset] if not reduce else reduce
        if reduce <= 0: raise ValueError("reduce must be a positive integer.")

        # find the path to the dataset, and verify its structure
        self.root = self.locate(dataset_folder=dataset_folder, data_dir=data_dir)
        person_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        if not person_dirs:
            raise ValueError(f"No subject directories were found in {self.root}.")
        
        resampling_method = Image.Resampling.BILINEAR
        images, labels, image_shape = [], [], None

        for label, person_dir in enumerate(person_dirs):
            for image_path in sorted(person_dir.iterdir()):

                filename_lower = image_path.name.lower()
                if filename_lower.endswith("ambient.pgm"):
                    continue
                if image_path.suffix.lower() != ".pgm":
                    continue

                # perform size reduction 
                with Image.open(image_path) as image:
                    image = image.convert("L")
                    new_width = image.width // reduce
                    new_height = image.height // reduce
                    if new_width <= 0 or new_height <= 0:
                        raise ValueError(f"reduce={reduce} is too large for {image_path}.")
                    image = image.resize((new_width, new_height), resample=resampling_method)
                    image_array = np.asarray(image, dtype=np.float64) / 255.0

                # ensure all resized images are of the same size
                current_shape = image_array.shape
                if image_shape is None:
                    image_shape = current_shape
                elif current_shape != image_shape:
                    raise ValueError("All images must have the same resized shape; "
                        f"expected {image_shape}, got {current_shape} for {image_path}.")

                images.append(image_array.reshape(-1, 1))
                labels.append(label)

        if not images or not image_shape:
            raise ValueError(f"No usable .pgm images were found in {self.root}.")
        self.V = np.concatenate(images, axis=1)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.image_shape = image_shape

        # perform random sampling as necessary
        if not 0.0 < split <= 1.0: raise ValueError("split must be in (0, 1].")
        if split < 1.0:
            n_samples = self.V.shape[1]
            n_keep = round(split * n_samples)
            rng = np.random.default_rng(seed)
            keep_idx = rng.choice(n_samples, size=n_keep, replace=False)
            keep_idx.sort()

            self.V = self.V[:, keep_idx]
            self.labels = self.labels[keep_idx]

        return self.V, self.labels, self.image_shape

    def load_data(
        self, V : np.ndarray, labels : np.ndarray, image_shape : tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """
        Manually load the clean dataset, used to copy Processing instances
        """
        self.V = V
        self.labels = labels
        self.image_shape = image_shape
        return self.V, self.labels, self.image_shape

    def corrupt(
        self,
        block_size: int = 1,
        num_blocks: int = 100,
        fraction: float = 1.0,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Add square block occlusion to a fraction of images.

        Parameters
        ----------
        block_size : int, default = 1
            Edge length of each square occlusion block.
        num_blocks : int, defualt = 100
            Number of blocks added to each corrupted image.
        fraction : float, default = 1.0
            Fraction of images to corrupt, between 0 and 1.
        seed : int, default = 42
            Seed used to randomise pixel and image selection for corruption.

        Returns
        -------
        V_corrupted : np.ndarray
            Corrupted copy of V.
        corrupted_indices : np.ndarray
            Indices of images that were corrupted.
        """
        if self.V is None or self.labels is None or self.image_shape is None:
            raise RuntimeError("Call load() before corrupt_subset().")

        if not 0 <= fraction <= 1:
            raise ValueError("corruption_fraction must be between 0 and 1")
        if block_size < 0 or block_size > min(self.image_shape):
            raise ValueError("invalid block_size")
        if num_blocks < 0:
            raise ValueError("n_blocks must be non-negative")

        rng = np.random.default_rng(seed)
        V_corrupt = self.V.copy()
        masks = np.zeros_like(V_corrupt, dtype=bool)

        height, width = self.image_shape
        n_samples = self.V.shape[1]
        n_corrupt = round(fraction * n_samples)
        corrupt_idx = rng.choice(n_samples, size=n_corrupt, replace=False)
        corrupt_idx.sort()

        for sample_index in corrupt_idx:
            image = V_corrupt[:, sample_index].reshape(height, width)
            mask = masks[:, sample_index].reshape(height, width)
            for _ in range(num_blocks):
                row = int(rng.integers(0, height - block_size + 1))
                column = int(rng.integers(0, width - block_size + 1))
                image[row : row + block_size, column : column + block_size] = 1
                mask[row : row + block_size, column : column + block_size] = True

        self.V_corrupt = V_corrupt
        self.masks = masks
        self.corrupt_idx = corrupt_idx
        return self.V_corrupt, self.masks, corrupt_idx

    def n_classes(self) -> int:
        if self.labels is None: raise RuntimeError("Call load() first.")
        return len(np.unique(self.labels))

    def get_loaded(self):
        if self.V is None or self.labels is None or self.image_shape is None:
            raise RuntimeError("Dataset hasn't been loaded yet.")
        return self.V, self.labels, self.image_shape

    def get_corrupted(self):
        if self.V_corrupt is None or self.masks is None or self.corrupt_idx is None:
            raise RuntimeError("Dataset hasn't been corrupted yet.")
        return self.V_corrupt, self.masks, self.corrupt_idx
