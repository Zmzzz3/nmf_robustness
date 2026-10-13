import os
import json

import numpy as np
import matplotlib.pyplot as plt

from models import COLORS
from evaluation import METRICS

class Run:
    def __init__(self, 
        label : str,
        params : dict, 
        results : dict,
        images : dict, 
        history : list[float],
        corrupt_idx : list[int], 
        clean_idx : list[int]
    ) -> None:
        """
        Container for information related to a single run.

        Parameters
        ----------
        label : str
            Label used to identify the run when saving to disk.
        params : dict
            Dictionary containing run parameters.
            Should minimally include model_type, rank, num_blocks, block_size, fraction.
        results : dict
            Dictionary containing numerical outcomes.
            Should minimally include rre, acc, nmi.
        images : dict
            Dictionary containing image outputs.
            Should minimally include V, V_corrupt, reconstruction.
        history : list[float]
            Loss history of the run.
        corrupt_idx : list[int]
            Indices of corrupted images in the run.
        clean_idx : list[int]
            Indices of clean images in the run.
        """
        self.label = label
        self.params = params
        self.results = results
        self.images = images
        self.history = history
        self.corrupt_idx = corrupt_idx
        self.clean_idx = clean_idx

    def get_details(self):
        return {'label' : self.label} | self.params | self.results

    def get_images(self):
        V = self.images['V']
        V_corrupt = self.images['V_corrupt']
        reconstruction = self.images['reconstruction']
        return self.label, V, V_corrupt, reconstruction, self.corrupt_idx, self.clean_idx

class Visualisation:
    """
    Methods for generating plots and saving experiment results for a given experiment.
    """

    def __init__(self, image_shape : tuple[int, int], output_dir : str = 'results/unnamed_experiment') -> None:
        """
        Parameters
        ----------
        image_shape : tuple[int, int]
            (height, width) to reshape each flattened column into.
        output_dir : str, default = 'results'
            Reccommend specifying the experiment (like results/experiment_orl).
        """

        self.image_shape = image_shape
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_loss(self, experiments : dict[str, list[list[Run]]], log_scale : bool = True):
        """
        Plots all loss histories in one graph. 

        Parameters
        ----------
        runs : dict[str, list[list[Run]]]
            Each dictionary item will have its own loss plot.
            Each outer list item corresponds to its own loss line.
            The first sample will be chosen from each list of runs to represent that setting.
        log_scale : bool, default = True
            Use a log y-axis, since loss typically spans several orders of magnitude.
        """
        for label, experiment in experiments.items():
            for runs in experiment:
                sample = runs[0]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_title(f'{label} loss')
                ax.set_xlabel('iteration')
                ax.set_ylabel('loss')
                    
                ax.plot(sample.history, label=sample.label)
                if log_scale: ax.set_yscale('log')
                ax.legend(fontsize='small')
                fig.tight_layout()

                path = os.path.join(self.output_dir, f'{label} loss.png')
                fig.savefig(path)
                plt.close(fig)

    def plot_metrics(self, experiments : dict[str, list[list[Run]]], parameter : str):
        """
        Plots a graph of parameter against score, one for each evaluation metric. 

        Parameters
        ----------
        experiments : dict[str, list[list[Run]]]
            Each dictionary item corresponds to one line.
            Each outer list item corresponds to one set of runs at a given setting.
            Each list of runs will be aggregated into a point on each metric plot.
        parameter : str
            Defines the parameter to plot the score against (like block_size, num_blocks, fraction).
        """
        n = len(METRICS)
        fig, axes = plt.subplots(1, n, figsize=(12, 3))
        axes = np.atleast_1d(axes)

        for ax, metric in zip(axes, METRICS):
            ax.set_title(f'{metric} vs {parameter}')
            ax.set_xlabel(parameter)
            ax.set_ylabel(metric)

            for label, experiment in experiments.items():
                try:
                    experiment.sort(key=lambda p: p[0].params[parameter])
                    value = [runs[0].params[parameter] for runs in experiment]
                    means = [np.mean([run.results[metric] for run in runs]) for runs in experiment]
                    medians = [np.median([run.results[metric] for run in runs]) for runs in experiment]
                    stds = [np.std([run.results[metric] for run in runs]) for runs in experiment]
                except KeyError as e:
                    print(f'Skipping entry, missing field: {e}')
                    continue

                ax.errorbar(value, means, yerr=stds, fmt="none", ecolor=COLORS[label],
                            elinewidth=1, capsize=2, capthick=1, alpha=0.2, zorder=2)
                ax.plot(value, medians, "-o", color=COLORS[label], linewidth=2,
                        markersize=4, label=label, zorder=3)
            
            ax.legend(fontsize='small')

        fig.tight_layout()
        
        path = os.path.join(self.output_dir, f'metrics vs {parameter}.png')
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_details(self, experiments : dict[str, list[list[Run]]]):
        """
        Saves (label, params, results) of all runs to a JSON file.
        """

        details = {}
        path = os.path.join(self.output_dir, 'results.json')

        for label, experiment in experiments.items():
            for runs in experiment:
                summary = {}

                for metric in METRICS:
                    try:
                        summary[f'{metric} mean'] = np.mean([run.results[metric] for run in runs])
                        summary[f'{metric} median'] = np.median([run.results[metric] for run in runs])
                        summary[f'{metric} std'] = np.std([run.results[metric] for run in runs])
                    except KeyError as e:
                        print(f'Skipping entry, missing field: {e}')
                        continue
                
                summary['runs'] = [run.get_details() for run in runs]
                details[label] = summary

        with open(path, 'w') as file:
            json.dump(details, file, default=lambda x : x.item(), indent=2)

    def plot_images(self, V : np.ndarray, V_corrupt : np.ndarray, 
                    reconstruction : np.ndarray, idx : int, label : str, filename : str):
        
        clean = V[:, idx].reshape(self.image_shape)
        corrupted = V_corrupt[:, idx].reshape(self.image_shape)
        reconstructed = reconstruction[:, idx].reshape(self.image_shape)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        panels = [
            (clean, 'Clean', 'gray', 0.0, 1.0),
            (corrupted, 'Corrupted', 'gray', 0.0, 1.0),
            (reconstructed, 'Reconstruction', 'gray', 0.0, 1.0),
        ]
        
        for axis, (array, title, cmap, vmin, vmax) in zip(ax, panels):
            axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.axis("off")

        fig.suptitle(label)
        fig.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_images(self, experiments : dict[str, list[list[Run]]], sample_index : int = 0):
        """
        Saves sample images from runs, including V, V_corrupt and reconstruction.
        One sample will be chosen from both clean and corrupted images using sample_index.

        Parameters
        ----------
        experiments : dict[str, list[list[Run]]]
            The first sample in each list of runs will be used to save images.
        sample_index : int, default = 0
            Defines the index to sample from all clean and corrupt indices.
        """

        for label, experiment in experiments.items():
            for runs in experiment:
                sample = runs[0]

                try:
                    sample_label, V, V_corrupt, reconstruction, corrupt_idx, clean_idx = sample.get_images()
                except KeyError as e:
                    print(f'Skipping entry, missing field: {e}')
                    continue

                corrupt_idx = corrupt_idx[sample_index] if 0 <= sample_index < len(corrupt_idx) else None
                clean_idx = clean_idx[sample_index] if 0 <= sample_index < len(clean_idx) else None

                if corrupt_idx is not None:
                    self.plot_images(V=V, V_corrupt=V_corrupt, reconstruction=reconstruction, idx=corrupt_idx, 
                                    label=f'{label} {sample_label} reconstruction of image {corrupt_idx}', 
                                    filename=f'{label} {sample_label} img{corrupt_idx}.png')
                if clean_idx is not None:
                    self.plot_images(V=V, V_corrupt=V_corrupt, reconstruction=reconstruction, idx=clean_idx, 
                                    label=f'{label} {sample_label} reconstruction of image {clean_idx}', 
                                    filename=f'{label} {sample_label} img{clean_idx}.png')
