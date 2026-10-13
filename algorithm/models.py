"""
Non-negative Matrix Factorization algorithms.

Only the python standard library, numpy and scipy may be imported here.
Importing scikit-learn in this file breaks the assignment rules.

Shared interface:

    model = SomeNMF(rank=k, max_iter=3000, tol=1e-5, seed=42)
    model.fit(V)            # V: (d, n), non-negative
    model.W                 # (d, k)
    model.H                 # (k, n)
    model.loss_history      # list of objective values, one per iteration

Everything mechanical (initialisation, the iteration loop, convergence
checking, loss tracking) lives in BaseNMF. Each subclass only provides its
own objective function, update rules and hyperparameters. The math is written 
in the class docstring so the derivation in the report and the code stay in step.
"""

import numpy as np

EPS = 1e-10  # guards every division in a multiplicative update


class BaseNMF:
    """Common scaffolding for all NMF variants.

    Subclasses must implement:
        _objective(V) -> float
        _update(V)    -> None   (mutates self.W and self.H in place)
    """

    def __init__(self, rank, max_iter=2000, tol=1e-5, seed=42, W_init=None):
        """Initialize the factorization model.

        Parameters
        ----------
        rank : int
            Number of latent components.
        max_iter : int, default=3000
            Maximum number of optimization iterations.
        tol : float, default=1e-5
            Convergence tolerance.
        seed : int, default=42
            Random seed used to initialize ``W`` and ``H`` for
            reproducible results.
        W_init : np.ndarray
            Dictionary initialization.
        """
        self.rank : int = rank
        self.max_iter : int = max_iter
        self.tol : float = tol
        self.seed : int = seed

        self.W : np.ndarray | None = W_init
        self.H : np.ndarray | None = None
        self.loss_history : list[float] = []
        self.n_iter_ : int = 0

    # -- initialisation -------------------------------------------------

    def _init_factors(self, V : np.ndarray):
        """Uniform random non-negative initialisation, scaled to the data.

        Scaling by sqrt(mean(V) / k) keeps the initial WH in roughly the same
        range as V, which matters a lot for how fast the first iterations
        converge. Use the same seed across algorithms in one experiment run so
        the comparison is fair. If W was already intilialized, skip its initialization.
        """
        d, n = V.shape
        k = self.rank
        rng = np.random.default_rng(self.seed)

        scale = np.sqrt(max(V.mean(), EPS) / k)
        if self.W is None:
            self.W = rng.random((d, k)) * scale + EPS
        self.H = rng.random((k, n)) * scale + EPS

    # -- to be provided by subclasses -----------------------------------

    def _objective(self, V : np.ndarray):
        raise NotImplementedError

    def _update(self, V : np.ndarray, freeze : bool = False):
        raise NotImplementedError

    # -- driver ---------------------------------------------------------

    def fit(self, V : np.ndarray, verbose: bool = False, freeze : bool = False):
        """
        Run the multiplicative updates until convergence or max_iter.
        Freeze will fix the weight W and iteratively train H.
        """
        V = np.asarray(V, dtype=np.float64)
        if V.min() < 0:
            raise ValueError('NMF requires a non-negative input matrix')

        self._init_factors(V)
        self.loss_history = [self._objective(V)]

        for it in range(self.max_iter):
            self._update(V, freeze=freeze)

            loss = self._objective(V)
            self.loss_history.append(loss)
            self.n_iter_ = it + 1

            if verbose and (it + 1) % 20 == 0:
                print('  iter {:4d}  loss={:.6e}'.format(it + 1, loss))

            # Relative decrease of the objective, the usual stopping rule.
            prev = self.loss_history[-2]
            if prev > EPS and abs(prev - loss) / prev < self.tol:
                break
        return self

    # -- to be ran after fit --------------------------------------------

    def get_history(self):
        return self.loss_history

    def transform(self):
        """Return the transformed H and dictionary W"""
        if self.W is None or self.H is None:
            raise RuntimeError("Model has not been fit yet. Call fit(V) first.")
        return self.W, self.H

    def reconstruct(self):
        """Return the low rank approximation WH."""
        if self.W is None or self.H is None:
            raise RuntimeError("Model has not been fit yet. Call fit(V) first.")
        return self.W @ self.H


class StandardNMF(BaseNMF):
    """Lee and Seung (2001), squared Frobenius norm.

    Owner: Member 1

    Objective:
        J(W, H) = || V - W H ||_F^2 ,     W >= 0, H >= 0

    Multiplicative update rules, applied in the order W then H:
        W <- W * (V H^T) / (W H H^T)
        H <- H * (W^T V) / (W^T W H)

    Each rule is a gradient step whose step size is chosen elementwise so
    that the negative part of the gradient becomes the numerator and the
    positive part the denominator. Starting from positive factors the updates
    never produce negative entries, and Lee and Seung show that the objective
    is non-increasing under them.

    Cost per iteration is O(dnk). The products are grouped so that the small
    (k, k) Gram matrices H H^T and W^T W are formed first; computing W^T (W H)
    instead would build an extra full (d, n) intermediate every iteration.

    On ORL this algorithm needs roughly 2000 iterations to reach tol=1e-5,
    so an iteration cap of a few hundred stops it well short of convergence.

    Robustness: the squared loss is the maximum likelihood estimator under
    i.i.d. Gaussian noise. An occluded pixel with residual r contributes r^2
    to J, so a few saturated pixels dominate both the objective and the
    gradient. This algorithm is the baseline the robust variant is compared
    against.
    """

    def _objective(self, V):
        if self.W is None or self.H is None: raise RuntimeError()
        R = V - self.W @ self.H
        return float(np.sum(R * R))

    def _update(self, V, freeze : bool = False):
        if self.W is None or self.H is None: raise RuntimeError()
        W, H = self.W, self.H

        if not freeze:
            # W first, the alternating order used for L2,1-NMF in the report.
            # Forming the small (k, k) Gram matrix H H^T first avoids building an
            # extra (d, n) intermediate every iteration.
            VHt = V @ H.T
            HHt = H @ H.T
            W *= VHt / (W @ HHt + EPS)

        # H next, using the W just updated.
        WtV = W.T @ V
        WtW = W.T @ W
        H *= WtV / (WtW @ H + EPS)


class L21NMF(BaseNMF):
    """L2,1-norm based NMF, Kong, Ding and Huang (2011).

    Owner: Member 2

    Objective:
        J(W, H) = || V - W H ||_{2,1}
                = sum_j sqrt( sum_i (V - W H)_{ij}^2 )

    The residual of each column enters linearly instead of quadratically, so a
    badly occluded image can no longer dominate the whole objective.

    Optimisation: introduce the diagonal reweighting matrix D with
        D_jj = 1 / (2 * sqrt(||(V - W H)_{.j}||_2^2 + epsilon))
    and then iterate the reweighted updates
        H <- H * (W^T V D) / (W^T W H D)
        W <- W * (V D H^T) / (W H D H^T)
    recomputing D from the current residual at the start of every iteration.
    This is exactly iteratively reweighted least squares, which is the framing
    to use in the Methods section.

    Implementation notes:
        - D is diagonal, so never build the (n, n) matrix. Keep the diagonal
          as a length-n vector `d` and use V * d for V @ D.
        - Track the true L2,1 objective in _objective, not the reweighted
          surrogate, otherwise the loss curve is not monotone.
    """

    def _residual_weights(self, V):
        """Return the length-n diagonal of D for the current W, H."""
        if self.W is None or self.H is None: raise RuntimeError()
        residual = V - self.W @ self.H
        squared_column_norms = np.sum(residual * residual, axis=0)
        return 1.0 / (2.0 * np.sqrt(squared_column_norms + EPS))

    def _objective(self, V):
        if self.W is None or self.H is None: raise RuntimeError()
        residual = V - self.W @ self.H
        squared_column_norms = np.sum(residual * residual, axis=0)
        return float(np.sum(np.sqrt(squared_column_norms + EPS)))

    def _update(self, V, freeze : bool = False):
        if self.W is None or self.H is None: raise RuntimeError()
        eps = EPS
        W, H = self.W, self.H

        d = self._residual_weights(V)
        VD = V * d
        HD = H * d

        if not freeze:
            W *= (VD @ H.T) / np.maximum(W @ (HD @ H.T), eps)
            np.maximum(W, eps, out=W)

        H *= (W.T @ VD) / np.maximum((W.T @ W) @ HD, eps)
        np.maximum(H, eps, out=H)


# Registry used by pipeline.py so new algorithms can be added in one
# place. Keys become the algorithm names in the results table.
ALGORITHMS = {
    'standard': StandardNMF,
    'l21': L21NMF,
}
COLORS = {
    'standard' : 'b',
    'l21' : 'r'
}