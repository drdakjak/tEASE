"""Closed-form EASE recommenders and TEASE exploration variants.

This module provides:

- EASE: deterministic item-item recommendations.
- TEASE: Gaussian posterior score sampling on top of EASE.
- low-rank variants for lower serving memory.

The TEASE name is used for the exploration variants because it is shorter and
more library-friendly than the internal historical name
GaussianExplorationEASE. Backwards-compatible class aliases are kept in place.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse

__all__ = [
    "EASE",
    "GaussianExplorationEASE",
    "LowRankEASE",
    "LowRankGaussianExplorationEASE",
    "TEASE",
    "LowRankTEASE",
]


class EASE:
    """
    Embarrassingly Shallow Autoencoder for implicit-feedback recommendation.

    Parameters
    ----------
    l2_scale:
        Scale-adaptive L2 regularization.

        Instead of adding a fixed absolute value to the diagonal of X.T @ X,
        the model computes:

            effective_l2 = l2_scale * mean(diag(X.T @ X))

        and then builds:

            G = X.T @ X + effective_l2 * I

        This makes regularization less sensitive to the numeric scale of X.

        Why this matters
        ----------------
        EASE uses X.T @ X. Therefore, if you multiply all values in X by c,
        the Gram matrix grows by c²:

            (cX).T @ (cX) = c² X.T @ X

        With absolute regularization, changing event values such as:

            click = 1
            purchase = 3

        also changes the relative strength of regularization. Scale-adaptive
        regularization reduces this problem.

        Intuition
        ---------
        Smaller l2_scale:
            - stronger item-item associations
            - sharper / more personalized recommendations
            - higher risk of noisy co-occurrence effects
            - larger posterior uncertainty in the exploration variant

        Larger l2_scale:
            - weaker item-item associations
            - smoother and more conservative recommendations
            - less sensitivity to rare accidental co-occurrences
            - smaller posterior uncertainty in the exploration variant

        Reasonable values
        -----------------
        Good first sweep:

            [0.03, 0.1, 0.3, 1.0, 3.0]

        Wider sweep:

            [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

        Suggested default:

            l2_scale = 0.3

        If recommendations look too generic:
            decrease l2_scale.

        If recommendations look too noisy, repetitive, or driven by rare
        accidental co-occurrences:
            increase l2_scale.

        If exploration looks too random:
            increase l2_scale or decrease exploration_scale.

        If exploration barely changes the ranking:
            decrease l2_scale or increase exploration_scale.

    compute_dtype:
        Floating-point dtype used for dense linear algebra.

        Strong recommendation:
            use np.float64 or np.float32.

        Avoid np.float16 for fitting because matrix inversion in float16 is
        numerically unstable and often unsupported by NumPy linear algebra.

    storage_dtype:
        Optional dtype used to store B and P after fitting.

        If None, uses compute_dtype.

        You may use np.float32 to reduce memory after computing in np.float64.
        Avoid storing P in float16 if you use posterior variance for exploration,
        because small numerical errors can create unstable variance estimates.
    """

    def __init__(
        self,
        l2_scale: float = 0.3,
        compute_dtype: np.dtype = np.float64,
        storage_dtype: np.dtype | None = None,
    ):
        if l2_scale <= 0:
            raise ValueError("l2_scale must be positive.")

        self.l2_scale = float(l2_scale)
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype or compute_dtype

        self.X_: csr_matrix | None = None
        self.B_: np.ndarray | None = None
        self.effective_l2_: float | None = None
        self.gram_scale_: float | None = None

    def fit(self, X: csr_matrix) -> "EASE":
        if not issparse(X):
            X = csr_matrix(X)

        X = X.tocsr().astype(self.compute_dtype)
        self.X_ = X

        G = (X.T @ X).toarray().astype(self.compute_dtype, copy=False)

        # Scale-adaptive regularization:
        # effective_l2 = l2_scale * average item self-strength.
        gram_diag = np.diag(G)
        gram_scale = float(np.mean(gram_diag))

        if gram_scale <= 0:
            raise ValueError(
                "Cannot fit EASE: mean(diag(X.T @ X)) is zero. "
                "The input matrix appears to contain no usable signal."
            )

        effective_l2 = self.l2_scale * gram_scale

        self.gram_scale_ = gram_scale
        self.effective_l2_ = effective_l2

        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += effective_l2

        P = np.linalg.inv(G)
        P_diag = np.diag(P).copy()

        if np.any(np.isclose(P_diag, 0.0)):
            raise FloatingPointError(
                "Near-zero diagonal encountered in inverse Gram matrix. "
                "Try increasing l2_scale."
            )

        # EASE closed-form solution: B_ij = -P_ij / P_jj, with B_jj = 0.
        B = P / (-P_diag)
        np.fill_diagonal(B, 0.0)

        self.B_ = B.astype(self.storage_dtype, copy=False)
        self._on_fit(P, P_diag)

        return self

    def _on_fit(self, P: np.ndarray, P_diag: np.ndarray) -> None:
        """Hook called at the end of fit(). Subclasses may store P and P_diag."""
        pass

    @property
    def item_factors(self) -> np.ndarray:
        self._check_fitted()
        return self.B_

    @staticmethod
    def _mask_positive_interactions(
        scores: np.ndarray, user_vector: csr_matrix
    ) -> np.ndarray:
        """Mask items with positive interaction values by setting their scores to -inf."""
        scores = scores.copy()
        positive_mask = user_vector.data > 0
        positive_indices = user_vector.indices[positive_mask]
        scores[positive_indices] = -np.inf
        return scores

    def _predict(
        self, user_vector: csr_matrix, mask_seen: bool = True
    ) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        scores = np.asarray(user_vector @ self.B_).ravel()

        if mask_seen:
            scores = self._mask_positive_interactions(scores, user_vector)

        return scores

    def recommend(
        self,
        user_vector: csr_matrix,
        n: int | None = 10,
        mask_seen: bool = True,
    ) -> np.ndarray:
        """
        Generate top-N recommendations for a user.

        Parameters
        ----------
        user_vector:
            Sparse 1 x n_items vector representing user's interaction history.
        n:
            Number of items to recommend. If None, returns all items ranked.
        mask_seen:
            If True, excludes items with positive interactions from recommendations.

        Returns
        -------
        np.ndarray:
            Array of item indices ranked by score (highest first).
        """
        scores = self._predict(user_vector=user_vector, mask_seen=mask_seen)
        return self._top_n(scores, n)

    def related_items(
        self,
        item_id: int,
        n: int | None = 10,
        exclude_self: bool = True,
    ) -> np.ndarray:
        self._check_fitted()
        self._check_item_id(item_id)

        scores = self.B_[item_id].copy()

        if exclude_self:
            scores[item_id] = -np.inf

        return self._top_n(scores, n)

    def similar_items(
        self,
        item_id: int,
        n: int | None = 10,
        exclude_self: bool = True,
    ) -> np.ndarray:
        self._check_fitted()
        self._check_item_id(item_id)

        scores = self.B_[item_id] @ self.B_.T

        if exclude_self:
            scores = scores.copy()
            scores[item_id] = -np.inf

        return self._top_n(scores, n)

    def _check_fitted(self) -> None:
        if self.B_ is None or self.effective_l2_ is None:
            raise RuntimeError("Model is not fitted yet.")

    def _check_user_vector(self, user_vector: csr_matrix) -> None:
        self._check_fitted()

        if not issparse(user_vector):
            raise TypeError("user_vector must be a sparse matrix.")

        if user_vector.shape[0] != 1:
            raise ValueError(
                f"user_vector must have shape (1, n_items), got {user_vector.shape}."
            )

        if user_vector.shape[1] != self.B_.shape[0]:
            raise ValueError(
                f"user_vector has {user_vector.shape[1]} items, "
                f"but model was trained on {self.B_.shape[0]} items."
            )

    def _check_item_id(self, item_id: int) -> None:
        if self.B_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if item_id < 0 or item_id >= self.B_.shape[0]:
            raise IndexError(f"item_id={item_id} is out of bounds.")

    @staticmethod
    def _top_n(scores: np.ndarray, n: int | None) -> np.ndarray:
        if n is None:
            return np.argsort(-scores)

        if n <= 0:
            raise ValueError("n must be positive.")

        n = min(n, scores.shape[0])
        candidate_idx = np.argpartition(-scores, kth=n - 1)[:n]
        return candidate_idx[np.argsort(-scores[candidate_idx])]

    def _get_state(self) -> dict:
        return {
            "l2_scale": self.l2_scale,
            "compute_dtype": self.compute_dtype,
            "storage_dtype": self.storage_dtype,
            "B_": self.B_,
            "effective_l2_": self.effective_l2_,
            "gram_scale_": self.gram_scale_,
        }

    def _restore_state(self, state: dict) -> None:
        self.X_ = None  # Training data not stored
        self.B_ = state["B_"]
        self.effective_l2_ = state["effective_l2_"]
        self.gram_scale_ = state["gram_scale_"]

    def save(self, path: str | Path) -> None:
        """
        Save the fitted model to disk.

        Parameters
        ----------
        path:
            File path where the model will be saved. The file will be created
            or overwritten if it already exists.

        Raises
        ------
        RuntimeError:
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = EASE(l2_scale=0.3)
        >>> model.fit(X_train)
        >>> model.save("ease_model.pkl")
        >>>
        >>> # Later, load and use:
        >>> loaded = EASE.load("ease_model.pkl")
        >>> user_vec = csr_matrix([[1, 0, 3, 0, 2]])  # shape (1, n_items)
        >>> recs = loaded.recommend(user_vec, n=10)
        """
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "EASE":
        """
        Load a fitted model from disk.

        Parameters
        ----------
        path:
            File path from which the model will be loaded.

        Returns
        -------
        EASE:
            The loaded fitted model instance.

        Raises
        ------
        FileNotFoundError:
            If the file does not exist.

        Examples
        --------
        >>> model = EASE.load("ease_model.pkl")
        >>> user_vec = csr_matrix([[1, 0, 3, 0, 2]])  # shape (1, n_items)
        >>> recommendations = model.recommend(user_vec, n=10)
        """
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            l2_scale=state["l2_scale"],
            compute_dtype=state["compute_dtype"],
            storage_dtype=state["storage_dtype"],
        )
        instance._restore_state(state)
        return instance


class GaussianExplorationEASE(EASE):
    """
    EASE with approximate Thompson-style score sampling.

    This model is the basis for the public TEASE name.

    It is not exact Thompson sampling over click or purchase probabilities.
    Instead, it applies Gaussian posterior score perturbation on top of EASE.
    In practice, this is often a good fit when you want EASE-like speed and
    simplicity with exploration during ranking.

    The model keeps one exploration parameter:

        exploration_scale

    The sampled score is:

        sampled_score_ui = mean_ui + exploration_scale * std_ui * epsilon

    where:

        epsilon ~ Normal(0, 1)

    Interpretation
    --------------
    exploration_scale = 0.0
        Deterministic EASE. No exploration.

    exploration_scale < 1.0
        Conservative exploration.

    exploration_scale = 1.0
        Direct use of the Gaussian posterior approximation.

    exploration_scale > 1.0
        Aggressive exploration.

    Parameters
    ----------
    l2_scale:
        Scale-adaptive L2 regularization inherited from EASE.

        Reasonable values:

            [0.03, 0.1, 0.3, 1.0, 3.0]

        Lower values sharpen co-occurrence structure and typically increase the
        amount of useful uncertainty available for exploration. Higher values
        make the model more conservative.

    random_state:
        Optional random seed for reproducible exploration draws.

        Usage:
            Set this during experiments or evaluation when you want stable,
            repeatable rankings. Leave it as None in production if you want
            fresh stochastic exploration on every process start.

    compute_dtype:
        Floating-point dtype used during fitting and dense linear algebra.

        Reasonable values:

            [np.float64, np.float32]

        Use np.float64 for safer numerical behaviour on fit. Use np.float32 if
        memory pressure is higher and the catalog is not numerically fragile.

    storage_dtype:
        Optional dtype used to store fitted arrays after training.

        Reasonable values:

            [None, np.float32, np.float64]

        Usage:
            Keep None to store arrays in compute_dtype. Use np.float32 to
            reduce model size after fitting in np.float64. Avoid float16.

    Reasonable values
    -----------------
    Good first sweep:

        [0.0, 0.1, 0.25, 0.5, 1.0]

    Wider sweep:

        [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]

    Suggested default:

        exploration_scale = 0.5

    """

    def __init__(
        self,
        l2_scale: float = 0.3,
        random_state: int | None = None,
        compute_dtype: np.dtype = np.float64,
        storage_dtype: np.dtype | None = None,
    ):
        super().__init__(
            l2_scale=l2_scale,
            compute_dtype=compute_dtype,
            storage_dtype=storage_dtype,
        )

        self.rng = np.random.default_rng(random_state)
        self.P_: np.ndarray | None = None
        self.P_diag_: np.ndarray | None = None

    def _on_fit(self, P: np.ndarray, P_diag: np.ndarray) -> None:
        self.P_ = P.astype(self.compute_dtype, copy=False)
        self.P_diag_ = P_diag.astype(self.compute_dtype, copy=False)

    def _predictive_variance(self, user_vector: csr_matrix) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        n_items = self.B_.shape[0]

        if user_vector.nnz == 0:
            return np.zeros(n_items, dtype=self.storage_dtype)

        x = user_vector.astype(self.compute_dtype, copy=False)

        cross = np.asarray(x @ self.P_).ravel()

        history_quadratic = float(x @ cross)

        variance = history_quadratic - (cross**2 / self.P_diag_)

        return np.maximum(variance, 0.0).astype(self.storage_dtype, copy=False)

    def _sample_scores(
        self,
        user_vector: csr_matrix,
        mask_seen: bool = True,
        exploration_scale: float = 0.5,
    ) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        mean = np.asarray(user_vector @ self.B_).ravel()
        variance = self._predictive_variance(user_vector)
        std = np.sqrt(variance)

        sampled_scores = self.rng.normal(
            loc=mean,
            scale=exploration_scale * std,
        )

        if mask_seen:
            sampled_scores = self._mask_positive_interactions(
                sampled_scores, user_vector
            )

        return sampled_scores

    def recommend(
        self,
        user_vector: csr_matrix,
        n: int | None = 10,
        exploration_scale: float = 0.5,
        mask_seen: bool = True,
    ) -> np.ndarray:
        """
        Generate top-N recommendations with exploration for a user.

        Parameters
        ----------
        user_vector:
            Sparse 1 x n_items vector representing user's interaction history.
        n:
            Number of items to recommend. If None, returns all items ranked.
        exploration_scale:
            Controls exploration intensity. 0.0 = no exploration (deterministic),
            1.0 = use posterior variance directly, >1.0 = aggressive exploration.
        mask_seen:
            If True, excludes items with positive interactions from recommendations.

        Returns
        -------
        np.ndarray:
            Array of item indices ranked by score (highest first).
        """
        if exploration_scale < 0:
            raise ValueError("exploration_scale must be non-negative.")

        if exploration_scale > 0:
            scores = self._sample_scores(
                user_vector=user_vector,
                mask_seen=mask_seen,
                exploration_scale=exploration_scale,
            )
        else:
            scores = self._predict(
                user_vector=user_vector, mask_seen=mask_seen
            )

        return self._top_n(scores, n)

    def _get_state(self) -> dict:
        state = super()._get_state()
        state["P_"] = self.P_
        state["P_diag_"] = self.P_diag_
        state["rng_state"] = self.rng.bit_generator.state
        return state

    def _restore_state(self, state: dict) -> None:
        super()._restore_state(state)
        self.P_ = state["P_"]
        self.P_diag_ = state["P_diag_"]
        self.rng.bit_generator.state = state["rng_state"]


class LowRankEASE(EASE):
    """
    EASE with B stored in factored low-rank form to reduce serving RAM.

    The rank-k truncated eigendecomposition of G = X^T X + lambda*I gives:

        P_k = V_k Λ_k^{-1} V_k^T

    from which B is derived without materialising the full n x n matrix.
    Scores are computed in two steps:

        t     = x_u @ V_k                       O(nnz_u * k)
        cross = (t * Λ_k^{-1}) @ V_k^T          O(k * n)
        score = -cross / P_k_diag                O(n)

    Memory at rest: O(n * k)  vs  O(n^2) for exact EASE.
    Training RAM peak is still O(n^2) due to the full eigendecomposition.

    Note
    ----
    item_factors and similar_items are not supported because B is not
    stored explicitly. related_items computes the relevant row on the fly.

    Parameters
    ----------
    rank:
        Number of eigenvectors to retain.
        Higher rank improves approximation at the cost of RAM and inference
        time. Reasonable starting values: 50-500 depending on catalog size.
        With rank = n_items the result is identical to exact EASE.

        Usage:
            Start around 100-300 for medium catalogs. Increase rank if item
            quality drops too much versus exact EASE. Decrease it when serving
            memory is the main constraint.

    l2_scale:
        Scale-adaptive regularization inherited from EASE.

        Reasonable values:

            [0.03, 0.1, 0.3, 1.0, 3.0]

    compute_dtype:
        Dense linear algebra dtype used during fit.

        Reasonable values:

            [np.float64, np.float32]

    storage_dtype:
        Optional dtype used to store low-rank factors.

        Reasonable values:

            [None, np.float32, np.float64]
    """

    def __init__(
        self,
        rank: int = 200,
        l2_scale: float = 0.3,
        compute_dtype: np.dtype = np.float64,
        storage_dtype: np.dtype | None = None,
    ):
        super().__init__(
            l2_scale=l2_scale,
            compute_dtype=compute_dtype,
            storage_dtype=storage_dtype,
        )
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.rank = rank
        self.V_: np.ndarray | None = None
        self.lambda_inv_: np.ndarray | None = None
        self.P_k_diag_: np.ndarray | None = None

    def fit(self, X: csr_matrix) -> "LowRankEASE":
        if not issparse(X):
            X = csr_matrix(X)

        X = X.tocsr().astype(self.compute_dtype)
        self.X_ = X

        G = (X.T @ X).toarray().astype(self.compute_dtype, copy=False)

        gram_diag = np.diag(G)
        gram_scale = float(np.mean(gram_diag))

        if gram_scale <= 0:
            raise ValueError(
                "Cannot fit LowRankEASE: mean(diag(X.T @ X)) is zero. "
                "The input matrix appears to contain no usable signal."
            )

        effective_l2 = self.l2_scale * gram_scale
        self.gram_scale_ = gram_scale
        self.effective_l2_ = effective_l2

        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += effective_l2

        # eigh returns eigenvalues in ascending order.
        # Training peak RAM: O(n^2) for G and eigenvectors; only O(n*k)
        # is retained after slicing.
        eigenvalues, eigenvectors = np.linalg.eigh(G)

        k = min(self.rank, eigenvalues.shape[0])
        eigenvalues_k = eigenvalues[-k:][::-1].copy()
        eigenvectors_k = eigenvectors[:, -k:][:, ::-1].copy()

        if eigenvalues_k[-1] <= 0:
            raise FloatingPointError(
                "Non-positive eigenvalue in rank-k truncation. "
                "Try increasing l2_scale or decreasing rank."
            )

        lambda_inv_k = (1.0 / eigenvalues_k).astype(self.compute_dtype)
        P_k_diag = (eigenvectors_k**2) @ lambda_inv_k

        if np.any(np.isclose(P_k_diag, 0.0)):
            raise FloatingPointError(
                "Near-zero diagonal in low-rank P approximation. "
                "Try increasing l2_scale or rank."
            )

        self.V_ = eigenvectors_k.astype(self.storage_dtype, copy=False)
        self.lambda_inv_ = lambda_inv_k.astype(self.storage_dtype, copy=False)
        self.P_k_diag_ = P_k_diag.astype(self.storage_dtype, copy=False)

        return self

    def _check_fitted(self) -> None:
        if (
            self.V_ is None
            or self.lambda_inv_ is None
            or self.P_k_diag_ is None
            or self.effective_l2_ is None
        ):
            raise RuntimeError("Model is not fitted yet.")

    def _check_user_vector(self, user_vector: csr_matrix) -> None:
        self._check_fitted()

        if not issparse(user_vector):
            raise TypeError("user_vector must be a sparse matrix.")

        if user_vector.shape[0] != 1:
            raise ValueError(
                f"user_vector must have shape (1, n_items), got {user_vector.shape}."
            )

        if user_vector.shape[1] != self.V_.shape[0]:
            raise ValueError(
                f"user_vector has {user_vector.shape[1]} items, "
                f"but model was trained on {self.V_.shape[0]} items."
            )

    def _check_item_id(self, item_id: int) -> None:
        if self.V_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if item_id < 0 or item_id >= self.V_.shape[0]:
            raise IndexError(f"item_id={item_id} is out of bounds.")

    def _predict(
        self, user_vector: csr_matrix, mask_seen: bool = True
    ) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        x = user_vector.astype(self.compute_dtype, copy=False)
        t = np.asarray(x @ self.V_).ravel()  # (k,)
        cross = (t * self.lambda_inv_) @ self.V_.T  # (n,)
        scores = -cross / self.P_k_diag_

        if mask_seen:
            scores = self._mask_positive_interactions(scores, user_vector)

        return scores

    @property
    def item_factors(self) -> np.ndarray:
        """Low-rank item embeddings: row i is V_k[i] / sqrt(lambda_k).

        Satisfies F @ F.T = P_k, so inner products recover the rank-k
        approximation of the inverse Gram matrix.
        """
        self._check_fitted()
        return self.V_ * np.sqrt(self.lambda_inv_)

    def related_items(
        self,
        item_id: int,
        n: int | None = 10,
        exclude_self: bool = True,
    ) -> np.ndarray:
        self._check_fitted()
        self._check_item_id(item_id)

        # Row item_id of B_k: -P_k[item_id, :] / P_k_diag
        cross = (self.V_[item_id] * self.lambda_inv_) @ self.V_.T
        scores = -cross / self.P_k_diag_

        if exclude_self:
            scores = scores.copy()
            scores[item_id] = -np.inf

        return self._top_n(scores, n)

    def similar_items(
        self,
        item_id: int,
        n: int | None = 10,
        exclude_self: bool = True,
    ) -> np.ndarray:
        """Find items with the most similar B_k row to item_id.

        Exploits V_k^T V_k = I_k so the k x k cross-term collapses:

            B_k[i] . B_k[j] = h_i . h_j / (P_k_diag[i] * P_k_diag[j])

        where h = V_k * lambda_inv. Cost: O(k * n).
        """
        self._check_fitted()
        self._check_item_id(item_id)

        # h_i = V_k[i] * lambda_inv  (k,)
        h = self.V_[item_id] * self.lambda_inv_
        # c_j = h_i . h_j = h @ (V_k * lambda_inv).T  (n,)
        c = h @ (self.V_ * self.lambda_inv_).T
        scores = c / (self.P_k_diag_[item_id] * self.P_k_diag_)

        if exclude_self:
            scores = scores.copy()
            scores[item_id] = -np.inf

        return self._top_n(scores, n)

    def _get_state(self) -> dict:
        return {
            "l2_scale": self.l2_scale,
            "rank": self.rank,
            "compute_dtype": self.compute_dtype,
            "storage_dtype": self.storage_dtype,
            "V_": self.V_,
            "lambda_inv_": self.lambda_inv_,
            "P_k_diag_": self.P_k_diag_,
            "effective_l2_": self.effective_l2_,
            "gram_scale_": self.gram_scale_,
        }

    def _restore_state(self, state: dict) -> None:
        self.X_ = None
        self.rank = state["rank"]
        self.V_ = state["V_"]
        self.lambda_inv_ = state["lambda_inv_"]
        self.P_k_diag_ = state["P_k_diag_"]
        self.effective_l2_ = state["effective_l2_"]
        self.gram_scale_ = state["gram_scale_"]

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "LowRankEASE":
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(
            rank=state["rank"],
            l2_scale=state["l2_scale"],
            compute_dtype=state["compute_dtype"],
            storage_dtype=state["storage_dtype"],
        )
        instance._restore_state(state)
        return instance


class LowRankGaussianExplorationEASE(GaussianExplorationEASE):
    """
    Low-rank TEASE with diagonal-corrected uncertainty estimates.

    B is approximated identically to LowRankEASE (rank-k truncated eigen of G).

    P is approximated as P_k + diag(delta), where:

        P_k    = V_k Λ_k^{-1} V_k^T          rank-k reconstruction
        delta_i = P_exact_ii - P_k_ii         per-item diagonal correction

    The exact diagonal P_exact_diag is computed during fit from the full
    eigendecomposition (all n eigenpairs), then the full matrices are discarded.
    Only V_k (n x k), lambda_inv_k (k,), P_k_diag (n,), P_exact_diag (n,),
    and delta (n,) are retained at serving time.

    The diagonal correction preserves exact per-item variance normalisation,
    keeping exploration calibrated regardless of rank choice. With rank = n_items,
    delta is identically zero and the result matches exact GaussianExplorationEASE.

    Predictive variance per item i:

        t          = x_u @ V_k                           O(nnz_u * k)
        cross_k    = (t * Λ_k^{-1}) @ V_k^T             O(k * n)
        cross_full[j] = cross_k[j] + x_u[j] * delta[j]  O(nnz_u)
        q          = t.(Λ_k^{-1} o t) + x_u^2 . delta   O(k + nnz_u)
        var_i      = q - cross_full_i^2 / P_exact_ii

    Memory at rest: O(n * k + n)  vs  O(n^2) for exact GaussianExplorationEASE.
    Training RAM peak is still O(n^2) — full eigh required for exact diagonal.

    Note
    ----
    item_factors and similar_items are not supported.

    Parameters
    ----------
    rank:
        Number of eigenvectors to retain.

        Reasonable values:

            [50, 100, 200, 300, 500]

        Usage:
            Increase rank when approximation quality matters more than memory.
            Decrease it when you need lower serving RAM or faster scoring.

    l2_scale:
        Scale-adaptive regularization inherited from EASE.

        Reasonable values:

            [0.03, 0.1, 0.3, 1.0, 3.0]

    random_state:
        Optional seed for reproducible exploration sampling.

        Usage:
            Set for benchmarking and leave unset for naturally stochastic
            recommendation serving.

    compute_dtype:
        Dense fit dtype.

        Reasonable values:

            [np.float64, np.float32]

    storage_dtype:
        Optional dtype used to store fitted low-rank arrays.

        Reasonable values:

            [None, np.float32, np.float64]
    """

    def __init__(
        self,
        rank: int = 200,
        l2_scale: float = 0.3,
        random_state: int | None = None,
        compute_dtype: np.dtype = np.float64,
        storage_dtype: np.dtype | None = None,
    ):
        super().__init__(
            l2_scale=l2_scale,
            random_state=random_state,
            compute_dtype=compute_dtype,
            storage_dtype=storage_dtype,
        )
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.rank = rank
        self.V_: np.ndarray | None = None
        self.lambda_inv_: np.ndarray | None = None
        self.P_k_diag_: np.ndarray | None = None
        self.P_exact_diag_: np.ndarray | None = None
        self.delta_: np.ndarray | None = None

    def fit(self, X: csr_matrix) -> "LowRankGaussianExplorationEASE":
        if not issparse(X):
            X = csr_matrix(X)

        X = X.tocsr().astype(self.compute_dtype)
        self.X_ = X

        G = (X.T @ X).toarray().astype(self.compute_dtype, copy=False)

        gram_diag = np.diag(G)
        gram_scale = float(np.mean(gram_diag))

        if gram_scale <= 0:
            raise ValueError(
                "Cannot fit LowRankGaussianExplorationEASE: "
                "mean(diag(X.T @ X)) is zero. "
                "The input matrix appears to contain no usable signal."
            )

        effective_l2 = self.l2_scale * gram_scale
        self.gram_scale_ = gram_scale
        self.effective_l2_ = effective_l2

        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += effective_l2

        # Full eigendecomposition is required: the exact diagonal of P needs
        # all n eigenpairs (P_exact_ii = sum_j V[i,j]^2 / lambda_j).
        # Training peak RAM: O(n^2). Only O(n*k + n) is retained after fit.
        eigenvalues, eigenvectors = np.linalg.eigh(G)  # ascending order

        if eigenvalues[0] <= 0:
            raise FloatingPointError(
                "Non-positive eigenvalue encountered. Try increasing l2_scale."
            )

        # Exact diagonal of P from all n eigenpairs.
        lambda_inv_all = 1.0 / eigenvalues
        P_exact_diag = (eigenvectors**2) @ lambda_inv_all

        if np.any(np.isclose(P_exact_diag, 0.0)):
            raise FloatingPointError(
                "Near-zero exact diagonal in P. Try increasing l2_scale."
            )

        # Top-k slice (largest eigenvalues are at the end in ascending order).
        k = min(self.rank, eigenvalues.shape[0])
        eigenvalues_k = eigenvalues[-k:][::-1].copy()
        eigenvectors_k = eigenvectors[:, -k:][:, ::-1].copy()

        lambda_inv_k = (1.0 / eigenvalues_k).astype(self.compute_dtype)
        P_k_diag = (eigenvectors_k**2) @ lambda_inv_k

        if np.any(np.isclose(P_k_diag, 0.0)):
            raise FloatingPointError(
                "Near-zero diagonal in low-rank P approximation. "
                "Try increasing l2_scale or rank."
            )

        delta = P_exact_diag - P_k_diag

        self.V_ = eigenvectors_k.astype(self.storage_dtype, copy=False)
        self.lambda_inv_ = lambda_inv_k.astype(self.storage_dtype, copy=False)
        self.P_k_diag_ = P_k_diag.astype(self.storage_dtype, copy=False)
        self.P_exact_diag_ = P_exact_diag.astype(
            self.storage_dtype, copy=False
        )
        self.delta_ = delta.astype(self.storage_dtype, copy=False)

        return self

    def _check_fitted(self) -> None:
        if (
            self.V_ is None
            or self.lambda_inv_ is None
            or self.P_exact_diag_ is None
            or self.effective_l2_ is None
        ):
            raise RuntimeError("Model is not fitted yet.")

    def _check_user_vector(self, user_vector: csr_matrix) -> None:
        self._check_fitted()

        if not issparse(user_vector):
            raise TypeError("user_vector must be a sparse matrix.")

        if user_vector.shape[0] != 1:
            raise ValueError(
                f"user_vector must have shape (1, n_items), got {user_vector.shape}."
            )

        if user_vector.shape[1] != self.V_.shape[0]:
            raise ValueError(
                f"user_vector has {user_vector.shape[1]} items, "
                f"but model was trained on {self.V_.shape[0]} items."
            )

    def _check_item_id(self, item_id: int) -> None:
        if self.V_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if item_id < 0 or item_id >= self.V_.shape[0]:
            raise IndexError(f"item_id={item_id} is out of bounds.")

    def _predict(
        self, user_vector: csr_matrix, mask_seen: bool = True
    ) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        x = user_vector.astype(self.compute_dtype, copy=False)
        t = np.asarray(x @ self.V_).ravel()  # (k,)
        cross = (t * self.lambda_inv_) @ self.V_.T  # (n,)

        # Diagonal correction only at history positions.
        if x.nnz > 0:
            cross = cross.copy()
            cross[x.indices] += x.data * self.delta_[x.indices]

        scores = -cross / self.P_exact_diag_

        if mask_seen:
            scores = self._mask_positive_interactions(scores, user_vector)

        return scores

    def _predictive_variance(self, user_vector: csr_matrix) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        n_items = self.V_.shape[0]

        if user_vector.nnz == 0:
            return np.zeros(n_items, dtype=self.storage_dtype)

        x = user_vector.astype(self.compute_dtype, copy=False)
        t = np.asarray(x @ self.V_).ravel()  # (k,)
        cross_k = (t * self.lambda_inv_) @ self.V_.T  # (n,)

        # Quadratic x_u P_approx x_u^T split into rank-k and correction parts.
        q_k = float(np.dot(t * self.lambda_inv_, t))
        x_sq_delta = float(np.dot(x.data**2, self.delta_[x.indices]))
        q = q_k + x_sq_delta

        cross_full = cross_k.copy()
        cross_full[x.indices] += x.data * self.delta_[x.indices]

        variance = q - (cross_full**2 / self.P_exact_diag_)

        return np.maximum(variance, 0.0).astype(self.storage_dtype, copy=False)

    def _sample_scores(
        self,
        user_vector: csr_matrix,
        mask_seen: bool = True,
        exploration_scale: float = 0.5,
    ) -> np.ndarray:
        self._check_fitted()
        self._check_user_vector(user_vector)

        mean = self._predict(user_vector, mask_seen=False)
        variance = self._predictive_variance(user_vector)
        std = np.sqrt(variance)

        sampled_scores = self.rng.normal(
            loc=mean,
            scale=exploration_scale * std,
        )

        if mask_seen:
            sampled_scores = self._mask_positive_interactions(
                sampled_scores, user_vector
            )

        return sampled_scores

    @property
    def item_factors(self) -> np.ndarray:
        """Low-rank item embeddings: row i is V_k[i] / sqrt(lambda_k).

        Satisfies F @ F.T = P_k, so inner products recover the rank-k
        approximation of the inverse Gram matrix.
        """
        self._check_fitted()
        return self.V_ * np.sqrt(self.lambda_inv_)

    def related_items(
        self,
        item_id: int,
        n: int | None = 10,
        exclude_self: bool = True,
    ) -> np.ndarray:
        self._check_fitted()
        self._check_item_id(item_id)

        # Off-diagonal P_k[item_id, i] = (V_k[item_id] * lambda_inv) . V_k[i]
        # B_k[item_id, i] = -P_k[item_id, i] / P_exact_ii  (for i != item_id)
        cross = (self.V_[item_id] * self.lambda_inv_) @ self.V_.T
        scores = -cross / self.P_exact_diag_

        if exclude_self:
            scores = scores.copy()
            scores[item_id] = -np.inf

        return self._top_n(scores, n)

    def similar_items(
        self,
        item_id: int,
        n: int | None = 10,
        exclude_self: bool = True,
    ) -> np.ndarray:
        """Find items with the most similar B_k row to item_id.

        Uses P_exact_diag for normalisation (consistent with _predict),
        so similarity reflects the diagonally-corrected model.
        Cost: O(k * n).
        """
        self._check_fitted()
        self._check_item_id(item_id)

        h = self.V_[item_id] * self.lambda_inv_
        c = h @ (self.V_ * self.lambda_inv_).T
        scores = c / (self.P_exact_diag_[item_id] * self.P_exact_diag_)

        if exclude_self:
            scores = scores.copy()
            scores[item_id] = -np.inf

        return self._top_n(scores, n)

    def _get_state(self) -> dict:
        return {
            "l2_scale": self.l2_scale,
            "rank": self.rank,
            "compute_dtype": self.compute_dtype,
            "storage_dtype": self.storage_dtype,
            "V_": self.V_,
            "lambda_inv_": self.lambda_inv_,
            "P_k_diag_": self.P_k_diag_,
            "P_exact_diag_": self.P_exact_diag_,
            "delta_": self.delta_,
            "effective_l2_": self.effective_l2_,
            "gram_scale_": self.gram_scale_,
            "rng_state": self.rng.bit_generator.state,
        }

    def _restore_state(self, state: dict) -> None:
        self.X_ = None
        self.rank = state["rank"]
        self.V_ = state["V_"]
        self.lambda_inv_ = state["lambda_inv_"]
        self.P_k_diag_ = state["P_k_diag_"]
        self.P_exact_diag_ = state["P_exact_diag_"]
        self.delta_ = state["delta_"]
        self.effective_l2_ = state["effective_l2_"]
        self.gram_scale_ = state["gram_scale_"]
        self.rng.bit_generator.state = state["rng_state"]

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "LowRankGaussianExplorationEASE":
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(
            rank=state["rank"],
            l2_scale=state["l2_scale"],
            random_state=None,
            compute_dtype=state["compute_dtype"],
            storage_dtype=state["storage_dtype"],
        )
        instance._restore_state(state)
        return instance


class TEASE(GaussianExplorationEASE):
    """Public library name for GaussianExplorationEASE.

    TEASE stands for Thompson-style EASE. The algorithm uses Gaussian score
    sampling rather than exact Bayesian posterior sampling over interaction
    probabilities, so the method should be described as approximate
    Thompson-style exploration in user-facing documentation.

    Hyperparameters
    ---------------
    l2_scale:
        Reasonable values:

            [0.03, 0.1, 0.3, 1.0, 3.0]

        Use smaller values for sharper, more exploratory rankings and larger
        values for smoother, more conservative rankings.

    exploration_scale:
        Reasonable values:

            [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]

        Usage:
            0.0 disables exploration and reduces TEASE to deterministic EASE.
            0.1-0.5 is a good production starting range. 1.0 uses the Gaussian
            posterior approximation directly. Values above 1.0 are usually only
            useful when you intentionally want aggressive exploration.

    random_state:
        Use an integer seed for reproducible evaluation or debugging.
    """


class LowRankTEASE(LowRankGaussianExplorationEASE):
    """Low-rank public alias for LowRankGaussianExplorationEASE.

    Hyperparameters
    ---------------
    rank:
        Reasonable values:

            [50, 100, 200, 300, 500]

        Usage:
            Start near 200. Increase rank to recover more exact-model quality.
            Decrease rank to reduce serving memory and speed up scoring.

    l2_scale:
        Reasonable values:

            [0.03, 0.1, 0.3, 1.0, 3.0]

    exploration_scale:
        Reasonable values:

            [0.0, 0.1, 0.25, 0.5, 1.0]

        Usage:
            Tune this after rank and l2_scale. If results feel too noisy,
            reduce exploration_scale or increase l2_scale.
    """
