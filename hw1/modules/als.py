from typing import Any

import numpy as np
import scipy as sp
import scipy.sparse

from tqdm.auto import tqdm

from modules.base import MatrixFactorizationBase


def mse_loss(U: np.ndarray, I: np.ndarray, R: sp.sparse.spmatrix) -> float:
    user_id, item_id = R.nonzero()
    error = ((1 - (U[user_id] * I[item_id]).sum(-1)) ** 2).mean()
    return error


class ALS(MatrixFactorizationBase):
    def __init__(
            self,
            k: int,
            n_iter: int = 50,
            lambda_: float = 1e-2,
            verbose: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(k, n_iter, verbose, **kwargs)
        self.lambda_ = lambda_

    def _step(self, X: np.ndarray, Y: np.ndarray, R: sp.sparse.spmatrix) -> np.ndarray:
        X = X.copy()
        YTY_reg = Y.T @ Y + np.eye(self.k) * self.lambda_
        for i in range(X.shape[0]):
            X[i] = np.linalg.solve(YTY_reg, (R[i] @ Y).ravel())
        return X

    def fit(self, R: sp.sparse.spmatrix) -> "ALS":
        R_csr = R.tocsr()
        self._init_users_and_items(R.shape[0], R.shape[1])

        with tqdm(total=self.n_iter, position=0, disable=not self.verbose) as pbar:
            for _ in range(self.n_iter):
                pbar.set_description(f'Iteration')

                self._U = self._step(self._U, self._I, R_csr)
                self._I = self._step(self._I, self._U, R_csr.T)

                loss = mse_loss(self._U, self._I, R_csr)
                pbar.set_postfix_str(
                    f'MSE = {loss:.4f}, RMSE = {loss ** 0.5:.4f}'
                )
                pbar.update(1)

        return self
