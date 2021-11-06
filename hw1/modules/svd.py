from typing import Any, List, Tuple
import random

import numpy as np
import scipy as sp
import scipy.sparse

from tqdm.auto import tqdm

from modules.base import MatrixFactorizationBase


class SVD(MatrixFactorizationBase):
    _ub: np.ndarray
    _ib: np.ndarray
    _avg_bias: float

    def __init__(
            self,
            k: int,
            n_iter: int = 50,
            learning_rate: float = 1e-2,
            lambda_: float = 1e-2,
            verbose: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(k, n_iter, verbose, **kwargs)
        self.learning_rate = learning_rate
        self.lambda_ = lambda_

    def _init_users_and_items(self, n_users: int, n_items: int) -> None:
        super()._init_users_and_items(n_users, n_items)
        self._ub = np.zeros(shape=(n_users,))
        self._ib = np.zeros(shape=(n_items,))

    def fit(self, R: sp.sparse.spmatrix) -> "SVD":
        lr, l2 = self.learning_rate, self.lambda_
        R_coo = R.tocoo()
        self._init_users_and_items(R.shape[0], R.shape[1])
        self._avg_bias = R_coo.mean().item()

        uiv = list(zip(R_coo.row, R_coo.col, R_coo.data))

        with tqdm(total=self.n_iter, position=0, disable=not self.verbose) as pbar:
            for _ in range(self.n_iter):
                pbar.set_description(f'Iteration')
                random.shuffle(uiv)
                loss = 0
                with tqdm(
                        total=len(uiv),
                        disable=not self.verbose,
                        leave=False,
                        position=0,
                        desc='Fitting'
                ) as inner_pbar:

                    for user_idx, item_idx, rating in uiv:
                        user, item = self._U[user_idx], self._I[item_idx]
                        r_hat = user @ item + self._ub[user_idx] + self._ib[item_idx]
                        r_hat += self._avg_bias

                        e = rating - r_hat

                        loss += e ** 2

                        self._U[user_idx], self._I[item_idx] = (
                            user + lr * (e * item - l2 * user),
                            item + lr * (e * user - l2 * item)
                        )
                        self._ub[user_idx], self._ib[item_idx] = (
                            self._ub[user_idx] + lr * (e - l2 * self._ub[user_idx]),
                            self._ib[item_idx] + lr * (e - l2 * self._ib[item_idx])
                        )
                        inner_pbar.update(1)

                loss /= len(uiv)
                reg = sum([
                    np.linalg.norm(self._U, 2) ** 2,
                    np.linalg.norm(self._I, 2) ** 2,
                    np.linalg.norm(self._ub, 2) ** 2,
                    np.linalg.norm(self._ib, 2) ** 2
                ])
                pbar.set_postfix_str(
                    f'MSE = {loss:.4f}, RMSE = {loss ** 0.5:.4f}, L2 = {reg:.2f}'
                )
                pbar.update(1)

        return self

    def recommend(
            self,
            user_id: int,
            user_item: sp.sparse.spmatrix,
            n: int = 10
    ) -> List[Tuple[int, float]]:
        user_interactions = user_item.tocsr()[user_id].nonzero()[1]
        scores = self._I @ self._U[user_id] + self._ib + self._ub[user_id]
        scores += self._avg_bias
        scores[user_interactions] = -float('inf')
        item_ids = np.argpartition(-scores, n)[:n]
        out = sorted([(i, scores[i]) for i in item_ids], key=lambda x: -x[1])
        return out
