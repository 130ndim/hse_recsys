from typing import Any, Iterable, Tuple

import numpy as np
import scipy as sp
import scipy.sparse
from scipy.special import expit

from tqdm.auto import tqdm

from modules.base import MatrixFactorizationBase


class BPR(MatrixFactorizationBase):
    def __init__(
            self,
            k: int,
            n_iter: int = 250,
            learning_rate: float = 1e-2,
            lambda_: float = 1e-2,
            verbose: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(k, n_iter, verbose, **kwargs)
        self.learning_rate = learning_rate
        self.lambda_ = lambda_

    def _iter_triplets(self, R: sp.sparse.spmatrix) -> Iterable[Tuple[int, int, int]]:
        for user_id in np.random.permutation(self._U.shape[0]):
            pos_items = R[user_id].nonzero()[1]
            if len(pos_items) == 0:
                continue
            pos_item_id = np.random.choice(pos_items)
            while True:
                neg_item_id = np.random.choice(R.shape[1])
                if R[user_id, neg_item_id] == 0:
                    break
            yield user_id, pos_item_id, neg_item_id

    def fit(self, R: sp.sparse.spmatrix) -> "BPR":
        R_csr = R.tocsr()
        self._init_users_and_items(R.shape[0], R.shape[1])
        with tqdm(total=self.n_iter, position=0, disable=not self.verbose) as pbar:
            for _ in range(self.n_iter):
                pbar.set_description(f'Iteration')
                correct = 0
                with tqdm(
                        total=self._U.shape[0],
                        disable=not self.verbose,
                        leave=False,
                        position=0,
                        desc='Fitting'
                ) as inner_pbar:
                    for user_id, pos_item_id, neg_item_id in self._iter_triplets(R_csr):
                        user = self._U[user_id]
                        pos_item, neg_item = self._I[pos_item_id], self._I[neg_item_id]
                        score = expit(user @ (neg_item - pos_item))

                        correct += score < 0.5

                        du = score * (neg_item - pos_item) + self.lambda_ * user
                        dip = -score * user + self.lambda_ * pos_item
                        din = score * user + self.lambda_ * neg_item

                        self._U[user_id] -= self.learning_rate * du
                        self._I[pos_item_id] -= self.learning_rate * dip
                        self._I[neg_item_id] -= self.learning_rate * din

                        inner_pbar.update(1)

                    inner_pbar.update(inner_pbar.total - inner_pbar.n)

                pbar.set_postfix_str(f'ACC = {correct / self._U.shape[0]:.4f}')
                pbar.update(1)

        return self
