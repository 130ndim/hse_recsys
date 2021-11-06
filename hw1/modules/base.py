from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as la

import scipy as sp
import scipy.sparse


class MatrixFactorizationBase(ABC):
    _U: np.ndarray
    _I: np.ndarray

    def __init__(
            self,
            k: int,
            n_iter: int,
            verbose: bool = True,
            random_state: Optional[int] = None
    ) -> None:
        self.k = k
        self.n_iter = n_iter
        self.verbose = verbose
        np.random.seed(random_state)

    def _init_users_and_items(self, n_users: int, n_items: int) -> None:
        self._U = np.random.uniform(
            -self.k ** -0.5, self.k ** -0.5, size=(n_users, self.k)
        )
        self._I = np.random.uniform(
            -self.k ** -0.5, self.k ** -0.5, size=(n_items, self.k)
        )

    @abstractmethod
    def fit(self, R: np.ndarray) -> "MatrixFactorizationBase":
        pass

    def recommend(
            self,
            user_id: int,
            user_item: sp.sparse.spmatrix,
            n: int = 10
    ) -> List[Tuple[int, float]]:
        user_interactions = user_item.tocsr()[user_id].nonzero()[1]
        scores = self._I @ self._U[user_id]
        scores[user_interactions] = -float('inf')
        item_ids = np.argpartition(-scores, n)[:n]
        out = sorted([(i, scores[i]) for i in item_ids], key=lambda x: -x[1])
        return out

    def similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        item = self._I[item_id]
        cs = self._I @ item / la.norm(item, 2) / la.norm(self._I, 2, axis=1)
        sim_ids = np.argpartition(1 - cs, n + 1)[:n + 1]
        out = sorted([(i, cs[i]) for i in sim_ids], key=lambda x: -x[1])[1:]
        return out
