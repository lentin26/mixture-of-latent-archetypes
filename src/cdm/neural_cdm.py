"""NeuralCDM (EduCDM's NCDM) wrapper for the EdNet high-dimensional
evaluation (see notebooks/high-dim-experiment.ipynb and
src/experiments/high_dim.py).

Matches this repo's existing CDM-wrapper interface shape
(src/cdm/cdm_model.py, src/cdm/dina.py) as closely as the two-stage
protocol below allows: `fit(...)`, `predict_item_proba(...)`.

NCDM (`EduCDM.NCDM`) represents each student's proficiency as one row of
a fixed-size `nn.Embedding(student_n, knowledge_n)`, learned jointly with
the item parameters during training -- there is no built-in way to infer
an unseen student's representation afterward (confirmed by reading
`EduCDM/NCDM/NCDM.py` directly: `train`/`eval`/`save`/`load` are the only
public methods, and `eval` only returns `(auc, accuracy)`, never raw
probabilities). It also fixes the student embedding's dimensionality to
`knowledge_n` internally (`Net.stu_dim = self.knowledge_dim`) -- there is
no independent `stu_dim` to configure, so this wrapper doesn't expose one
either.

This wrapper implements the two-stage protocol the evaluation needs:

1. `fit`: train NCDM's population parameters (item difficulty/
   discrimination embeddings and the prediction sub-network) plus the
   parameter-estimation learners' own embedding rows, exactly as EduCDM's
   own example does. Every resulting parameter is then frozen.
2. `profile_learner`: infer one new (previously unseen) learner's
   proficiency from a small set of profiling responses, with every
   population parameter frozen. Rather than literally growing
   `student_emb`'s embedding table -- `requires_grad` is a whole-tensor
   property on a leaf parameter, so masking gradients on individual rows
   of a *shared* table is possible but fiddly to get exactly right --
   this creates one fresh, small `nn.Parameter` per new learner and
   reuses the network's *other* (frozen) layers directly via
   `_forward_from_stu_emb`, a copy of `Net.forward` from the
   `stat_emb = sigmoid(stu_emb)` line onward that takes the student
   embedding as an argument instead of looking it up from the table.
   Since every `net.*` parameter has `requires_grad=False` by this point,
   they never enter the autograd graph at all during profiling -- not
   just "excluded from the optimizer step" but structurally unreachable
   by `loss.backward()`, which is a stronger and more easily verified
   guarantee than gradient-masking would give.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from EduCDM import NCDM


def _forward_from_stu_emb(
    net: nn.Module,
    stu_emb: torch.Tensor,
    input_exercise: torch.Tensor,
    input_knowledge_point: torch.Tensor,
) -> torch.Tensor:
    """Reimplements EduCDM's `Net.forward` from the
    `stat_emb = sigmoid(stu_emb)` line onward, taking the student
    embedding directly as an argument instead of looking it up via
    `net.student_emb(stu_id)`. Lets a new learner's profiling pass reuse
    every other (frozen) layer of `net` without touching the shared
    student embedding table at all.
    """
    stat_emb = torch.sigmoid(stu_emb)
    k_difficulty = torch.sigmoid(net.k_difficulty(input_exercise))
    e_difficulty = torch.sigmoid(net.e_difficulty(input_exercise))
    input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
    input_x = net.drop_1(torch.sigmoid(net.prednet_full1(input_x)))
    input_x = net.drop_2(torch.sigmoid(net.prednet_full2(input_x)))
    output_1 = torch.sigmoid(net.prednet_full3(input_x))
    return output_1.view(-1)


class NeuralCDM:
    """Wraps `EduCDM.NCDM` for the fit-then-freeze-then-profile protocol.

    Parameters
    ----------
    n_skills, n_items : int
        Q-matrix dimensions.
    n_learners_param_est : int
        Number of learners in the parameter-estimation set -- sizes
        NCDM's own student-embedding table. Row `i` of that table holds
        parameter-estimation learner `i`'s fitted proficiency and is
        frozen after `fit`; evaluation-learner profiling never touches
        this table (see module docstring).
    epoch : int
        Epochs for the population-parameter fit (`fit`).
    profile_epochs : int
        Epochs for one learner's profiling fit (`profile_learner`).
    lr, profile_lr : float
        Learning rates for the two stages.
    random_state : int or None
        Seed for `NCDM`'s weight initialization (here, in `__init__`,
        before the network is constructed) and, separately, re-applied at
        the start of `fit` (before `DataLoader(..., shuffle=True)` draws
        its shuffle order) -- PyTorch has no equivalent of a per-instance
        RNG the way MoLA's `random_state` uses NumPy's `Generator`, so
        reproducibility here means reseeding torch's global RNG at each
        point that consumes it, not just once at construction. `None`
        (the default) leaves torch's RNG exactly as the caller left it --
        matches prior behavior, but results then depend on what else ran
        in the same process before this call, not just this call's own
        arguments.
    """

    def __init__(
        self,
        n_skills: int,
        n_items: int,
        n_learners_param_est: int,
        epoch: int = 10,
        profile_epochs: int = 30,
        lr: float = 0.002,
        profile_lr: float = 0.01,
        device: str = "cpu",
        batch_size: int = 256,
        random_state: int | None = None,
    ):
        self.n_skills = n_skills
        self.n_items = n_items
        self.n_learners_param_est = n_learners_param_est
        self.epoch = epoch
        self.profile_epochs = profile_epochs
        self.lr = lr
        self.profile_lr = profile_lr
        self.device = device
        self.batch_size = batch_size
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
        self.model = NCDM(n_skills, n_items, n_learners_param_est)
        self.train_time_sec: float | None = None

    def _make_loader(self, df: pd.DataFrame, item2knowledge: np.ndarray) -> DataLoader:
        """Builds the `(user_id, item_id, knowledge_emb, y)`
        TensorDataset/DataLoader `NCDM.train` expects. `df` must have
        0-indexed `user_row` (< `n_learners_param_est`) and `item_row`
        (< `n_items`) columns plus binary `correct`.
        """
        user_ids = torch.tensor(df["user_row"].to_numpy(), dtype=torch.int64)
        item_rows = df["item_row"].to_numpy()
        item_ids = torch.tensor(item_rows, dtype=torch.int64)
        knowledge = torch.tensor(item2knowledge[item_rows], dtype=torch.float32)
        y = torch.tensor(df["correct"].to_numpy(), dtype=torch.float32)
        dataset = TensorDataset(user_ids, item_ids, knowledge, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fit(self, train_df: pd.DataFrame, item2knowledge: np.ndarray) -> "NeuralCDM":
        """Fits NCDM's population parameters on the parameter-estimation
        learners' responses, then freezes every resulting parameter.
        `item2knowledge` is the `(n_items, n_skills)` Q-matrix.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # controls _make_loader's DataLoader(shuffle=True) order
        start = time.perf_counter()
        loader = self._make_loader(train_df, item2knowledge)
        self.model.train(loader, epoch=self.epoch, device=self.device, lr=self.lr, silence=True)
        self.train_time_sec = time.perf_counter() - start

        for param in self.model.ncdm_net.parameters():
            param.requires_grad_(False)
        self.model.ncdm_net.zero_grad(set_to_none=True)  # clear stale grads left over from training
        self.model.ncdm_net.eval()  # also disables dropout for profiling/prediction
        return self

    def profile_learner(
        self, profile_df: pd.DataFrame, item2knowledge: np.ndarray, seed: int | None = None
    ) -> torch.Tensor:
        """Infers one new learner's proficiency embedding from their
        profiling-period responses, with every population parameter
        frozen. `profile_df` needs only `item_row` and `correct` columns
        (no `user_row` -- there is exactly one learner here). Returns the
        fitted `(1, n_skills)` embedding, detached -- pass it to
        `predict_item_proba`.

        `seed` (falling back to `self.random_state` if not given) reseeds
        torch's global RNG immediately before `stu_emb`'s Xavier-normal
        initialization, the only randomness in this method -- without it,
        results depend on whatever else ran in this process beforehand.
        """
        seed = self.random_state if seed is None else seed
        if seed is not None:
            torch.manual_seed(seed)

        net = self.model.ncdm_net
        item_rows = profile_df["item_row"].to_numpy()
        item_ids = torch.tensor(item_rows, dtype=torch.int64)
        knowledge = torch.tensor(item2knowledge[item_rows], dtype=torch.float32)
        y = torch.tensor(profile_df["correct"].to_numpy(), dtype=torch.float32)

        stu_emb = nn.Parameter(torch.empty(1, self.n_skills))
        nn.init.xavier_normal_(stu_emb)  # matches NCDM's own init convention for embedding weights
        optimizer = torch.optim.Adam([stu_emb], lr=self.profile_lr)
        loss_fn = nn.BCELoss()

        for _ in range(self.profile_epochs):
            pred = _forward_from_stu_emb(net, stu_emb.expand(len(item_ids), -1), item_ids, knowledge)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return stu_emb.detach()

    def predict_item_proba(
        self, stu_emb: torch.Tensor, item_rows: np.ndarray, item2knowledge: np.ndarray
    ) -> np.ndarray:
        """Predicts response probabilities for `item_rows` given a fitted
        learner embedding (from `profile_learner`, or
        `population_learner_embedding` for a parameter-estimation
        learner). Returns a `(len(item_rows),)` numpy array.
        """
        net = self.model.ncdm_net
        item_ids = torch.tensor(item_rows, dtype=torch.int64)
        knowledge = torch.tensor(item2knowledge[item_rows], dtype=torch.float32)
        with torch.no_grad():
            pred = _forward_from_stu_emb(net, stu_emb.expand(len(item_rows), -1), item_ids, knowledge)
        return pred.cpu().numpy()

    def population_learner_embedding(self, user_row: int) -> torch.Tensor:
        """Returns a parameter-estimation learner's own fitted embedding
        (row `user_row` of the frozen student embedding table) -- mainly
        useful for sanity-checking `fit` against `predict_item_proba`
        using the same code path `profile_learner`'s output does."""
        with torch.no_grad():
            return self.model.ncdm_net.student_emb.weight[user_row : user_row + 1].clone()
