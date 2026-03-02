from __future__ import annotations

from dataclasses import dataclass


# COPIED-AND-ADAPTED FROM:
# https://github.com/Quandela/HybridAIQuantum-Challenge
# Source path: src/qml/utils_budget.py (mirrored locally in aaquandela/qml/utils_budget.py)
@dataclass
class BudgetCounter:
    """Tracks quantum and algorithmic feature-evaluation budgets."""

    shots: int = 128
    train_qevals: int = 0
    infer_qevals: int = 0
    feat_evals_train: int = 0
    feat_evals_infer: int = 0

    def add_feat_evals(self, n: int, phase: str) -> None:
        n_int = int(max(0, n))
        if phase == "train":
            self.feat_evals_train += n_int
        elif phase == "infer":
            self.feat_evals_infer += n_int
        else:
            raise ValueError(f"Unknown phase for feat-eval accounting: {phase}")

    def add_qevals(self, n: int, phase: str) -> None:
        n_int = int(max(0, n))
        if phase == "train":
            self.train_qevals += n_int
        elif phase == "infer":
            self.infer_qevals += n_int
        else:
            raise ValueError(f"Unknown phase for qeval accounting: {phase}")
        self.add_feat_evals(n_int, phase=phase)

    @property
    def train_total_shots(self) -> int:
        return int(self.train_qevals * self.shots)

    @property
    def infer_total_shots(self) -> int:
        return int(self.infer_qevals * self.shots)

    @property
    def total_qevals(self) -> int:
        return int(self.train_qevals + self.infer_qevals)

    @property
    def total_shots(self) -> int:
        return int(self.train_total_shots + self.infer_total_shots)

    def as_dict(self) -> dict[str, int]:
        return {
            "shots_per_eval": int(self.shots),
            "train_qevals": int(self.train_qevals),
            "train_total_shots": int(self.train_total_shots),
            "infer_qevals": int(self.infer_qevals),
            "infer_total_shots": int(self.infer_total_shots),
            "total_qevals": int(self.total_qevals),
            "total_shots": int(self.total_shots),
            "train_feat_evals": int(self.feat_evals_train),
            "infer_feat_evals": int(self.feat_evals_infer),
        }
