"""HMM 转移矩阵校准：带人工标签约束的 EM 与留出集条件对数概率评估。

业务背景
--------
电台用人工校对过的录音估计各语种间的切换倾向。每条录音由逐帧后验
（观测似然）与可选的人工语种标签组成：

* **训练录音**标签可缺（整条缺标或逐帧 ``null`` 均可）；
* **留出录音**必须逐帧标全，只用于打分、不参与 EM。

算法约定（与需求逐条对应）
--------------------------
1. 沿用在线会话的 HMM 观测规则：每帧后验按行归一化后取对数（下限
   1e-12）作为发射对数似然；**每条录音首帧使用均匀语种先验**
   ``log pi(j) = -log(n)``，与在线会话首帧一致。
2. 起步矩阵为"切换惩罚 2"矩阵：保持给零对数代价、切换给 -2 纳特，
   再按行 logsumexp 归一化（与 :class:`~app.smoother.LanguageSmoother`
   的构造完全相同）。
3. 共跑 **5 轮 EM**。已标语种把对应帧的隐状态硬约束为该语种（缺标帧
   保持全部状态开放）；在约束下做前向-后向，统计所有录音、所有相邻
   帧的期望转移次数 ``xi_t(i,j)``。
4. 每轮把期望计数矩阵的**每种转移加 1 个伪计数**，再按行归一化得到
   下一轮的转移矩阵。
5. 留出打分：对每条留出录音，求真实语种序列在观测下的**条件对数概率**
   ``log P(z_1:T | x_1:T) = log P(x, z) - log P(x)``，其中证据项
   ``log P(x)`` 由前向算法得到；各录音得分直接相加。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_ROUNDS = 5
CALIBRATION_PENALTY = 2.0
MAX_CALIBRATION_FRAMES = 1000
_OBS_FLOOR = 1e-12

# 一条录音：(逐帧后验 T×n, 逐帧语种下标 T 元素；-1 表示该帧缺标)。
PreparedRecording = Tuple[np.ndarray, Optional[np.ndarray]]


class CalibrationError(Exception):
    """校准请求的业务错误，由 API 层映射为 4xx。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


# --------------------------------------------------------------------- 矩阵
def penalty_matrix(n: int, penalty: float = CALIBRATION_PENALTY) -> np.ndarray:
    """切换惩罚矩阵（概率域，逐行归一化）。

    与 ``LanguageSmoother`` 内部构造保持同一规则：
    ``log A(i,i)=0``、``log A(i,j)=-penalty (i!=j)`` 后逐行归一化。
    """
    log_a = np.zeros((n, n), dtype=np.float64)
    off = ~np.eye(n, dtype=bool)
    log_a[off] = -float(penalty)
    log_a -= _logsumexp(log_a, axis=1)[:, None]
    return np.exp(log_a)


def reorder_matrix(
    matrix: np.ndarray, old_order: Sequence[str], new_order: Sequence[str]
) -> np.ndarray:
    """把按 ``old_order`` 排列的矩阵重排成 ``new_order`` 的列/行顺序。"""
    pos = {lang.lower(): i for i, lang in enumerate(old_order)}
    idx = [pos[lang.lower()] for lang in new_order]
    return matrix[np.ix_(idx, idx)].copy()


def _to_log_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(matrix, _OBS_FLOOR))


# ------------------------------------------------------------------ 输入整理
def prepare_recordings(
    languages: Sequence[str],
    raw_recordings: Sequence[dict],
    *,
    require_labels: bool,
) -> List[PreparedRecording]:
    """把 API 层的录音字典整理成数值数组并做语义校验。

    ``raw_recordings`` 元素形如 ``{"posteriors": [[...]], "labels": [...]}``。
    训练录音 ``labels`` 可缺（整体为 None 或逐帧 null）；留出录音必须逐帧
    标全，否则抛 :class:`CalibrationError`。
    """
    n = len(languages)
    lang_pos = {lang.lower(): i for i, lang in enumerate(languages)}
    prepared: List[PreparedRecording] = []

    for rec_idx, rec in enumerate(raw_recordings):
        rows = rec["posteriors"]
        if not rows:
            raise CalibrationError(
                422, "empty_recording", f"recording {rec_idx} must contain at least one frame"
            )
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != n:
            raise CalibrationError(
                422,
                "dimension_mismatch",
                f"recording {rec_idx}: expected {n} posteriors per frame, got {arr.shape[1]}",
            )
        if not np.all(np.isfinite(arr)):
            raise CalibrationError(422, "non_finite", f"recording {rec_idx}: posteriors contain NaN or Inf")
        if np.any(arr < 0.0):
            raise CalibrationError(
                422, "negative_probability", f"recording {rec_idx}: posteriors must be >= 0"
            )
        if np.any(arr.sum(axis=1) <= 0.0):
            raise CalibrationError(
                422, "invalid_distribution", f"recording {rec_idx}: each frame must sum to > 0"
            )

        raw_labels = rec.get("labels")
        if raw_labels is None:
            if require_labels:
                raise CalibrationError(
                    422,
                    "heldout_labels_required",
                    f"held-out recording {rec_idx} must carry labels for every frame",
                )
            prepared.append((arr, None))
            continue

        if len(raw_labels) != arr.shape[0]:
            raise CalibrationError(
                422,
                "label_length_mismatch",
                f"recording {rec_idx}: {len(raw_labels)} labels for {arr.shape[0]} frames",
            )
        idx = np.empty(arr.shape[0], dtype=np.int64)
        for t, label in enumerate(raw_labels):
            if label is None:
                if require_labels:
                    raise CalibrationError(
                        422,
                        "heldout_labels_required",
                        f"held-out recording {rec_idx}: frame {t} is unlabeled",
                    )
                idx[t] = -1
                continue
            if not isinstance(label, str) or label.strip().lower() not in lang_pos:
                raise CalibrationError(
                    422,
                    "unknown_label",
                    f"recording {rec_idx} frame {t}: label {label!r} is not one of {list(languages)}",
                )
            idx[t] = lang_pos[label.strip().lower()]
        prepared.append((arr, idx))

    return prepared


def _log_observations(posteriors: np.ndarray) -> np.ndarray:
    """沿用在线 HMM 观测规则：按行归一化，对数加 1e-12 下限。"""
    obs = posteriors / posteriors.sum(axis=1, keepdims=True)
    return np.log(np.maximum(obs, _OBS_FLOOR))


def _state_mask(labels: Optional[np.ndarray], n: int, length: int) -> np.ndarray:
    """(T, n) 布尔掩码：True 表示该帧允许处于该隐状态。"""
    mask = np.ones((length, n), dtype=bool)
    if labels is None:
        return mask
    for t, state in enumerate(labels):
        if state >= 0:
            mask[t, :] = False
            mask[t, state] = True
    return mask


# ----------------------------------------------------------- 前向-后向 / EM
def expected_transition_counts(
    log_obs: np.ndarray,
    log_a: np.ndarray,
    labels: Optional[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """单条录音在（标签约束下的）前向-后验期望转移计数。

    返回 ``(counts, log_evidence)``，``counts[i,j]`` 是本录音所有相邻帧
    ``xi_t(i,j)`` 之和；单帧录音计数为零矩阵。
    """
    t_len, n = log_obs.shape
    mask = _state_mask(labels, n, t_len)
    log_pi = np.full(n, -np.log(n), dtype=np.float64)  # 首帧均匀先验

    with np.errstate(divide="ignore"):
        # 前向。
        log_alpha = np.full((t_len, n), -np.inf, dtype=np.float64)
        log_alpha[0] = np.where(mask[0], log_pi + log_obs[0], -np.inf)
        for t in range(1, t_len):
            propagated = _logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
            log_alpha[t] = np.where(mask[t], propagated + log_obs[t], -np.inf)
        log_evidence = float(_logsumexp(log_alpha[-1]))

        # 后向。
        log_beta = np.full((t_len, n), -np.inf, dtype=np.float64)
        log_beta[-1] = np.where(mask[-1], 0.0, -np.inf)
        for t in range(t_len - 2, -1, -1):
            back = _logsumexp(
                log_a + (log_obs[t + 1] + log_beta[t + 1])[None, :], axis=1
            )
            log_beta[t] = np.where(mask[t], back, -np.inf)

        # xi_t(i,j) = alpha_t(i) A(i,j) obs_{t+1}(j) beta_{t+1}(j) / evidence
        if t_len >= 2:
            pair = (
                log_alpha[:-1][:, :, None]
                + log_a[None, :, :]
                + (log_obs[1:] + log_beta[1:])[:, None, :]
            )
            # 目标帧的标签约束：不允许的 j 整列置 -inf。
            pair = np.where(mask[1:][:, None, :], pair, -np.inf)
            log_xi = pair - log_evidence
            counts = np.sum(np.exp(log_xi), axis=0)
        else:
            counts = np.zeros((n, n), dtype=np.float64)

    return counts, log_evidence


def run_em(
    train_records: Sequence[PreparedRecording],
    n: int,
    *,
    rounds: int = CALIBRATION_ROUNDS,
    penalty: float = CALIBRATION_PENALTY,
) -> np.ndarray:
    """从切换惩罚矩阵起步，做 ``rounds`` 轮带标签约束的 EM。"""
    matrix = penalty_matrix(n, penalty)
    cached = [(_log_observations(rec[0]), rec[1]) for rec in train_records]

    for _ in range(rounds):
        log_a = _to_log_matrix(matrix)
        counts = np.zeros((n, n), dtype=np.float64)
        for log_obs, labels in cached:
            rec_counts, _ = expected_transition_counts(log_obs, log_a, labels)
            counts += rec_counts
        # 每种转移加 1 个伪计数，再按行归一化。
        counts += 1.0
        matrix = counts / counts.sum(axis=1, keepdims=True)
    return matrix


# ------------------------------------------------------------------ 留出打分
def sequence_conditional_logprob(
    log_obs: np.ndarray,
    label_idx: np.ndarray,
    log_a: np.ndarray,
) -> float:
    """单条录音真实语种序列的条件对数概率 ``log P(z|x)``。"""
    t_len, n = log_obs.shape
    log_pi = np.full(n, -np.log(n), dtype=np.float64)

    # 联合对数概率 log P(x, z)：均匀先验 + 真实路径上的发射与转移。
    joint = float(log_pi[label_idx[0]] + log_obs[0, label_idx[0]])
    for t in range(1, t_len):
        joint += float(
            log_a[label_idx[t - 1], label_idx[t]] + log_obs[t, label_idx[t]]
        )

    # 证据 log P(x)：无约束前向（首帧同样使用均匀先验）。
    alpha = log_pi + log_obs[0]
    for t in range(1, t_len):
        alpha = _logsumexp(alpha[:, None] + log_a, axis=0) + log_obs[t]
    log_evidence = float(_logsumexp(alpha))

    return joint - log_evidence


def held_out_score(
    held_out_records: Sequence[PreparedRecording], matrix: np.ndarray
) -> float:
    """各留出录音真实序列条件对数概率之和。"""
    log_a = _to_log_matrix(matrix)
    total = 0.0
    for posteriors, labels in held_out_records:
        assert labels is not None  # 留出录音须标全，由输入校验保证。
        total += sequence_conditional_logprob(_log_observations(posteriors), labels, log_a)
    return total


# --------------------------------------------------------------------- 编排
@dataclass
class CalibrationOutcome:
    languages: List[str]
    baseline_matrix: np.ndarray       # 请求语种顺序
    candidate_matrix: np.ndarray      # 请求语种顺序
    baseline_score: float
    candidate_score: float
    enabled: bool
    train_frames: int
    held_out_frames: int

    @property
    def active_matrix(self) -> np.ndarray:
        return self.candidate_matrix if self.enabled else self.baseline_matrix

    @property
    def active_source(self) -> str:
        return "candidate" if self.enabled else "baseline"


def calibrate(
    languages: Sequence[str],
    train_records: Sequence[PreparedRecording],
    held_out_records: Sequence[PreparedRecording],
    baseline_matrix: np.ndarray,
) -> CalibrationOutcome:
    """跑 EM 并按"留出分数严格更高才替换"决定是否启用候选矩阵。"""
    n = len(languages)
    candidate = run_em(train_records, n)
    baseline_score = held_out_score(held_out_records, baseline_matrix)
    candidate_score = held_out_score(held_out_records, candidate)
    enabled = candidate_score > baseline_score
    train_frames = int(sum(rec[0].shape[0] for rec in train_records))
    held_out_frames = int(sum(rec[0].shape[0] for rec in held_out_records))
    return CalibrationOutcome(
        languages=list(languages),
        baseline_matrix=np.asarray(baseline_matrix, dtype=np.float64),
        candidate_matrix=candidate,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        enabled=enabled,
        train_frames=train_frames,
        held_out_frames=held_out_frames,
    )
