"""FastAPI 服务：流式语种识别抖动 → 稳定分段，外加 HMM 转移矩阵校准。"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .calibration import (
    CALIBRATION_PENALTY,
    CALIBRATION_ROUNDS,
    MAX_CALIBRATION_FRAMES,
    CalibrationError,
    calibrate,
    penalty_matrix,
    prepare_recordings,
    reorder_matrix,
)
from .smoother import ChunkResult, LanguageSmoother, SessionError

app = FastAPI(
    title="Streaming Language Segmenter",
    version="1.1.0",
    description=(
        "把流式语种识别的逐帧后验整理成稳定语种区间。"
        "会话设定 2~4 个语种、帧长、切换连续帧数与后验阈值；"
        "服务用带切换惩罚的 HMM 平滑状态，只在满足连续帧阈值时切段。"
        "另提供 /calibration：用人工校对录音做 5 轮带标签约束的 EM，"
        "按留出集条件对数概率决定是否启用校准矩阵；新会话可选用同语种组"
        "已启用的矩阵，已开会话保留建会话时的矩阵。"
    ),
)

_sessions: Dict[str, LanguageSmoother] = {}
# 同语种组当前已启用的校准矩阵。键为排序后的小写语种元组，矩阵按
# ``CalibratedMatrix.languages`` 的顺序存储；会话按请求语种顺序重排后使用。
_calibrations: Dict[Tuple[str, ...], "CalibratedMatrix"] = {}
_lock = threading.Lock()


@dataclass
class CalibratedMatrix:
    group: Tuple[str, ...]
    languages: List[str]
    matrix: np.ndarray
    baseline_score: float
    candidate_score: float
    enabled_at: str
    calibration_count: int


def _group_key(languages: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(x.lower() for x in languages))


def _validate_language_tags(v: List[str]) -> List[str]:
    cleaned = [x.strip() for x in v]
    if any(not x for x in cleaned):
        raise ValueError("language tags must be non-empty")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("language tags must be unique")
    if len(set(x.lower() for x in cleaned)) != len(cleaned):
        raise ValueError("language tags must be unique (case-insensitive)")
    return cleaned


# --------------------------------------------------------------------- 模型
class SessionCreate(BaseModel):
    languages: List[str] = Field(
        ..., min_length=2, max_length=4, description="2~4 个语种标签，顺序即后验向量顺序"
    )
    frame_ms: int = Field(20, ge=1, le=1000, description="单帧时长（毫秒）")
    switch_frames: int = Field(
        5, ge=1, le=1000, description="切换（或首段建立）所需连续帧数"
    )
    posterior_threshold: float = Field(
        0.55, gt=0.0, le=1.0, description="低置信帧阈值：归一化原始后验最大值低于此值的帧沿用上一状态"
    )
    switch_penalty: float = Field(
        2.0, ge=0.0, le=50.0, description="HMM 切换到其它语种的对数惩罚（纳特）；use_calibrated_matrix=true 时仅作回显"
    )
    use_calibrated_matrix: bool = Field(
        False,
        description="true 时使用同语种组当前已启用的校准矩阵（按请求语种顺序重排）；无已启用矩阵则 409",
    )

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, v: List[str]) -> List[str]:
        return _validate_language_tags(v)


class RecordingIn(BaseModel):
    posteriors: List[List[float]] = Field(
        ..., description="形状 (帧数, 语种数) 的逐帧后验，每行非负且和 > 0"
    )
    labels: Optional[List[Optional[str]]] = Field(
        None, description="逐帧人工语种标签；训练可缺标（null 或整体省略），留出须逐帧标全"
    )

    @model_validator(mode="after")
    def _check_shape(self) -> "RecordingIn":
        if not self.posteriors:
            raise ValueError("each recording must contain at least one frame")
        width = len(self.posteriors[0])
        if width == 0:
            raise ValueError("posterior rows must be non-empty")
        if any(len(row) != width for row in self.posteriors):
            raise ValueError("all posterior rows must have the same length")
        return self


class CalibrationCreate(BaseModel):
    languages: List[str] = Field(..., min_length=2, max_length=4)
    train: List[RecordingIn] = Field(..., min_length=1, description="训练录音，至少 1 条，标签可缺")
    held_out: List[RecordingIn] = Field(
        ..., min_length=1, description="留出录音，至少 1 条，须逐帧标全"
    )

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, v: List[str]) -> List[str]:
        return _validate_language_tags(v)


class ChunkSubmit(BaseModel):
    start_seq: int = Field(..., ge=0, description="本块第一帧的全局序号（从 0 开始）")
    posteriors: List[List[float]] = Field(
        ..., description="形状 (帧数, 语种数) 的后验概率"
    )
    final: bool = Field(False, description="置 true 等价于随后调用 /seal")

    @model_validator(mode="after")
    def _check_posteriors(self) -> "ChunkSubmit":
        if not self.posteriors:
            raise ValueError("posteriors must contain at least one frame")
        width = len(self.posteriors[0])
        if width == 0:
            raise ValueError("posteriors rows must be non-empty")
        if any(len(row) != width for row in self.posteriors):
            raise ValueError("all posterior rows must have the same length")
        return self


class SessionOut(BaseModel):
    session_id: str
    languages: List[str]
    frame_ms: int
    switch_frames: int
    posterior_threshold: float
    switch_penalty: float
    matrix_source: str
    transition_matrix: List[List[float]]


class CalibrationOut(BaseModel):
    languages: List[str]
    group: List[str]
    baseline_score: float
    candidate_score: float
    score_delta: float
    enabled: bool
    replaced: bool
    active_source: str
    rounds: int
    switch_penalty_start: float
    train_frames: int
    held_out_frames: int
    total_frames: int
    baseline_matrix: List[List[float]]
    candidate_matrix: List[List[float]]
    active_matrix: List[List[float]]
    calibration_count: int
    enabled_at: Optional[str] = None


class CalibrationStatusOut(BaseModel):
    languages: List[str]
    group: List[str]
    matrix: List[List[float]]
    baseline_score: float
    candidate_score: float
    enabled_at: str
    calibration_count: int


class ChunkOut(BaseModel):
    session_id: str
    start_seq: int
    end_seq: int
    accepted: bool
    duplicate: bool
    sealed: bool
    segments: List[dict]
    open_segment: Optional[dict]
    current_language: Optional[str]
    pending_language: Optional[str]
    pending_frames: int
    next_seq: int


class SealOut(BaseModel):
    session_id: str
    sealed: bool
    segments: List[dict]
    next_seq: int


# --------------------------------------------------------------------- 错误
@app.exception_handler(SessionError)
async def _session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(CalibrationError)
async def _calibration_error_handler(_: Request, exc: CalibrationError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def _get(session_id: str) -> LanguageSmoother:
    with _lock:
        s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


# --------------------------------------------------------------------- 路由
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate) -> SessionOut:
    sid = uuid.uuid4().hex

    matrix: Optional[np.ndarray] = None
    matrix_source = "penalty"
    with _lock:
        entry = _calibrations.get(_group_key(body.languages))
        if body.use_calibrated_matrix:
            if entry is None:
                # 同语种组尚无校准启用记录：请求失败，不创建会话。
                raise CalibrationError(
                    409,
                    "calibrated_matrix_unavailable",
                    "no enabled calibrated matrix for this language group yet",
                )
            # 按本请求的语种顺序重排已启用矩阵。
            matrix = reorder_matrix(entry.matrix, entry.languages, body.languages)
            matrix_source = "calibrated"

        smoother = LanguageSmoother(
            session_id=sid,
            languages=body.languages,
            frame_ms=body.frame_ms,
            switch_frames=body.switch_frames,
            posterior_threshold=body.posterior_threshold,
            switch_penalty=body.switch_penalty,
            transition_matrix=matrix,
            matrix_source=matrix_source,
        )
        _sessions[sid] = smoother
        active = smoother.transition_matrix
    return SessionOut(
        session_id=sid,
        languages=body.languages,
        frame_ms=body.frame_ms,
        switch_frames=body.switch_frames,
        posterior_threshold=body.posterior_threshold,
        switch_penalty=body.switch_penalty,
        matrix_source=matrix_source,
        transition_matrix=active.tolist(),
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    s = _get(session_id)
    with _lock:
        return s.state()


@app.post("/sessions/{session_id}/chunks", response_model=ChunkOut)
def submit_chunk(session_id: str, body: ChunkSubmit) -> dict:
    s = _get(session_id)
    arr = np.asarray(body.posteriors, dtype=np.float64)
    with _lock:
        result: ChunkResult = s.submit_chunk(body.start_seq, arr)
        sealed_now = False
        if body.final and not s.sealed:
            s.seal()
            sealed_now = True
        payload = result.to_dict(sealed=s.sealed)
        if sealed_now:
            # final 触发的封存会把尾段并入最终区间，响应反映封存后的完整列表。
            payload["segments"] = [seg.to_dict() for seg in s.final_segments()]
            payload["open_segment"] = None
    return payload


@app.post("/sessions/{session_id}/seal", response_model=SealOut)
def seal_session(session_id: str) -> SealOut:
    s = _get(session_id)
    with _lock:
        segments = s.seal()
        return SealOut(
            session_id=session_id,
            sealed=True,
            segments=[seg.to_dict() for seg in segments],
            next_seq=s.next_seq,
        )


@app.get("/sessions/{session_id}/segments")
def get_segments(session_id: str) -> dict:
    s = _get(session_id)
    with _lock:
        return {
            "session_id": session_id,
            "sealed": s.sealed,
            "segments": [seg.to_dict() for seg in s.final_segments()],
            "next_seq": s.next_seq,
        }


# ------------------------------------------------------------------ 矩阵校准
def _calibration_payload(entry: CalibratedMatrix) -> dict:
    return {
        "languages": list(entry.languages),
        "group": list(entry.group),
        "matrix": entry.matrix.tolist(),
        "baseline_score": entry.baseline_score,
        "candidate_score": entry.candidate_score,
        "enabled_at": entry.enabled_at,
        "calibration_count": entry.calibration_count,
    }


@app.post("/calibration", response_model=CalibrationOut)
def submit_calibration(body: CalibrationCreate) -> dict:
    languages = body.languages
    n = len(languages)

    # 结构之外的语义校验（概率合法性、标签、总帧数上限）。
    train_records = prepare_recordings(
        languages, [rec.model_dump() for rec in body.train], require_labels=False
    )
    held_out_records = prepare_recordings(
        languages, [rec.model_dump() for rec in body.held_out], require_labels=True
    )
    total_frames = sum(rec[0].shape[0] for rec in train_records + held_out_records)
    if total_frames > MAX_CALIBRATION_FRAMES:
        raise CalibrationError(
            422,
            "frame_budget_exceeded",
            f"total frames {total_frames} exceed limit {MAX_CALIBRATION_FRAMES}",
        )

    group = _group_key(languages)
    # 校准与块提交共用一把锁：校准期间块请求短暂等待、不丢块（收块继续），
    # 且"当时在用的矩阵"在本次校准全程保持稳定。
    with _lock:
        entry = _calibrations.get(group)
        if entry is None:
            # 首次校准：以切换惩罚 2 的矩阵作为对照基线。
            baseline = penalty_matrix(n, CALIBRATION_PENALTY)
            calibration_count = 0
        else:
            # 后续校准：对照同语种组当时已启用的矩阵，按本次语种顺序重排。
            baseline = reorder_matrix(entry.matrix, entry.languages, languages)
            calibration_count = entry.calibration_count

        outcome = calibrate(languages, train_records, held_out_records, baseline)
        replaced = False
        enabled_at: Optional[str] = None
        if outcome.enabled:
            enabled_at = datetime.now(timezone.utc).isoformat()
            _calibrations[group] = CalibratedMatrix(
                group=group,
                languages=list(languages),
                matrix=outcome.active_matrix,
                baseline_score=outcome.baseline_score,
                candidate_score=outcome.candidate_score,
                enabled_at=enabled_at,
                calibration_count=calibration_count + 1,
            )
            replaced = entry is not None
        active_entry = _calibrations.get(group)

    return {
        "languages": list(languages),
        "group": list(group),
        "baseline_score": outcome.baseline_score,
        "candidate_score": outcome.candidate_score,
        "score_delta": outcome.candidate_score - outcome.baseline_score,
        "enabled": outcome.enabled,
        "replaced": replaced,
        "active_source": outcome.active_source,
        "rounds": CALIBRATION_ROUNDS,
        "switch_penalty_start": CALIBRATION_PENALTY,
        "train_frames": outcome.train_frames,
        "held_out_frames": outcome.held_out_frames,
        "total_frames": total_frames,
        "baseline_matrix": outcome.baseline_matrix.tolist(),
        "candidate_matrix": outcome.candidate_matrix.tolist(),
        "active_matrix": (
            active_entry.matrix.tolist()
            if active_entry is not None
            else outcome.baseline_matrix.tolist()
        ),
        "calibration_count": (
            active_entry.calibration_count if active_entry is not None else 0
        ),
        "enabled_at": enabled_at,
    }


@app.get("/calibration", response_model=CalibrationStatusOut)
def get_calibration(languages: List[str] = Query(..., min_length=2, max_length=4)) -> dict:
    try:
        cleaned = _validate_language_tags(languages)
    except ValueError as exc:
        raise CalibrationError(422, "invalid_languages", str(exc)) from exc
    group = _group_key(cleaned)
    with _lock:
        entry = _calibrations.get(group)
        if entry is None:
            raise CalibrationError(
                404,
                "calibration_not_found",
                "no enabled calibrated matrix for this language group",
            )
        # 按查询语种顺序重排返回，便于客户端直接对照后验列。
        matrix = reorder_matrix(entry.matrix, entry.languages, cleaned)
    payload = _calibration_payload(entry)
    payload["languages"] = cleaned
    payload["matrix"] = matrix.tolist()
    return payload
