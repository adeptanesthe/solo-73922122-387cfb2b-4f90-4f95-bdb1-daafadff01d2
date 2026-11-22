"""校准调用示例：提交训练/留出录音 → EM 校准 → 启用判定 → 新会话选用矩阵。

先启动服务（二选一）：
    docker compose up --build
    # 或本地：
    uvicorn app.main:app --port 8000

再运行：
    python examples/calibration_demo.py [base_url]

只依赖标准库。录音用一个强"粘性"三语种 HMM 合成（观测故意较软），
因此 5 轮 EM 得到的候选矩阵通常能在留出集上跑赢切换惩罚 2 的基线并被启用。
"""

from __future__ import annotations

import json
import random
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LANGUAGES = ["zh", "en", "ja"]
N = len(LANGUAGES)


def post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def simulate(rng: random.Random, length: int, stick: float = 0.95) -> dict:
    # 真实语种路径：自转移 0.95，其余均分。
    states = [rng.randrange(N)]
    for _ in range(1, length):
        if rng.random() < stick:
            states.append(states[-1])
        else:
            states.append(rng.choice([j for j in range(N) if j != states[-1]]))
    # 软观测：真实语种 0.45，其余 0.275，再加小抖动并归一化。
    rows = []
    for s in states:
        row = [(0.45 if j == s else 0.275) + rng.uniform(-0.02, 0.02) for j in range(N)]
        row = [max(x, 1e-3) for x in row]
        total = sum(row)
        rows.append([round(x / total, 6) for x in row])
    return {
        "posteriors": rows,
        "labels": [LANGUAGES[s] for s in states],
    }


def show(label: str, status: int, body: dict, keys: tuple[str, ...]) -> None:
    print(f"\n[{label}] HTTP {status}")
    print(json.dumps({k: body[k] for k in keys if k in body}, ensure_ascii=False, indent=2))


def main() -> None:
    rng = random.Random(20261122)

    # 6 条训练（共 480 帧）+ 3 条留出（240 帧），合计 720 帧 <= 1000。
    train = [simulate(rng, 80) for _ in range(6)]
    held_out = [simulate(rng, 80) for _ in range(3)]

    # 演示训练可缺标：第 2 条整体缺标，第 3 条前 20 帧之外缺标。
    train[1]["labels"] = None
    train[2]["labels"] = train[2]["labels"][:20] + [None] * 60

    payload = {"languages": LANGUAGES, "train": train, "held_out": held_out}
    status, body = post("/calibration", payload)
    show(
        "calibration #1 (baseline = penalty-2 matrix)",
        status,
        body,
        (
            "languages",
            "baseline_score",
            "candidate_score",
            "score_delta",
            "enabled",
            "replaced",
            "active_source",
            "rounds",
            "switch_penalty_start",
            "train_frames",
            "held_out_frames",
            "total_frames",
            "calibration_count",
            "baseline_matrix",
            "candidate_matrix",
        ),
    )

    # 第二次校准：对照基线变为同组当时已启用的矩阵。
    status2, body2 = post("/calibration", payload)
    show(
        "calibration #2 (baseline = currently enabled matrix)",
        status2,
        body2,
        ("baseline_score", "candidate_score", "enabled", "replaced", "calibration_count"),
    )

    # 查询同语种组当前启用矩阵（换一个语种顺序，服务端按顺序重排）。
    status3, status_body = get(
        "/calibration?languages=ja&languages=zh&languages=en"
    )
    show(
        "GET /calibration (reordered to ja, zh, en)",
        status3,
        status_body,
        ("languages", "group", "matrix", "enabled_at", "calibration_count"),
    )

    # 新会话选用已启用矩阵；再开一个默认（惩罚矩阵）会话作对照。
    status4, calibrated_session = post(
        "/sessions",
        {
            "languages": LANGUAGES,
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.4,
            "use_calibrated_matrix": True,
        },
    )
    show(
        "session with calibrated matrix",
        status4,
        calibrated_session,
        ("session_id", "languages", "matrix_source", "transition_matrix"),
    )

    status5, default_session = post(
        "/sessions",
        {"languages": LANGUAGES, "frame_ms": 20, "switch_frames": 3},
    )
    show(
        "session with default penalty matrix",
        status5,
        default_session,
        ("session_id", "matrix_source", "transition_matrix"),
    )

    # 校准矩阵会话照常收块。
    sid = calibrated_session["session_id"]
    chunk = {"start_seq": 0, "posteriors": [rec["posteriors"][0] for rec in held_out]}
    status6, chunk_body = post(f"/sessions/{sid}/chunks", chunk)
    show(
        "submit chunk to calibrated session",
        status6,
        chunk_body,
        ("accepted", "duplicate", "current_language", "next_seq"),
    )

    # 未校准过的语种组请求校准矩阵 -> 409。
    status7, err = post(
        "/sessions",
        {"languages": ["zh", "fr"], "use_calibrated_matrix": True},
    )
    show("no matrix for group (zh, fr)", status7, err, ("error",))


if __name__ == "__main__":
    main()
