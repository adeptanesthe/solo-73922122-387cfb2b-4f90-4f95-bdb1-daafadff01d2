"""校准 API 与核心算法自测：EM 启用/不替换、标签约束、矩阵重排、
新会话选矩阵、已开会话保留旧矩阵、并发收块不中断、校验与帧预算。"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.calibration import (
    MAX_CALIBRATION_FRAMES,
    penalty_matrix,
    reorder_matrix,
)
from app.main import _calibrations, _sessions, app
from app.smoother import LanguageSmoother

LANGS = ["zh", "en", "ja"]


@pytest.fixture(autouse=True)
def _clear_state():
    _sessions.clear()
    _calibrations.clear()
    yield
    _sessions.clear()
    _calibrations.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------- 数据合成
def simulate(rng: np.random.Generator, n: int, length: int, stick: float = 0.95):
    """从"粘性"真实转移矩阵合成一条录音：返回后验观测与真实语种下标。

    观测刻意做得较"软"（真实语种仅 0.45），这样转移矩阵对真实路径的
    条件概率有实质影响，校准矩阵才能在留出集上稳定拉开与惩罚 2 基线的差距。
    """
    a = np.full((n, n), (1.0 - stick) / (n - 1))
    np.fill_diagonal(a, stick)
    states = np.empty(length, dtype=int)
    states[0] = rng.integers(n)
    for t in range(1, length):
        states[t] = rng.choice(n, p=a[states[t - 1]])
    hi = 0.45
    post = np.full((length, n), (1.0 - hi) / (n - 1))
    post[np.arange(length), states] = hi
    post += rng.normal(0.0, 0.02, size=post.shape)
    post = np.clip(post, 1e-3, 1.0)
    post /= post.sum(axis=1, keepdims=True)
    return post, states


def make_payload(rng: np.random.Generator, n_train: int, n_held: int, length: int = 60):
    train, held = [], []
    for _ in range(n_train):
        post, states = simulate(rng, len(LANGS), length)
        train.append(
            {
                "posteriors": post.round(6).tolist(),
                "labels": [LANGS[s] for s in states],
            }
        )
    for _ in range(n_held):
        post, states = simulate(rng, len(LANGS), length)
        held.append(
            {
                "posteriors": post.round(6).tolist(),
                "labels": [LANGS[s] for s in states],
            }
        )
    return {"languages": LANGS, "train": train, "held_out": held}


def _labels(rec: dict, keep: int) -> list:
    return rec["labels"][:keep] + [None] * (len(rec["labels"]) - keep)


# ------------------------------------------------------------ 核心校准流程
def test_calibration_enables_candidate_and_reuses_baseline_next_time(client: TestClient):
    rng = np.random.default_rng(11)
    payload = make_payload(rng, n_train=6, n_held=3, length=80)
    r = client.post("/calibration", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    # 合成数据来自强粘性 HMM，5 轮 EM 后候选在留出集上应严格优于惩罚 2 基线。
    assert body["enabled"] is True
    assert body["replaced"] is False  # 首次校准，无旧矩阵可替换
    assert body["candidate_score"] > body["baseline_score"]
    assert body["score_delta"] == pytest.approx(
        body["candidate_score"] - body["baseline_score"]
    )
    assert body["rounds"] == 5
    assert body["switch_penalty_start"] == 2.0
    assert body["total_frames"] == body["train_frames"] + body["held_out_frames"]
    assert body["total_frames"] == 9 * 80
    assert body["calibration_count"] == 1
    assert body["active_source"] == "candidate"

    # 矩阵合法性与字段一致性。
    cand = np.array(body["candidate_matrix"])
    assert cand.shape == (3, 3)
    assert np.allclose(cand.sum(axis=1), 1.0)
    assert np.all(cand > 0.0)
    assert np.allclose(np.diag(cand), np.diag(np.array(body["active_matrix"])))
    base = np.array(body["baseline_matrix"])
    assert np.allclose(base, penalty_matrix(3, 2.0))

    # 同语种组已可查询，且第二次校准以当前启用矩阵为对照（replaced 语义）。
    g = client.get("/calibration", params=[("languages", "zh"), ("languages", "en"), ("languages", "ja")])
    assert g.status_code == 200
    assert np.allclose(np.array(g.json()["matrix"]), cand)

    r2 = client.post("/calibration", json=payload)
    assert r2.status_code == 200
    body2 = r2.json()
    # 第二次的对照基线就是上次启用的候选矩阵。
    assert np.allclose(np.array(body2["baseline_matrix"]), cand)
    assert body2["calibration_count"] == (2 if body2["enabled"] else 1)
    if body2["enabled"]:
        assert body2["replaced"] is True


def test_candidate_not_replaced_when_score_not_higher(client: TestClient):
    # 确定性构造：留出录音全部只有单帧。单帧序列没有任何转移，其条件对数
    # 概率 log P(z|x) 只取决于均匀先验与发射似然，与转移矩阵无关，因此
    # 候选与基线分数必然相等——"严格更高才替换"稳定走不启用分支。
    rng = np.random.default_rng(3)
    rows = rng.dirichlet(np.ones(3), size=40)
    rec_train = {"posteriors": rows.round(6).tolist(), "labels": [None] * 40}
    held_single = [
        {
            "posteriors": rng.dirichlet(np.ones(3), size=1).round(6).tolist(),
            "labels": [LANGS[int(rng.integers(3))]],
        }
        for _ in range(3)
    ]
    body_in = {"languages": LANGS, "train": [rec_train], "held_out": held_single}
    r = client.post("/calibration", json=body_in)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["candidate_score"] == pytest.approx(out["baseline_score"])
    assert out["enabled"] is False
    assert out["replaced"] is False
    assert out["active_source"] == "baseline"
    assert np.allclose(np.array(out["active_matrix"]), out["baseline_matrix"])
    # 未启用 => 同语种组查询仍 404，且不留任何启用记录。
    g = client.get(
        "/calibration", params=[("languages", "zh"), ("languages", "en"), ("languages", "ja")]
    )
    assert g.status_code == 404
    assert g.json()["error"]["code"] == "calibration_not_found"
    assert _calibrations == {}


def test_candidate_losing_strictly_keeps_baseline(client: TestClient):
    # 确定性反向构造：训练真实路径只在 zh/en 两状态间来回（ja 从未出现），
    # 5 轮 EM 后候选里 ja 的入边只剩伪计数；留出真实路径却只在 en↔ja 间
    # 来回切换，候选必然显著差于对称的惩罚 2 基线。
    t_states = np.tile(np.array([0, 1]), 60)  # 120 帧，只有 zh/en
    t_post = np.full((120, 3), 0.1)
    t_post[np.arange(120), t_states] = 0.8
    train = [{"posteriors": t_post.round(6).tolist(), "labels": [LANGS[s] for s in t_states]}]
    h_states = np.tile(np.array([1, 2]), 30)  # 60 帧，只在 en/ja 间切
    h_post = np.full((60, 3), 0.1)
    h_post[np.arange(60), h_states] = 0.8
    held = [{"posteriors": h_post.round(6).tolist(), "labels": [LANGS[s] for s in h_states]}]
    r = client.post(
        "/calibration", json={"languages": LANGS, "train": train, "held_out": held}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["candidate_score"] < out["baseline_score"]
    assert out["enabled"] is False
    assert _calibrations == {}


def test_first_baseline_is_penalty_two_matrix(client: TestClient):
    rng = np.random.default_rng(5)
    r = client.post("/calibration", json=make_payload(rng, 2, 2, length=30))
    assert r.status_code == 200
    base = np.array(r.json()["baseline_matrix"])
    assert np.allclose(base, penalty_matrix(3, 2.0))


# ------------------------------------------------------------ 标签与校验
def test_train_may_have_missing_labels(client: TestClient):
    rng = np.random.default_rng(8)
    payload = make_payload(rng, n_train=3, n_held=2, length=50)
    # 训练录音 1 整体缺标；录音 2 逐帧缺标；留出全部标全。
    payload["train"][1]["labels"] = None
    payload["train"][2] = {
        "posteriors": payload["train"][2]["posteriors"],
        "labels": _labels(payload["train"][2], keep=10),
    }
    r = client.post("/calibration", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["train_frames"] == 150


def test_held_out_must_be_fully_labeled(client: TestClient):
    rng = np.random.default_rng(9)
    payload = make_payload(rng, 2, 2, length=40)
    payload["held_out"][0]["labels"][5] = None
    r = client.post("/calibration", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "heldout_labels_required"

    payload2 = make_payload(rng, 2, 1, length=20)
    payload2["held_out"][0]["labels"] = None
    r2 = client.post("/calibration", json=payload2)
    assert r2.status_code == 422
    assert r2.json()["error"]["code"] == "heldout_labels_required"


def test_frame_budget_and_validation_errors(client: TestClient):
    rng = np.random.default_rng(4)
    over = make_payload(rng, n_train=10, n_held=10, length=60)  # 1200 帧
    r = client.post("/calibration", json=over)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "frame_budget_exceeded"

    base = make_payload(rng, 1, 1, length=10)
    # 维度不匹配。
    bad_width = {
        "languages": LANGS,
        "train": [{"posteriors": [[0.5, 0.5]] * 5}],
        "held_out": base["held_out"],
    }
    rw = client.post("/calibration", json=bad_width)
    assert rw.status_code == 422
    assert rw.json()["error"]["code"] == "dimension_mismatch"

    # 负概率 / 零和 / 未知标签。
    neg = {
        "languages": LANGS,
        "train": [{"posteriors": [[1.1, -0.1, 0.0]] * 4, "labels": ["zh"] * 4}],
        "held_out": base["held_out"],
    }
    assert client.post("/calibration", json=neg).json()["error"]["code"] == "negative_probability"
    zero = {
        "languages": LANGS,
        "train": [{"posteriors": [[0.0, 0.0, 0.0]] * 4, "labels": ["zh"] * 4}],
        "held_out": base["held_out"],
    }
    assert client.post("/calibration", json=zero).json()["error"]["code"] == "invalid_distribution"
    unknown = {
        "languages": LANGS,
        "train": base["train"],
        "held_out": [
            {
                "posteriors": base["held_out"][0]["posteriors"],
                "labels": ["xx"] * 10,
            }
        ],
    }
    assert client.post("/calibration", json=unknown).json()["error"]["code"] == "unknown_label"
    # 失败请求不产生任何启用矩阵。
    assert _calibrations == {}


def test_pydantic_structure_errors(client: TestClient):
    # 语种数量越界、缺训练、空录音。
    r1 = client.post(
        "/calibration",
        json={"languages": ["a"], "train": [{"posteriors": [[0.5, 0.5]]}],
              "held_out": [{"posteriors": [[0.5, 0.5]], "labels": ["a"]}]},
    )
    assert r1.status_code == 422
    r2 = client.post(
        "/calibration",
        json={"languages": ["a", "b"], "train": [], "held_out": []},
    )
    assert r2.status_code == 422
    r3 = client.post(
        "/calibration",
        json={
            "languages": ["a", "b"],
            "train": [{"posteriors": [[0.5, 0.5]]}],
            "held_out": [{"posteriors": [], "labels": []}],
        },
    )
    assert r3.status_code == 422


def test_max_1000_frames_boundary_accepted(client: TestClient):
    rng = np.random.default_rng(2)
    payload = make_payload(rng, n_train=5, n_held=5, length=100)  # 正好 1000 帧
    assert sum(len(x["posteriors"]) for x in payload["train"] + payload["held_out"]) == MAX_CALIBRATION_FRAMES
    r = client.post("/calibration", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["total_frames"] == 1000


# ------------------------------------------------------------ 会话选矩阵
def test_new_session_can_use_enabled_matrix_with_reordering(client: TestClient):
    rng = np.random.default_rng(13)
    r = client.post("/calibration", json=make_payload(rng, 5, 3, length=70))
    assert r.status_code == 200 and r.json()["enabled"]
    cand = np.array(r.json()["candidate_matrix"])

    # 默认会话仍是惩罚矩阵。
    d = client.post("/sessions", json={"languages": LANGS})
    assert d.status_code == 201
    assert d.json()["matrix_source"] == "penalty"
    assert np.allclose(np.array(d.json()["transition_matrix"]), penalty_matrix(3, 2.0))

    # 选用校准矩阵：同顺序。
    c = client.post(
        "/sessions",
        json={"languages": LANGS, "use_calibrated_matrix": True},
    )
    assert c.status_code == 201, c.text
    assert c.json()["matrix_source"] == "calibrated"
    assert np.allclose(np.array(c.json()["transition_matrix"]), cand)

    # 不同语种顺序：矩阵须按请求顺序重排。
    rev = ["ja", "zh", "en"]
    c2 = client.post(
        "/sessions",
        json={"languages": rev, "use_calibrated_matrix": True},
    )
    assert c2.status_code == 201
    expect = reorder_matrix(cand, LANGS, rev)
    assert np.allclose(np.array(c2.json()["transition_matrix"]), expect)

    # GET 状态里也带矩阵与来源。
    sid = c2.json()["session_id"]
    st = client.get(f"/sessions/{sid}").json()
    assert st["matrix_source"] == "calibrated"
    assert np.allclose(np.array(st["transition_matrix"]), expect)

    # GET /calibration 用不同顺序查询，返回重排后的矩阵。
    g = client.get(
        "/calibration", params=[("languages", "ja"), ("languages", "zh"), ("languages", "en")]
    )
    assert g.status_code == 200
    assert g.json()["languages"] == rev
    assert np.allclose(np.array(g.json()["matrix"]), expect)


def test_use_calibrated_without_enabled_matrix_is_409(client: TestClient):
    r = client.post(
        "/sessions",
        json={"languages": LANGS, "use_calibrated_matrix": True},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "calibrated_matrix_unavailable"
    assert _sessions == {}  # 失败不创建会话。


def test_existing_session_keeps_its_matrix_after_recalibration(client: TestClient):
    rng = np.random.default_rng(17)
    r = client.post("/calibration", json=make_payload(rng, 7, 3, length=80))
    assert r.status_code == 200 and r.json()["enabled"]
    cand1 = np.array(r.json()["candidate_matrix"])

    opened = client.post(
        "/sessions",
        json={"languages": LANGS, "use_calibrated_matrix": True},
    )
    sid = opened.json()["session_id"]
    m_before = np.array(opened.json()["transition_matrix"])
    assert np.allclose(m_before, cand1)

    # 再校准（换数据让候选发生变化并再次启用）。
    rng2 = np.random.default_rng(99)
    r2 = client.post("/calibration", json=make_payload(rng2, 7, 3, length=80))
    assert r2.status_code == 200
    assert r2.json()["enabled"]
    cand2 = np.array(r2.json()["candidate_matrix"])
    assert not np.allclose(cand2, cand1)

    # 已开会话矩阵不变；新会话拿到当时启用的新矩阵。
    st = client.get(f"/sessions/{sid}").json()
    assert np.allclose(np.array(st["transition_matrix"]), cand1)
    fresh = client.post(
        "/sessions",
        json={"languages": LANGS, "use_calibrated_matrix": True},
    )
    assert np.allclose(
        np.array(fresh.json()["transition_matrix"]),
        np.array(_calibrations[tuple(sorted(x.lower() for x in LANGS))].matrix),
    )


def test_chunks_keep_being_accepted_during_calibration(client: TestClient):
    # 先开一个普通会话并提交部分帧。
    s = client.post(
        "/sessions",
        json={"languages": LANGS, "switch_frames": 2, "posterior_threshold": 0.5},
    )
    sid = s.json()["session_id"]
    strong = [[0.9, 0.05, 0.05] for _ in range(4)]
    assert client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": strong}
    ).status_code == 200

    rng = np.random.default_rng(21)
    payload = make_payload(rng, 4, 2, length=50)
    done = threading.Event()

    def calibrate():
        rr = client.post("/calibration", json=payload)
        done.result = rr  # type: ignore[attr-defined]
        done.set()

    t = threading.Thread(target=calibrate)
    t.start()
    # 校准进行中继续收块：全部成功且序号连续。
    for start in (4, 5, 6):
        rr = client.post(
            f"/sessions/{sid}/chunks",
            json={"start_seq": start, "posteriors": [[0.05, 0.9, 0.05]]},
        )
        assert rr.status_code == 200
    t.join(timeout=10)
    assert done.is_set()
    assert done.result.status_code == 200  # type: ignore[attr-defined]
    # 会话推进到 7 帧，且仍是建会话时的惩罚矩阵。
    st = client.get(f"/sessions/{sid}").json()
    assert st["next_seq"] == 7
    assert st["matrix_source"] == "penalty"


# ------------------------------------------------------------ 核心层直测
def test_smoother_custom_matrix_matches_penalty_construction():
    a = penalty_matrix(2, 2.0)
    sm = LanguageSmoother(
        "x", ["zh", "en"], 20, 2, 0.55, transition_matrix=a, matrix_source="calibrated"
    )
    default = LanguageSmoother("y", ["zh", "en"], 20, 2, 0.55, switch_penalty=2.0)
    assert np.allclose(sm.transition_matrix, default.transition_matrix)
    assert sm.matrix_source == "calibrated"


def test_smoother_rejects_bad_custom_matrix():
    with pytest.raises(Exception) as e:
        LanguageSmoother("x", ["zh", "en"], 20, 2, 0.55, transition_matrix=np.ones((2, 3)))
    assert e.value.code == "dimension_mismatch"
    with pytest.raises(Exception) as e2:
        LanguageSmoother(
            "x",
            ["zh", "en"],
            20,
            2,
            0.55,
            transition_matrix=np.array([[0.6, 0.4], [0.7, 0.7]]),  # 行和 1.4 ≠ 1
        )
    assert e2.value.code == "invalid_matrix"
    with pytest.raises(Exception) as e3:
        LanguageSmoother(
            "x",
            ["zh", "en"],
            20,
            2,
            0.55,
            transition_matrix=np.array([[1.0, 0.0], [0.2, 0.8]]),  # 含零
        )
    assert e3.value.code == "invalid_matrix"


def test_two_language_calibration_and_404_for_other_group(client: TestClient):
    rng = np.random.default_rng(23)
    n = 2
    langs = ["zh", "en"]
    train, held = [], []
    for _ in range(4):
        post, states = simulate(rng, n, 50)
        train.append({"posteriors": post.round(6).tolist(), "labels": [langs[s] for s in states]})
    for _ in range(2):
        post, states = simulate(rng, n, 50)
        held.append({"posteriors": post.round(6).tolist(), "labels": [langs[s] for s in states]})
    r = client.post("/calibration", json={"languages": langs, "train": train, "held_out": held})
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True
    # 另一语种组无矩阵。
    g = client.get("/calibration", params=[("languages", "zh"), ("languages", "ja")])
    assert g.status_code == 404
    assert g.json()["error"]["code"] == "calibration_not_found"
    # 大小写不敏感的同组匹配。
    g2 = client.get("/calibration", params=[("languages", "EN"), ("languages", "ZH")])
    assert g2.status_code == 200
    assert g2.json()["languages"] == ["EN", "ZH"]
