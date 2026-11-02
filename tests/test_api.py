"""End-to-end API test: build (synthetic) -> train -> predict. Needs torch + fastapi + httpx."""
import importlib
import io
import time

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient              # noqa: E402

from ecgmega.sources.synthetic import synth_record     # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ECG_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("ECG_API_KEY", "secret")
    import ecgmega.config as cfg
    importlib.reload(cfg)
    import ecgmega.api as api
    importlib.reload(api)
    with TestClient(api.app) as c:                      # context manager runs the lifespan
        c.headers.update({"X-API-Key": "secret"})
        yield c


def wait(client, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] not in {"queued", "running"}:
            return j
        time.sleep(0.5)
    raise TimeoutError


def test_full_flow(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/labels", headers={"X-API-Key": "wrong"}).status_code == 401
    assert len(client.get("/labels").json()) == 18

    r = client.post("/datasets/build", json={
        "store_name": "demo", "holdout_sources": ["b"],
        "sources": [{"name": "a", "kind": "synthetic", "options": {"n": 200, "seed": 1}},
                    {"name": "b", "kind": "synthetic", "options": {"n": 40, "seed": 2, "fs": 250}}]})
    assert r.status_code == 202
    assert wait(client, r.json()["job_id"])["status"] == "succeeded"
    assert client.get("/datasets/demo").json()["n"] > 200

    # paths outside the allowed roots are refused
    bad = client.post("/datasets/build", json={"store_name": "x", "sources": [
        {"name": "p", "kind": "ptbxl", "root": "/etc"}]})
    assert bad.status_code == 403

    assert client.post("/predict/json", json={"signal": [[0.0] * 10], "fs": 500}).status_code == 503

    r = client.post("/train", json={"store_name": "demo", "run_name": "r1",
                                    "hyper": {"model_size": "tiny", "epochs": 3, "batch_size": 32,
                                              "num_workers": 0}})
    assert r.status_code == 202
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job.get("error")
    assert client.get("/model").json()["train_fs"] == 100
    assert [x["name"] for x in client.get("/runs").json()] == ["r1"]

    rng = np.random.default_rng(1)
    sig = synth_record("LBBB", 500, 10, rng)
    buf = io.BytesIO()
    np.save(buf, sig)
    r = client.post("/predict", files=[("files", ("ecg.npy", buf.getvalue()))], data={"fs": "500"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["probabilities"]) >= {"NORM", "AFIB"} and "disclaimer" in body

    r = client.post("/predict/json", json={"signal": sig.tolist(), "fs": 500})
    assert r.status_code == 200

    csv = "\n".join(",".join(f"{v:.4f}" for v in row) for row in sig.T)        # headerless samples x 12
    r = client.post("/predict", files=[("files", ("ecg.csv", csv.encode()))], data={"fs": "500"})
    assert r.status_code == 200, r.text

    assert client.post("/predict", files=[("files", ("x.txt", b"hi"))], data={"fs": "500"}).status_code == 415
