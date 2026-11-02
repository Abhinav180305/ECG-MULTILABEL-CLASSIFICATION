"""Model / training / inference tests (need torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgmega import config as C                       # noqa: E402
from ecgmega import ontology as O                     # noqa: E402
from ecgmega.builder import build_store                # noqa: E402
from ecgmega.inference import Predictor                # noqa: E402
from ecgmega.model import build_model                  # noqa: E402
from ecgmega.sources import build_source               # noqa: E402
from ecgmega.sources.synthetic import synth_record     # noqa: E402
from ecgmega.train import TrainConfig, masked_bce, train   # noqa: E402


def test_model_shapes():
    m = build_model(O.N_LABELS, "tiny")
    x = torch.randn(3, 12, 800)
    assert m(x).shape == (3, O.N_LABELS)
    assert m.reconstruct(x).shape == x.shape
    assert m(torch.randn(2, 12, 1000)).shape == (2, O.N_LABELS)       # any length works


def test_masked_bce_ignores_unannotated():
    logits = torch.zeros(2, 3)
    y = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    m = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[m > 0], y[m > 0])
    assert torch.isclose(masked_bce(logits, y, m), expected)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    srcs = [build_source("synthetic", "a", options={"n": 240, "seed": 1}),
            build_source("synthetic", "b", options={"n": 60, "seed": 2, "fs": 250})]
    build_store(srcs, d / "store", holdout_sources=["b"])
    cfg = TrainConfig(store_path=str(d / "store"), out_dir=str(d / "run"), model_size="tiny", epochs=6,
                      pretrain_epochs=1, batch_size=32, num_workers=0, lr=3e-3, patience=6)
    res = train(cfg, log=lambda s: None)
    return d, res


def test_training_learns(trained):
    _, res = trained
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
    assert res["val"]["macro_auroc"] is None or res["val"]["macro_auroc"] > 0.5
    assert "external" in res


def test_predictor_roundtrip(trained):
    d, res = trained
    pred = Predictor(res["checkpoint"])
    rng = np.random.default_rng(0)
    sig = synth_record("AFIB", 500, 10, rng)
    out = pred.predict(sig, 500)
    assert set(out["probabilities"]) == set(O.LABELS) and out["n_windows"] == 1
    with pytest.raises(ValueError):
        pred.predict(sig[:, :1000], 500)                                  # < 8 s
    with pytest.raises(ValueError):
        pred.predict(np.zeros_like(sig), 500)                             # flat line
