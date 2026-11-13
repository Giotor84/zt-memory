"""TransformersHiddenStateEncoder on a tiny randomly-initialised local Qwen3 (no network, no weights download)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from zt_memory import NodeKind, route_to_compute_engine  # noqa: E402
from zt_memory.encoders import TransformersHiddenStateEncoder, build_encoder  # noqa: E402

WORDS = ("def class return import parse json config file path database pool connection acquire release "
         "render template html request route handler slug title camel snake strict value error missing").split()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-qwen3")
    vocab = {"[PAD]": 0, "[UNK]": 1, "<eos>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", eos_token="<eos>")
    fast.save_pretrained(d)
    cfg = transformers.Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=256)
    torch.manual_seed(0)
    transformers.Qwen3Model(cfg).save_pretrained(d)
    return str(d)


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_vectors_do_not_depend_on_batch_composition(tiny_model, pooling):
    enc = TransformersHiddenStateEncoder(tiny_model, pooling=pooling, batch_size=4)
    short = "parse json config"
    alone = enc.encode([short])
    padded = enc.encode([short, "database pool connection acquire release render template html request route"])
    assert enc.dim == 32 and alone.shape == (1, 32)
    assert torch.allclose(alone[0], padded[0], atol=1e-5)


def test_query_prefix_and_fingerprint(tiny_model):
    plain = build_encoder(f"hf:{tiny_model}", pooling="last")
    prefixed = build_encoder(f"hf:{tiny_model}", pooling="last", query_prefix="route handler ")
    q = "parse json config"
    assert torch.allclose(plain.encode([q], is_query=True), plain.encode([q]))
    assert not torch.allclose(prefixed.encode([q], is_query=True), prefixed.encode([q]))
    assert plain.fingerprint != prefixed.fingerprint  # changing the instruction invalidates the store


def test_manager_end_to_end_with_hf_encoder(repo, make_manager, tiny_model):
    enc = build_encoder(f"hf:{tiny_model}", pooling="last", max_length=128)
    mgr = make_manager(encoder=enc, rank=16, calibration_samples=96)
    first = mgr.sync()
    assert first.ingest.chunks_added > 0
    assert mgr.sync().ingest.surprise_events == 0  # deterministic vectors: a no-op resync stays silent
    ctx = route_to_compute_engine("parse json config file", mgr)
    assert any(b.kind == NodeKind.CHUNK for b in ctx.blocks)


def test_length_bucketing_preserves_order_and_values(tiny_model):
    enc = TransformersHiddenStateEncoder(tiny_model, pooling="last", batch_size=2)
    texts = ["parse json config file path database pool connection", "slug", "render template html request", "camel snake"]
    batched = enc.encode(texts)
    singles = torch.cat([enc.encode([t]) for t in texts])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_release_frees_the_model(tiny_model):
    enc = TransformersHiddenStateEncoder(tiny_model, pooling="last")
    assert enc.device == "cpu"
    enc.encode(["parse json"])
    enc.release()
    with pytest.raises(RuntimeError):
        enc.encode(["parse json"])
