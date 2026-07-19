"""Phase 4 validation: TurboQuant KV cache inside a real transformer.

Sanity ladder (each rung isolates one failure mode):
1. b=8 ~= fp32 baseline logits -> cache plumbing correct independent of
   quantization strength.
2. packed == simulate logits -> bit-packing and deferred dequantization are
   faithful to the stored codes.
3. KL monotone in bits, and b=2 clearly worse -> the knob really does
   something (guards against silently bypassing quantization).
4. Split-point invariance -> tokens are quantized once at arrival, so
   prefill/decode chunking cannot change results.
5. Key centering regression -> without the frozen-mu centering, b=4 keys
   destroy this model (KL ~6.5); with it, KL ~0.004 (see kv_cache.py
   docstring for the mechanism). This pins the finding.

Uses Qwen2.5-0.5B-Instruct (ungated, head_dim=64 like Llama-3.2-1B) on a
512-token WikiText prompt. Skipped automatically if model/data unavailable.
"""

import pytest
import torch

from turboquant.kv_cache import TurboQuantCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def setup():
    transformers = pytest.importorskip("transformers")
    try:
        from datasets import load_dataset
        tok = transformers.AutoTokenizer.from_pretrained(MODEL)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32).eval()
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test")
    except Exception as e:  # offline / not cached
        pytest.skip(f"model or data unavailable: {e}")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        base = model(ids).logits[0, -1].float()
    return model, ids, base


@torch.no_grad()
def _last_logits(model, ids, cache):
    return model(ids, past_key_values=cache, use_cache=True).logits[0, -1].float()


def _kl(base, other):
    p = torch.softmax(base, -1)
    return (p * (torch.log_softmax(base, -1)
                 - torch.log_softmax(other, -1))).sum().item()


class TestKVCache:
    def test_b8_matches_baseline(self, setup):
        model, ids, base = setup
        q8 = _last_logits(model, ids, TurboQuantCache(bits=8))
        assert _kl(base, q8) < 1e-3
        assert base.argmax() == q8.argmax()

    def test_b4_quality(self, setup):
        model, ids, base = setup
        q4 = _last_logits(model, ids, TurboQuantCache(bits=4))
        assert _kl(base, q4) < 0.05
        assert base.argmax() == q4.argmax()

    def test_packed_equals_simulate(self, setup):
        model, ids, _ = setup
        sim = _last_logits(model, ids, TurboQuantCache(bits=4, packed=False))
        pak = _last_logits(model, ids, TurboQuantCache(bits=4, packed=True))
        assert torch.allclose(sim, pak, atol=2e-3), \
            (sim - pak).abs().max().item()

    def test_degradation_is_monotone(self, setup):
        model, ids, base = setup
        kls = {b: _kl(base, _last_logits(model, ids, TurboQuantCache(bits=b)))
               for b in (8, 4, 2)}
        assert kls[8] < kls[4] < kls[2], kls
        assert kls[2] > 20 * kls[8]

    def test_split_point_invariance(self, setup):
        model, ids, _ = setup
        one = _last_logits(model, ids, TurboQuantCache(bits=4))
        cache = TurboQuantCache(bits=4)
        with torch.no_grad():
            model(ids[:, :500], past_key_values=cache, use_cache=True)
            out = model(ids[:, 500:], past_key_values=cache, use_cache=True)
        assert torch.allclose(one, out.logits[0, -1].float(), atol=2e-3)

    def test_key_centering_is_load_bearing(self, setup):
        """The headline Phase 4 finding, pinned as a regression test."""
        model, ids, base = setup
        kl_on = _kl(base, _last_logits(model, ids, TurboQuantCache(bits=4)))
        kl_off = _kl(base, _last_logits(
            model, ids, TurboQuantCache(bits=4, center_keys=False, warmup=0)))
        assert kl_off > 50 * kl_on, (kl_off, kl_on)
        assert kl_on < 0.05

    def test_memory_accounting(self, setup):
        model, ids, _ = setup
        cfg = model.config
        t = ids.shape[1]
        h_kv = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        n_l = cfg.num_hidden_layers

        # paper-faithful config: everything quantized, no warmup, no mu
        cache = TurboQuantCache(bits=4, packed=True, center_keys=False, warmup=0)
        _ = _last_logits(model, ids, cache)
        expected = n_l * 2 * h_kv * t * (head_dim * 4 // 8 + 2)
        assert cache.quantized_bytes == expected

        # default config adds the fp warmup window and one mu vector per layer
        w = 32
        cache = TurboQuantCache(bits=4, packed=True)
        _ = _last_logits(model, ids, cache)
        expected = (n_l * 2 * h_kv * (t - w) * (head_dim * 4 // 8 + 2)
                    + n_l * 2 * h_kv * w * head_dim * 4      # warmup, fp32 ids
                    + n_l * h_kv * head_dim * 4)             # frozen mu, fp32
        assert cache.quantized_bytes == expected
