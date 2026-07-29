"""TurboQuant KV cache for HF transformers (>= v5 Cache/CacheLayer API).

Each attention layer stores K and V as TurboQuant-mse codes: per (token, head)
the post-RoPE key vector (and value vector) of length head_dim is rotated by a
shared per-layer Haar matrix and quantized to `bits` bits per channel, plus an
fp16 norm. Quantization is *online* in the paper's sense: every token is
quantized exactly once, immutably, the moment it enters the cache -- during
prefill and during generation alike. No calibration set, no residual
re-quantization window.

Deliberate divergence from transformers' built-in `QuantizedLayer` (KIVI
style): that implementation keeps a fp16 residual buffer and, on flush,
re-quantizes the *dequantized history* -- repeated lossy re-encoding with
drifting group scales. TurboQuant needs neither: its codebook is fixed and
data-oblivious, so tokens are encoded once and never touched again.

Key centering (`center_keys`, on by default). Transformer keys are far from
isotropic: a large mean vector is shared across tokens (in Qwen2.5-0.5B layer
0, ~98% of the key norm is the shared mean -- driven by K-projection biases /
massive activations). TurboQuant's guarantee is *relative* MSE per vector, so
on raw keys the bit budget is spent re-encoding that constant, and the
guaranteed ~0.9% error at b=4 swamps the informative component: measured
next-token KL 6.5 vs baseline on Qwen2.5-0.5B. The fix implemented here stays
fully online: the first `warmup` tokens are stored fp16 (doubling as
attention-sink protection); their mean is frozen as mu once, and every later
key is quantized as (k - mu), with mu added back at dequantization. Per-vector
worst-case guarantees apply verbatim to the centered vectors. Measured KL at
b=4 drops ~1500x to 0.004. The paper's own KV recipe addresses the same
pathology differently -- its Table 1 configurations always split channels
into higher-precision outlier sets and never run uniform no-split
quantization; the outlier channels *are* where the shared constant lives.

Asymmetric bit-widths (`bits_k`/`bits_v`, override `bits` when set). Measured
on Qwen2.5-0.5B (WikiText-2 perplexity, 10x2048 tokens): K4V8 -- same total
budget as K8V4, precision on the wrong tensor -- reproduces the uniform-b=4
blowup (ppl 25.68 vs fp16's 13.30), while K8V4 (6.125 effective bits/coord)
is within 0.7% of baseline. Keys need precision; values are nearly free
(V-only b=4 next-token KL 2.6e-4). K8V4 is the config to reach for at small
model scale; uniform low-bit K+V is not paper-faithful without an outlier
split (PLAN.md, deferred).

Two storage modes:
- packed=True: true uint8 bit-packed codes + fp16 norms live on the device;
  every attention call dequantizes the history. This is the honest
  memory-measurement mode (bytes on device match the analytic formula).
- packed=False ("simulate"): codes are decoded once at update time and the
  dequantized fp16 tensors are stored via the parent DynamicLayer. Numerically
  identical logits (verified by test), much faster on the Python level --
  used for perplexity/needle sweeps.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer

from turboquant.outlier import OutlierSplitQuantizer, select_outlier_channels
from turboquant.quantizer import QuantizedBatch, TurboQuantMSE


class TurboQuantLayer(DynamicLayer):
    """One attention layer's K/V store, TurboQuant-encoded."""

    def __init__(
        self,
        bits: int = 4,
        seed: int = 0,
        warmup: int = 32,
        center_keys: bool = True,
        packed: bool = False,
        codebook: str = "sphere",
        bits_k: int | None = None,
        bits_v: int | None = None,
        outlier_channels: int = 0,
        bits_k_outlier: int | None = None,
    ) -> None:
        super().__init__()
        self.bits_k = bits if bits_k is None else bits_k
        self.bits_v = bits if bits_v is None else bits_v
        self.seed = seed
        self.warmup = warmup
        self.center_keys = center_keys
        self.packed = packed
        self.codebook_kind = codebook
        # Paper Section 4.3 mixed precision (keys only -- values are nearly
        # free to quantize, Phase 4): `outlier_channels` key channels get
        # `bits_k_outlier` bits (default bits_k + 1, the paper's printed 3&2
        # pattern; pass bits_k + 2 for the label-faithful reading -- see the
        # arithmetic discrepancy note in outlier.py), the rest get bits_k.
        # Selection is frozen from the warmup window on *centered* keys.
        self.outlier_channels = outlier_channels
        self.bits_k_outlier = (
            min(bits_k_outlier if bits_k_outlier is not None
                else self.bits_k + 1, 8))
        self._tq_k: TurboQuantMSE | None = None
        self._tq_v: TurboQuantMSE | None = None
        self._split_k: OutlierSplitQuantizer | None = None
        self._k: QuantizedBatch | None = None  # packed mode storage
        self._v: QuantizedBatch | None = None
        self._warm_k: torch.Tensor | None = None  # first `warmup` tokens, fp
        self._warm_v: torch.Tensor | None = None
        self._mu: torch.Tensor | None = None  # frozen key mean (B, H, 1, D)
        self._stats_frozen = False
        self._len = 0

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        head_dim = key_states.shape[-1]
        # fp32 quantizer math regardless of model dtype (bf16/fp16 rotations
        # would pollute the distortion); K and V share the rotation -- they
        # are separate vector streams, and Haar guarantees are per-stream.
        self._tq_k = TurboQuantMSE(
            head_dim, self.bits_k, seed=self.seed,
            codebook=self.codebook_kind, device=key_states.device,
        )
        self._tq_v = TurboQuantMSE(
            head_dim, self.bits_v, seed=self.seed + 500_000,
            codebook=self.codebook_kind, device=key_states.device,
        )

    def _freeze_key_stats(self, fallback_keys: torch.Tensor) -> None:
        """Fix the key-centering vector and (if enabled) the outlier-channel
        partition, once, from causal history only (warmup window, or the
        first quantized chunk when warmup=0)."""
        src = self._warm_k if (
            self._warm_k is not None and self._warm_k.shape[-2] > 0
        ) else fallback_keys
        src32 = src.to(torch.float32)
        if self.center_keys:
            self._mu = src32.mean(dim=-2, keepdim=True)
        if self.outlier_channels > 0:
            sample = src32 - self._mu if self.center_keys else src32
            idx = select_outlier_channels(sample, self.outlier_channels)
            self._split_k = OutlierSplitQuantizer(
                sample.shape[-1], idx,
                bits_outlier=self.bits_k_outlier, bits_regular=self.bits_k,
                codebook=self.codebook_kind, seed=self.seed + 700_000,
                device=sample.device,
            )
        self._stats_frozen = True

    def _encode_k(self, k: torch.Tensor):
        k32 = k.to(torch.float32)
        if self.center_keys:
            k32 = k32 - self._mu
        quantizer = self._split_k if self._split_k is not None else self._tq_k
        return quantizer.quantize(k32)

    def _decode_k(self, q) -> torch.Tensor:
        quantizer = self._split_k if self._split_k is not None else self._tq_k
        k32 = quantizer.dequantize(q)
        if self.center_keys:
            k32 = k32 + self._mu
        return k32.to(self.dtype)

    def _encode_v(self, v: torch.Tensor) -> QuantizedBatch:
        return self._tq_v.quantize(v.to(torch.float32))

    def _decode_v(self, q: QuantizedBatch) -> torch.Tensor:
        return self._tq_v.dequantize(q).to(self.dtype)

    @staticmethod
    def _cat(a, b):
        """Append batch b after batch a (either QuantizedBatch or
        SplitQuantizedBatch -- both expose .cat)."""
        return b if a is None else a.cat(b)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Split off tokens still inside the fp16 warmup window (they provide
        # the frozen key mean and attention-sink protection).
        n_new = key_states.shape[-2]
        take = max(0, min(self.warmup - self._len, n_new))
        self._len += n_new
        warm_k, key_q = key_states[..., :take, :], key_states[..., take:, :]
        warm_v, val_q = value_states[..., :take, :], value_states[..., take:, :]
        if take:
            self._warm_k = warm_k if self._warm_k is None else torch.cat(
                [self._warm_k, warm_k], dim=-2)
            self._warm_v = warm_v if self._warm_v is None else torch.cat(
                [self._warm_v, warm_v], dim=-2)
        has_new_q = key_q.shape[-2] > 0
        if has_new_q and not self._stats_frozen:
            self._freeze_key_stats(fallback_keys=key_q)

        if not self.packed:
            # simulate mode: quantize-dequantize exactly once, then hand the
            # result (warmup tokens untouched) to DynamicLayer fp storage.
            if has_new_q:
                key_q = self._decode_k(self._encode_k(key_q))
                val_q = self._decode_v(self._encode_v(val_q))
            k = torch.cat([warm_k, key_q], dim=-2) if take else key_q
            v = torch.cat([warm_v, val_q], dim=-2) if take else val_q
            return super().update(k, v, *args, **kwargs)

        if has_new_q:
            self._k = self._cat(self._k, self._encode_k(key_q))
            self._v = self._cat(self._v, self._encode_v(val_q))
        parts_k, parts_v = [], []
        if self._warm_k is not None:
            parts_k.append(self._warm_k)
            parts_v.append(self._warm_v)
        if self._k is not None:
            parts_k.append(self._decode_k(self._k))
            parts_v.append(self._decode_v(self._v))
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    def get_seq_length(self) -> int:
        if not self.packed:
            return super().get_seq_length()
        return self._len

    @property
    def quantized_bytes(self) -> int:
        """Device bytes of the cache store (packed mode), warmup included."""
        total = 0
        for q in (self._k, self._v):
            if q is not None:
                total += q.num_bytes
        for s in (self._warm_k, self._warm_v):
            if s is not None:
                total += s.nbytes
        if self._mu is not None:
            total += self._mu.nbytes
        return total

    # Beam/assisted-decoding cache surgery is out of scope for packed storage.
    def crop(self, max_length):
        raise NotImplementedError("TurboQuantLayer does not support crop")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError

    def batch_select_indices(self, indices):
        raise NotImplementedError


class TurboQuantCache(Cache):
    """Drop-in `past_key_values` quantizing K and V online with TurboQuant.

    Usage:
        cache = TurboQuantCache(bits=4)
        model(input_ids, past_key_values=cache, use_cache=True)
    """

    def __init__(
        self,
        bits: int = 4,
        seed: int = 0,
        warmup: int = 32,
        center_keys: bool = True,
        packed: bool = False,
        codebook: str = "sphere",
        bits_k: int | None = None,
        bits_v: int | None = None,
        outlier_channels: int = 0,
        bits_k_outlier: int | None = None,
        **kwargs,
    ) -> None:
        self._layer_kwargs = dict(
            bits=bits, seed=seed, warmup=warmup, center_keys=center_keys,
            packed=packed, codebook=codebook, bits_k=bits_k, bits_v=bits_v,
            outlier_channels=outlier_channels, bits_k_outlier=bits_k_outlier,
        )
        super().__init__(layer_class_to_replicate=self._make_layer, **kwargs)

    def _make_layer(self) -> TurboQuantLayer:
        # each layer gets its own rotation seed (derived, reproducible)
        kw = dict(self._layer_kwargs)
        kw["seed"] = kw["seed"] * 1000 + len(self.layers)
        return TurboQuantLayer(**kw)

    @property
    def quantized_bytes(self) -> int:
        return sum(layer.quantized_bytes for layer in self.layers)
