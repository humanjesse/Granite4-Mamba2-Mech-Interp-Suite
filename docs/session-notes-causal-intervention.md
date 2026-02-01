# Session Notes: Causal Intervention Feature

**Date:** 2026-02-01
**Feature branch:** uncommitted changes on `master`
**Scope:** New activation patching / causal intervention capability for the Granite 4.0 Mamba-2 interpretability suite

This document covers what was built, key design decisions, a critical analysis of how activation patching interacts with Mamba's recurrent state, and an audit of issues to address.

**Assumes:** Familiarity with PyTorch hooks, forward passes, and basic neural network architecture. Interpretability-specific concepts are explained inline.

---

## Table of Contents

1. [Summary of Uncommitted Changes](#1-summary-of-uncommitted-changes)
2. [Discussion Points](#2-discussion-points)
3. [Mamba vs Transformer Causal Roles — The Critical Finding](#3-mamba-vs-transformer-causal-roles--the-critical-finding)
4. [Audit Findings](#4-audit-findings)
5. [Recommended Next Steps](#5-recommended-next-steps)

---

## 1. Summary of Uncommitted Changes

Four files are affected: one new extraction module, one new visualization module, one new test file, and modifications to the Gradio app.

### 1.1 `src/extraction/causal_intervention.py` (new, ~420 lines)

This is the core engine. It implements **activation patching** — a technique from mechanistic interpretability where you:

1. Run a "clean" prompt (one that should produce the correct answer) and cache the model's internal activations at every layer
2. Run a "corrupted" prompt (one that produces the wrong answer) but intervene at a specific layer by swapping in the clean activations
3. Measure whether the correct answer is recovered — if so, that layer is causally responsible

**Key components:**

- `InterventionType` enum (line 23) — four intervention modes:
  - `ACTIVATION_PATCH` — replace corrupted activations with clean values
  - `ZERO_ABLATION` — set activations to zero (tests necessity)
  - `MEAN_ABLATION` — replace with sequence-mean (tests specificity)
  - `NOISE` — add Gaussian noise (tests robustness)

- `SweepResult` dataclass (line 32) — holds all results from a sweep: token lists, baseline logit diffs, probabilities, and the recovery matrix (1D for layer sweep, 2D for full position x layer sweep)

- `CausalInterventionEngine` class (line 56):
  - `resolve_token(text)` (line 72) — converts a token string like "Paris" to a token ID, handling the space-prefix tokenization issue where `"Paris"` and `" Paris"` may produce different IDs
  - `run_and_cache(prompt)` (line 95) — single forward pass with read-only hooks on all 32 layers, caches residual stream outputs to `self._prompt_cache`
  - `run_with_intervention(prompt, intervention_type, target_layer, positions, clean_cache)` (line 151) — single forward pass with a write hook on one layer that modifies the output tensor according to `intervention_type`
  - `sweep_layers(...)` (line 265) — patches all positions at each layer in turn (32 forward passes), producing a 1D recovery vector answering "which layers matter?"
  - `sweep_positions_and_layers(...)` (line 344) — patches one position at one layer at a time (32 x seq_len forward passes), producing a 2D heatmap

- **Recovery formula** (lines 317-318):
  ```
  recovery = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
  ```
  Where `logit_diff = logit[correct_token] - logit[incorrect_token]`. A recovery of 1.0 means the intervention fully restored the correct answer; 0.0 means no effect. Values above 1.0 or below 0.0 are possible and meaningful (overcorrection or further damage).

### 1.2 `src/visualization/intervention_viz.py` (new, ~287 lines)

Two dashboard layouts:

- `create_layer_sweep_dashboard()` (line 9) — 2-panel figure: bar chart of recovery per layer (purple for Mamba, teal for Transformer) with background shading by layer type, plus a text panel showing baselines and top-5 layers
- `create_full_sweep_dashboard()` (line 29) — 2x2 figure: position-by-layer heatmap (diverging RdBu colormap centered at 0), per-layer marginal bar chart, per-position marginal bar chart, and summary statistics
- `format_intervention_info()` (line 246) — text summary for the Gradio info panel
- `_truncate_token()` (line 277) — strips HuggingFace tokenizer prefixes (`"▁"`, `"Ġ"`) for cleaner display

Notable: the Mamba cascade annotation at line 98 warns users that "Mamba layers propagate state sequentially — patching cascades to later positions, which may yield higher recovery than Transformer layers." This foreshadows the deeper issue discussed in Section 3.

### 1.3 `tests/test_intervention.py` (new, ~523 lines)

Tests organized into groups:

| Group | What's tested |
|-------|---------------|
| `InterventionType` enum | String values, construction from strings |
| Recovery score math | Full recovery (1.0), no recovery (0.0), partial, zero denominator |
| `SweepResult` dataclass | 1D and 2D recovery matrices |
| Hook logic simulation | All 4 intervention types tested on raw tensors |
| Mock engine fixture | Realistic mock with tokenizer, model layers, arch_map |
| `resolve_token` | Single token, multi-token warning, empty input error, display format |
| Hook lifecycle | Cleanup after success, cleanup after exception |
| Caching | Cache hit, cache miss, cache clearing |
| Position handling | Out-of-bounds filtering, empty positions, mismatched seq_len |
| Numerical edge cases | bfloat16 precision, small denominators, NaN/Inf prevention |
| Integration-style | Sweep forward pass count, progress callback, result shapes |

**Coverage gap:** `sweep_positions_and_layers` has zero direct test coverage. See audit finding 5.2.1.

### 1.4 `app.py` modifications (+147/-10 lines)

- New imports for `CausalInterventionEngine`, `InterventionType`, and viz functions (lines 23-28)
- New globals `INTERVENTION_ENGINE` and `INTERVENTION_CACHE` (lines 37-38)
- Engine initialized in `initialize()` sharing the same `MODEL`, `TOKENIZER`, and `ARCH_MAP` (line 52)
- `intervention_view()` function (line 261) — validates inputs, checks cache, dispatches to layer or full sweep, returns figure + info text
- Five new Gradio controls: correct/incorrect token text inputs, token resolution display, intervention type radio, sweep mode radio — all hidden unless "Causal Intervention" view mode is selected
- Real-time token resolution via `.change()` callbacks on the token inputs (lines 455-469)
- "Comparison Prompt" renamed to "Comparison / Corrupted Prompt" since it's now shared between Activation Diff and Causal Intervention modes
- Security: `share=False` and `server_name="127.0.0.1"` replaces `share=True` and `0.0.0.0`

---

## 2. Discussion Points

### 2.1 Tokenizer Reuse

No custom tokenizer was written. The `CausalInterventionEngine` receives the existing HuggingFace tokenizer instance from `initialize()` and calls standard methods: `.encode()`, `.decode()`, `.__call__()`, `.convert_ids_to_tokens()`.

The `resolve_token()` method (line 72) is a thin wrapper that:
1. Tokenizes the input text
2. Takes the last sub-token if the word splits into multiple tokens (e.g., "Warsaw" might become `["War", "saw"]`)
3. Issues a warning if multi-token splitting occurs
4. Returns a display string showing the exact resolution (e.g., `"Paris" -> token 1234 ("Paris")`) so users can verify the tokenization is correct

This matters because activation patching measures the logit difference between a "correct" and "incorrect" token. If those tokens are resolved incorrectly (e.g., getting the first sub-token instead of the last), the recovery scores are meaningless.

### 2.2 Mamba vs Transformer Causal Roles

The Granite 4.0 model has 28 Mamba layers and 4 Transformer layers (at indices 10, 13, 17, 27). This extreme ratio means most computation happens in Mamba layers. The activation patching sweep can help answer:

- **Are factual associations stored in the few Transformer attention layers** (which have global token-to-token attention and could implement factual lookup) **or distributed across Mamba layers** (which process information sequentially through recurrent state)?
- **Do Mamba and Transformer layers play different roles?** For example, early Mamba layers might build up context representations while Transformer layers at positions 10, 13, 17 might perform specific information routing, and the final Transformer layer at 27 might handle output formatting.
- **Does the position of a Transformer layer within the Mamba stack matter?** Layer 10 (surrounded by Mamba layers) might serve a different function than layer 27 (near the output).

The visualization already color-codes layer types, making these comparisons immediately visible in the bar charts.

### 2.3 Research Directions

**Path patching:** Instead of patching an entire layer, patch the *connection between two specific layers*. This traces information flow through the computation graph — e.g., "factual knowledge flows from Mamba layer 8 through Transformer layer 10's attention heads to the output."

**Indirect vs direct effects:** The current sweep conflates two things: a layer's direct effect on the output logits, and its indirect effect through downstream layers. Decomposing these requires patching layer L and measuring the change at each subsequent layer, not just at the final output.

**Head-level patching for Transformer layers:** The 4 Transformer layers each have multiple attention heads. Patching individual heads (rather than the full layer) would identify which specific heads perform factual recall. This complements the existing head-level attention visualization.

**Cross-referencing with hidden attention:** The existing extraction tools compute Mamba hidden attention patterns (the implicit attention matrix derived from dt, B, C parameters). Comparing these patterns with causal intervention results could reveal whether "attending to" a token (high attention weight) corresponds to being "causally responsible" for the output (high recovery score). Mismatches would be particularly interesting.

---

## 3. Mamba vs Transformer Causal Roles — The Critical Finding

### 3.1 How the Hook Works

The intervention hooks register on `model.model.layers[layer_idx]`, which is a `GraniteMoeHybridDecoderLayer` wrapper. Regardless of whether the inner sublayer is Mamba or Transformer, this wrapper always outputs a single tensor (the residual stream hidden state). The hook code correctly handles both cases:

```python
# causal_intervention.py:122-127 (read hook)
def hook_fn(module, input, output, _idx=layer_idx):
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    storage[_idx] = hidden.detach().clone().cpu()
```

This is correct at the hooking level — both layer types produce a tensor at the decoder-layer boundary.

### 3.2 The Problem: Mamba's Internal Recurrent State

Activation patching was developed for Transformer-only models where every layer is stateless. Mamba layers are fundamentally different — they maintain internal recurrent state that evolves as each token is processed.

```
TRANSFORMER LAYER (stateless):

  residual_in --> [ Attention(Q,K,V) + MLP ] --> residual_out
                                                     ^
                                                     |
                                            Hook patches HERE.
                                            This fully corrects the
                                            information flowing to
                                            the next layer.


MAMBA LAYER (stateful):

  residual_in --> [ Conv1D --> SSM computation --> output proj ] --> residual_out
                     |              |                                     ^
                     v              v                                     |
                conv_states    ssm_states                        Hook patches HERE.
                (internal)     (internal)                        But the SSM state
                                                                 inside the layer
                These states evolve token-by-token               remains corrupted.
                and persist across the sequence.
                The hook CANNOT see or modify them.
```

Specifically, Mamba layers maintain two internal state buffers:
- `conv_states` — a sliding window buffer for the 1D convolution (shape: `[batch, d_inner, d_conv]`)
- `ssm_states` — the recurrent state of the selective state space model (shape: `[batch, d_inner, d_state]`)

These states are updated inside the Mamba forward pass *before* the output is produced. When we patch the output, the internal state has already been computed from the corrupted input and remains corrupted.

### 3.3 What This Means for Recovery Scores

For **Transformer layers** (indices 10, 13, 17, 27): patching the residual stream output fully corrects the information flowing to downstream layers. Recovery scores are trustworthy.

For **Mamba layers** (the other 28 layers): patching the output corrects what downstream layers *see*, but the SSM state inside the patched layer remains corrupted. If a downstream Mamba layer reads state that was influenced by the corrupted layer's internal state (through the residual stream that was subsequently patched), the correction is incomplete.

Possible consequences:
- **Mamba recovery scores may be underestimated.** A Mamba layer might be causally important, but because patching its output doesn't fix the internal state damage, we measure less recovery than the layer truly contributes.
- **Mamba layers may show artificially high recovery in some cases.** The cascade annotation in `intervention_viz.py:98` already notes this: because Mamba state propagates sequentially, patching one layer's output can "fix" the input to the next Mamba layer, which then propagates the correction through its own state.
- **Comparison between Mamba and Transformer recovery scores is not apples-to-apples.** The measurement is confounded by the state issue for Mamba but not for Transformer layers.

### 3.4 This Is a Known Limitation, Not a Bug

This is an inherent limitation of residual-stream-level activation patching when applied to recurrent architectures. The code is implemented correctly for what it does — it patches residual stream outputs. The limitation is in what that *means* for Mamba layers.

The existing annotation in the visualization (`intervention_viz.py:98-109`) partially communicates this, but a more prominent warning would help users interpret results correctly.

### 3.5 Future Approaches

- **SSM-state-level patching:** Hook deeper into the Mamba layer to also patch `conv_states` and `ssm_states` from the clean run. This would make Mamba recovery scores more comparable to Transformer scores, but requires hooking internal Mamba submodules rather than the decoder-layer wrapper.
- **Separate sweep analysis:** Run layer sweeps for Mamba and Transformer layers independently and apply different interpretation frameworks to each.
- **Input-side patching:** Instead of patching the layer's output, patch its input (the residual stream arriving at the layer). This tests "does this layer need the correct input to produce the correct output?" rather than "does this layer's output need to be correct?"

---

## 4. Audit Findings

### 4.1 Critical

**Mamba SSM state not patched during intervention**
- **Where:** `causal_intervention.py:176-216` (the `make_intervention_hook` closure)
- **What:** The hook modifies the decoder layer's output tensor but cannot access or modify the Mamba layer's internal `conv_states` and `ssm_states`
- **Impact:** Recovery scores for 28 of 32 layers (all Mamba layers) may be systematically inaccurate. Cross-layer-type comparisons are confounded.
- **Why it matters:** This is the core measurement the tool produces. If it's systematically biased for the majority of layers, users may draw incorrect conclusions about which layers store factual knowledge.

### 4.2 High

**4.2.1 `sweep_positions_and_layers` has zero test coverage**
- **Where:** `tests/test_intervention.py` — no `test_sweep_positions_and_layers_*` functions exist
- **What:** The most expensive method (N_layers x seq_len forward passes, potentially 32 x 50+ = 1600 calls) is completely untested
- **Why it matters:** This is the code path most likely to have edge cases (2D indexing, position-layer iteration order). It also produces the full heatmap — the most complex visualization.

**4.2.2 `INTERVENTION_CACHE` grows without bound**
- **Where:** `app.py:38` (declaration), `app.py:303` (writes)
- **What:** Each `SweepResult` stores a recovery matrix (up to `[32, seq_len]` floats), two token lists, and several scalars. The cache is never evicted.
- **Why it matters:** With varied inputs over a long session, memory usage grows indefinitely. The existing `CACHE` and `MULTISTEP_CACHE` have the same issue, but intervention results tend to be larger due to the 2D matrix.

**4.2.3 `INTERVENTION_ENGINE` null checks missing**
- **Where:** `app.py:261` (`intervention_view` function), `app.py:455-469` (`update_token_info` closure)
- **What:** If `initialize()` fails or hasn't completed, `INTERVENTION_ENGINE` is `None`. Both `intervention_view()` and the token resolution callback will raise `AttributeError`.
- **Why it matters:** Initialization failures should show a user-friendly error, not a stack trace.

**4.2.4 Thread-safety on `_prompt_cache`**
- **Where:** `causal_intervention.py:70` (declaration), lines 107-108 and 144 (read/write)
- **What:** `_prompt_cache` is a plain dict. Gradio can dispatch concurrent requests, and two threads could race on checking and populating the same cache key.
- **Why it matters:** Race conditions could cause duplicate computation (wasteful) or, worse, one thread reading a partially-written cache entry (crash or corrupted results).

### 4.3 Medium

**4.3.1 `noise_std` not calibrated to activation magnitudes**
- **Where:** `causal_intervention.py:159` (parameter default), line 211 (usage)
- **What:** The default `noise_std=1.0` is arbitrary. Depending on the model's activation scale, this could be negligibly small (if activations have magnitude ~100) or overwhelmingly large (if activations are ~0.01).
- **Why it matters:** The noise intervention is meant to test "how robust is this layer's computation to perturbation?" If the noise magnitude is miscalibrated, the answer is always "very robust" (noise too small) or "not at all" (noise too large).

**4.3.2 Zero denominator silently masked as recovery = 0.0**
- **Where:** `causal_intervention.py:317-320`, lines 394-397
- **What:** When `clean_logit_diff == corrupted_logit_diff` (denominator < 1e-6), recovery is set to 0.0. No warning is logged.
- **Why it matters:** This condition means the clean and corrupted prompts produce the same logit difference for the target tokens — the experiment is not well-posed. The user should know this, not see a silent 0.0 that looks like "no layer matters."

**4.3.3 Progress callbacks not wired in the UI**
- **Where:** `app.py:289-302` (`intervention_view` function)
- **What:** Both sweep methods accept a `progress_callback` parameter, but the app never passes one.
- **Why it matters:** The full sweep can require 1000+ forward passes. With no progress indication, users may think the app has frozen.

**4.3.4 Special token handling in visualization**
- **Where:** `intervention_viz.py:280-285` (`_truncate_token`)
- **What:** Strips `"▁"` and `"Ġ"` prefixes but doesn't handle other special tokens (`<s>`, `</s>`, `<pad>`). A token like `"▁"` alone becomes an empty string after stripping.
- **Why it matters:** Empty labels on heatmap axes are confusing. Special tokens appearing as raw strings (`<s>`) look like HTML in some renderers.

### 4.4 Low

**4.4.1 Magic number `1e-6` hardcoded in two places**
- **Where:** `causal_intervention.py:317` and `causal_intervention.py:394`
- **What:** The denominator threshold is a bare literal in both `sweep_layers` and `sweep_positions_and_layers`.
- **Suggestion:** Extract to a class constant like `DENOMINATOR_THRESHOLD = 1e-6`.

**4.4.2 Empty `__init__.py` exports**
- **Where:** `src/extraction/__init__.py`, `src/visualization/__init__.py`
- **What:** These are empty files. Users must use full import paths like `from src.extraction.causal_intervention import CausalInterventionEngine`.
- **Suggestion:** Adding `__all__` exports would improve discoverability, though this is stylistic and the current approach works fine.

---

## 5. Recommended Next Steps

Ordered by priority. Items 1-5 are concrete fixes; items 6-9 are enhancements.

1. **Add tests for `sweep_positions_and_layers`** — This is the biggest test gap. At minimum: verify the result shape is 2D `[num_layers, seq_len]`, verify the correct number of forward passes, and test the progress callback.

2. **Add a warning when the denominator is near zero** — Log a warning or return a special message in the info panel when `clean_logit_diff ~= corrupted_logit_diff`. This helps users catch ill-posed experiments.

3. **Wire progress callbacks in the UI** — Gradio supports `gr.Progress`. Passing a callback to the sweep methods would show a progress bar during long runs.

4. **Add `INTERVENTION_ENGINE` null guard** — Check for `None` in `intervention_view()` and `update_token_info()`, return a friendly error message.

5. **Add cache eviction** — Implement a max-size limit on `INTERVENTION_CACHE` (and ideally the other caches too). An LRU strategy with a configurable cap would prevent unbounded memory growth.

6. **Document the SSM limitation in the UI** — The existing annotation in the bar chart (intervention_viz.py:98-109) is a start. Consider adding a more prominent note in the Gradio info panel when Mamba-layer recovery scores are displayed.

7. **Calibrate `noise_std`** — Compute the actual activation norm at each layer and scale noise relative to it (e.g., `noise_std = 0.5 * activation_norm`). This makes the noise intervention actually informative.

8. **Extract the `1e-6` threshold to a named constant** — Minor cleanup for consistency and discoverability.

9. **Research: prototype SSM-state-level patching** — Long-term. Would require hooking into the Mamba layer's internal submodules to capture and restore `conv_states` and `ssm_states`. This would make Mamba recovery scores trustworthy and enable fair cross-layer-type comparison.
