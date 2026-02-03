# Session Notes: Causal Intervention & Circuit Discovery

**Date:** 2026-02-01 (initial), updated 2026-02-02
**Feature branch:** uncommitted changes on `master`
**Scope:** Causal intervention (activation patching), circuit discovery (path patching), shared base class refactor, Plotly visualizations, hardening, and expanded test coverage

This document covers what was built, key design decisions, a critical analysis of how activation patching interacts with Mamba's recurrent state, and an audit of remaining issues.

**Assumes:** Familiarity with PyTorch hooks, forward passes, and basic neural network architecture. Interpretability-specific concepts are explained inline.

---

## Table of Contents

1. [Summary of Changes](#1-summary-of-changes)
2. [Discussion Points](#2-discussion-points)
3. [Mamba vs Transformer Causal Roles — The Critical Finding](#3-mamba-vs-transformer-causal-roles--the-critical-finding)
4. [Audit Findings](#4-audit-findings)
5. [Remaining Issues](#5-remaining-issues)

---

## 1. Summary of Changes

Nine files are affected across three categories: extraction engines, visualizations, and tests.

### 1.1 `src/extraction/intervention_base.py` (new, 180 lines)

Shared base class extracted from duplicated code in `CausalInterventionEngine` and `CircuitDiscoveryEngine`. Both engines now inherit from `InterventionBase`.

**Key components:**

- `MAX_SEQ_LEN = 256` and `MIN_RECOVERY_DENOMINATOR = 1e-6` — constants formerly hardcoded in multiple places
- `InterventionBase.__init__()` — validates model/tokenizer/arch_map are not None, stores references, initializes `_prompt_cache`
- `resolve_token(text)` — converts token text to ID, handles space-prefix tokenization, warns on multi-token splits
- `run_and_cache(prompt)` — single forward pass with hooks on embedding + all 32 layers, caches `{residual_stream, embedding, logits, tokens}` per prompt
- `clear_cache()` — evicts the internal prompt cache
- `_get_logit_diff(logits, correct_id, incorrect_id)` — computes logit difference at last position
- `_compute_baselines(clean_cache, corrupted_cache, correct_id, incorrect_id)` — computes clean/corrupted logit diffs, denominator, and probabilities
- `_compute_layer_output(prompt)` — computes isolated per-layer contributions to the residual stream (layer L output = residual after L − residual before L)

### 1.2 `src/extraction/causal_intervention.py` (modified, 321 lines — down from ~420)

Now inherits from `InterventionBase`. The duplicated `resolve_token`, `run_and_cache`, `clear_cache`, baseline computation, and constant definitions have been removed.

**Retained components:**

- `InterventionType` enum — four modes: `ACTIVATION_PATCH`, `ZERO_ABLATION`, `MEAN_ABLATION`, `NOISE`
- `SweepResult` dataclass — recovery matrix (1D or 2D), token lists, baseline metrics
- `CausalInterventionEngine` — now only contains intervention-specific logic:
  - `run_with_intervention()` — forward pass with a write hook on one layer
  - `sweep_layers()` — 1D sweep (32 forward passes)
  - `sweep_positions_and_layers()` — 2D sweep (32 × seq_len forward passes)

**Recovery formula** (unchanged):
```
recovery = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
```

### 1.3 `src/extraction/circuit_discovery.py` (modified, 672 lines)

Now inherits from `InterventionBase`. Implements circuit discovery via path patching — tracing information flow between specific pairs of layers rather than patching one layer at a time.

**Key components:**

- `SweepGranularity` enum — `FAST` (layer paths only) or `DETAILED` (layer paths + component-level)
- `PathPatchResult`, `ComponentResult`, `CircuitNode`, `CircuitEdge`, `CircuitResult` — data classes for structured results
- `CircuitDiscoveryEngine`:
  - `run_with_layer_patch(prompt, target_layer, replacement_output)` — patches one layer's output with a computed tensor (for path patching)
  - `_path_patch_single(source, target, ...)` — patches the connection between two specific layers
  - `_patch_attention_head(layer_idx, head_idx, ...)` — patches a single attention head within a Transformer layer
  - `sweep_paths(...)` — O(n²/2) sweep of all source→target layer pairs, produces [num_layers × num_layers] importance matrix
  - `sweep_components(...)` — sweeps individual attention heads within important Transformer layers
  - `_extract_circuit(...)` — thresholds the path matrix and component results to extract nodes and edges
  - `find_circuit(...)` — top-level method combining layer importance, path patching, optional component patching, and circuit extraction
  - All sweep methods accept `progress_callback` for UI integration

### 1.4 `src/visualization/circuit_viz.py` (modified, 751 lines — up from ~170)

Added Plotly-based interactive visualizations for the Gradio tabbed UI alongside the existing matplotlib dashboard.

**New Plotly functions:**

- `create_path_matrix_plotly(result, arch_map)` — interactive heatmap of the [num_layers × num_layers] path importance matrix, with layer type annotations
- `create_layer_importance_plotly(result, arch_map)` — bar chart of per-layer importance scores, color-coded by Mamba vs Transformer
- `create_circuit_diagram_plotly(result, arch_map)` — network-style diagram showing discovered circuit nodes and edges, filters to top-N nodes
- `create_component_importance_plotly(result, arch_map)` — grid of per-head importance for Transformer layers that were component-swept
- `create_circuit_summary_markdown(result, arch_map)` — markdown text summary of circuit statistics and top nodes/edges

All Plotly functions handle `None` inputs gracefully (return empty figures with instructional text).

**Existing matplotlib functions** (unchanged):

- `create_circuit_dashboard(result, arch_map)` — multi-panel static figure
- `format_circuit_info(result)` — text summary for the info panel

### 1.5 `app.py` (modified, 762 lines)

**Circuit discovery UI overhaul:**

- `circuit_discovery_view()` now returns `(circuit_result, info)` instead of `(fig, info)` — the raw result object is passed to individual Plotly renderers
- New tabbed output panel for Circuit Discovery mode with 5 tabs: Path Matrix, Layer Importance, Circuit Diagram, Components, Summary
- `single_plot_wrapper` and `circuit_output_wrapper` toggle visibility based on view mode
- `_default_circuit_outputs()` helper returns default `gr.update()` values for the 7 circuit-specific output slots
- `analyze()` function now accepts `progress=gr.Progress(track_tqdm=False)` and returns 10 outputs instead of 3

**Hardening fixes:**

- Null guards on both `INTERVENTION_ENGINE` and `CIRCUIT_ENGINE` — return friendly error messages if initialization failed
- `INTERVENTION_CACHE` bounded to 20 entries with oldest-first eviction
- `CIRCUIT_CACHE` bounded to 10 entries with oldest-first eviction
- Input stripping moved to the top of both view functions (strip once, use everywhere)
- Progress callbacks wired for both circuit discovery (`cd_progress`) and causal intervention (`iv_progress`) via `gr.Progress`

### 1.6 `tests/test_intervention.py` (modified, 736 lines — up from ~523)

**New tests for `sweep_positions_and_layers`** (formerly zero coverage):

| Test | What it verifies |
|------|------------------|
| `test_full_sweep_result_shape` | Recovery matrix is 2D `[num_layers, seq_len]` |
| `test_full_sweep_forward_pass_count` | Exactly `num_layers × seq_len` forward passes |
| `test_full_sweep_progress_callback` | Callback fired with monotonically increasing steps |
| `test_full_sweep_layer_subset` | `layer_subset` parameter reduces sweep scope |
| `test_full_sweep_each_call_patches_single_position` | Each call patches exactly one position |
| `test_full_sweep_no_nan_inf` | No NaN or Inf in recovery matrix |
| `test_full_sweep_populates_sweep_result_fields` | All SweepResult fields populated correctly |

Also added: `test_residual_stream_has_only_integer_keys` verifying cache structure.

### 1.7 `tests/test_circuit_discovery.py` (modified, 783 lines — up from ~193)

Comprehensive test suite covering:

- Data class construction and validation for all 5 circuit data types
- Constructor validation (rejects None model/tokenizer/arch_map)
- `resolve_token` via inheritance from `InterventionBase`
- Circuit extraction: empty below threshold, finds important nodes, finds edges, includes components
- Path patching: source ≥ target returns zero, equal layers return zero
- Attention head patching: returns zero for Mamba layers, returns zero when no `self_attn`
- Hook lifecycle: cleanup on success and on exception
- Caching: hit, miss, clear
- `run_with_layer_patch`: returns logits, hook cleanup on success and exception
- `sweep_paths`: correct shapes, forward pass counts, upper-triangle fill, progress callback
- `sweep_components`: only sweeps Transformer layers, filters by important layers, progress callback, one clean pass per layer
- `find_circuit`: fast mode, detailed mode calls components, detailed progress monotonic, empty tokens raises
- `_compute_layer_output`: per-layer contributions, layer 0 uses embedding
- `_patch_attention_head`: returns zero when no `self_attn` attribute

### 1.8 `tests/test_circuit_viz.py` (modified, 231 lines — up from ~150)

Added tests for all Plotly visualization functions:

- `test_path_matrix_plotly_returns_figure`
- `test_layer_importance_plotly_returns_figure`
- `test_circuit_diagram_plotly_returns_figure`
- `test_circuit_diagram_plotly_filters_top_n`
- `test_component_importance_plotly_returns_figure` / `_empty`
- `test_circuit_summary_markdown_returns_string`
- `test_plotly_functions_handle_none`

### 1.9 `requirements.txt` (modified)

Added `plotly` dependency for the new interactive visualizations.

---

## 2. Discussion Points

### 2.1 Tokenizer Reuse

No custom tokenizer was written. The `InterventionBase` (and by inheritance both engines) receives the existing HuggingFace tokenizer instance from `initialize()` and calls standard methods: `.encode()`, `.decode()`, `.__call__()`, `.convert_ids_to_tokens()`.

The `resolve_token()` method is a thin wrapper that:
1. Tokenizes the input text
2. Takes the last sub-token if the word splits into multiple tokens (e.g., "Warsaw" might become `["War", "saw"]`)
3. Issues a warning if multi-token splitting occurs
4. Returns a display string showing the exact resolution (e.g., `"Paris" → token 1234 ("Paris")`) so users can verify the tokenization is correct

This matters because activation patching measures the logit difference between a "correct" and "incorrect" token. If those tokens are resolved incorrectly (e.g., getting the first sub-token instead of the last), the recovery scores are meaningless.

### 2.2 Mamba vs Transformer Causal Roles

The Granite 4.0 model has 28 Mamba layers and 4 Transformer layers (at indices 10, 13, 17, 27). This extreme ratio means most computation happens in Mamba layers. The activation patching sweep can help answer:

- **Are factual associations stored in the few Transformer attention layers** (which have global token-to-token attention and could implement factual lookup) **or distributed across Mamba layers** (which process information sequentially through recurrent state)?
- **Do Mamba and Transformer layers play different roles?** For example, early Mamba layers might build up context representations while Transformer layers at positions 10, 13, 17 might perform specific information routing, and the final Transformer layer at 27 might handle output formatting.
- **Does the position of a Transformer layer within the Mamba stack matter?** Layer 10 (surrounded by Mamba layers) might serve a different function than layer 27 (near the output).

The visualization already color-codes layer types, making these comparisons immediately visible in the bar charts.

### 2.3 Research Directions

**Path patching (now implemented):** Instead of patching an entire layer, patch the *connection between two specific layers*. This traces information flow through the computation graph — e.g., "factual knowledge flows from Mamba layer 8 through Transformer layer 10's attention heads to the output." The `sweep_paths()` method produces a full [num_layers × num_layers] importance matrix showing all pairwise connections.

**Component-level patching (now implemented):** The 4 Transformer layers each have multiple attention heads. The `sweep_components()` method patches individual heads to identify which specific heads perform factual recall.

**Indirect vs direct effects:** The current sweep conflates two things: a layer's direct effect on the output logits, and its indirect effect through downstream layers. Decomposing these requires patching layer L and measuring the change at each subsequent layer, not just at the final output.

**Cross-referencing with hidden attention:** The existing extraction tools compute Mamba hidden attention patterns (the implicit attention matrix derived from dt, B, C parameters). Comparing these patterns with causal intervention results could reveal whether "attending to" a token (high attention weight) corresponds to being "causally responsible" for the output (high recovery score). Mismatches would be particularly interesting.

### 2.4 InterventionBase Refactor

The original `CausalInterventionEngine` and `CircuitDiscoveryEngine` duplicated ~150 lines of identical code: token resolution, forward-pass caching with hooks, logit diff computation, and baseline calculation. These were extracted into `InterventionBase`.

Design decisions:
- **Inheritance over composition** — both engines *are* intervention runners with shared state (`_prompt_cache`, model references). Composition would have required forwarding every method call.
- **Constants at module level** — `MAX_SEQ_LEN` and `MIN_RECOVERY_DENOMINATOR` are importable from `intervention_base` so tests can reference them without magic numbers.
- **`_compute_layer_output()` in base** — used by circuit discovery for path patching (computing the isolated contribution of source layers). Placed in base because it only depends on `run_and_cache()`.

---

## 3. Mamba vs Transformer Causal Roles — The Critical Finding

### 3.1 How the Hook Works

The intervention hooks register on `model.model.layers[layer_idx]`, which is a `GraniteMoeHybridDecoderLayer` wrapper. Regardless of whether the inner sublayer is Mamba or Transformer, this wrapper always outputs a single tensor (the residual stream hidden state). The hook code correctly handles both cases:

```python
# intervention_base.py:95-100 (read hook)
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

### 4.1 Critical (unchanged)

**Mamba SSM state not patched during intervention**
- **Where:** `causal_intervention.py` — the `make_intervention_hook` closure
- **What:** The hook modifies the decoder layer's output tensor but cannot access or modify the Mamba layer's internal `conv_states` and `ssm_states`
- **Impact:** Recovery scores for 28 of 32 layers (all Mamba layers) may be systematically inaccurate. Cross-layer-type comparisons are confounded.
- **Why it matters:** This is the core measurement the tool produces. If it's systematically biased for the majority of layers, users may draw incorrect conclusions about which layers store factual knowledge.

### 4.2 Resolved Issues

The following issues from the original audit have been addressed:

| Issue | Resolution |
|-------|------------|
| **`sweep_positions_and_layers` zero test coverage** (was 4.2.1) | 7 new tests added: shape, forward pass count, progress callback, layer subset, single-position patching, NaN/Inf, field population |
| **`INTERVENTION_CACHE` grows without bound** (was 4.2.2) | Bounded to 20 entries (`INTERVENTION_CACHE_MAX`), oldest-first eviction at `app.py:321-323` |
| **`CIRCUIT_CACHE` grows without bound** (was 4.2.2) | Bounded to 10 entries (`CIRCUIT_CACHE_MAX`), oldest-first eviction at `app.py:374-376` |
| **`INTERVENTION_ENGINE` null checks missing** (was 4.2.3) | Null guard at `app.py:279` — returns friendly error |
| **`CIRCUIT_ENGINE` null checks missing** (was 4.2.3) | Null guard at `app.py:338` — returns friendly error |
| **Progress callbacks not wired in UI** (was 4.3.3) | Circuit discovery wired at `app.py:649-655`, causal intervention wired at `app.py:695-697`, both use `gr.Progress` |
| **Magic number `1e-6` hardcoded** (was 4.4.1) | Extracted to `MIN_RECOVERY_DENOMINATOR` constant in `intervention_base.py:17` |

---

## 5. Remaining Issues

### 5.1 High

**Thread-safety on `_prompt_cache`**
- **Where:** `intervention_base.py:35` (declaration), lines 72-73 and 118 (read/write)
- **What:** `_prompt_cache` is a plain dict. Gradio can dispatch concurrent requests, and two threads could race on checking and populating the same cache key.
- **Why it matters:** Race conditions could cause duplicate computation (wasteful) or, worse, one thread reading a partially-written cache entry (crash or corrupted results).

### 5.2 Medium

**`CACHE` and `MULTISTEP_CACHE` grow without bound**
- **Where:** `app.py:45-46`
- **What:** The original extraction cache and multi-step generation cache have no size limits, unlike the now-bounded intervention and circuit caches.
- **Why it matters:** Long sessions with varied inputs will accumulate memory indefinitely.

**`noise_std` not calibrated to activation magnitudes**
- **Where:** `causal_intervention.py` — noise intervention hook
- **What:** The default `noise_std=1.0` is arbitrary. Depending on the model's activation scale, this could be negligibly small or overwhelmingly large.
- **Why it matters:** The noise intervention is meant to test robustness. Miscalibrated noise makes the results uninformative.

**Zero denominator silently masked as recovery = 0.0**
- **Where:** Recovery computation in `causal_intervention.py`
- **What:** When `clean_logit_diff == corrupted_logit_diff` (denominator < `MIN_RECOVERY_DENOMINATOR`), recovery is set to 0.0 with no warning.
- **Why it matters:** This condition means the experiment is ill-posed (clean and corrupted prompts produce the same logit difference). Users should be informed, not shown a silent 0.0.

### 5.3 Low

**Special token handling in visualization**
- **Where:** `intervention_viz.py` — `_truncate_token()`
- **What:** Strips `"▁"` and `"Ġ"` prefixes but doesn't handle `<s>`, `</s>`, `<pad>`. A bare `"▁"` becomes an empty string.
- **Why it matters:** Empty labels on heatmap axes are confusing.

**Empty `__init__.py` exports**
- **Where:** `src/extraction/__init__.py`, `src/visualization/__init__.py`
- **What:** Empty files; users must use full import paths.
- **Why it matters:** Stylistic only. Current approach works fine.
