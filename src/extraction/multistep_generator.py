"""Autoregressive generation with full interpretability extraction at each step."""

import torch


def run_multistep_generation(extractor, tokenizer, prompt, max_tokens=5, temperature=0.0):
    """Generate tokens one at a time with full extraction at each step.

    Args:
        extractor: GraniteAttentionExtractor instance.
        tokenizer: HuggingFace tokenizer.
        prompt: Initial prompt string.
        max_tokens: Number of tokens to generate (1-20).
        temperature: Sampling temperature. 0.0 = greedy.

    Returns:
        Dict with:
            original_prompt: str
            steps: list of per-step dicts
            final_text: str
    """
    steps = []
    current_text = prompt.strip()

    for step_idx in range(max_tokens):
        result = extractor.extract(current_text)

        # Get next token from logits
        logits = result["logits"]  # [1, seq_len, vocab_size]
        next_logits = logits[0, -1, :].float()

        if temperature <= 0:
            next_token_id = next_logits.argmax().item()
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

        # Top-5 predictions
        probs = torch.softmax(next_logits, dim=-1)
        top5_probs, top5_ids = probs.topk(5)
        top_predictions = [
            (tokenizer.decode(tid.item()).strip() or repr(tokenizer.convert_ids_to_tokens(tid.item())),
             tp.item())
            for tid, tp in zip(top5_ids, top5_probs)
        ]

        generated_token = tokenizer.decode(next_token_id)

        steps.append({
            "step_idx": step_idx,
            "prompt_so_far": current_text,
            "generated_token": generated_token,
            "generated_token_id": next_token_id,
            "extraction_result": result,
            "top_predictions": top_predictions,
        })

        # Extend text for next iteration
        current_text = tokenizer.decode(
            tokenizer.encode(current_text, add_special_tokens=False) + [next_token_id],
            skip_special_tokens=False,
        )

        # Stop on EOS
        if next_token_id == tokenizer.eos_token_id:
            break

    return {
        "original_prompt": prompt.strip(),
        "steps": steps,
        "final_text": current_text,
    }
