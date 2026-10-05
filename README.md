# Constrained decoding: lipograms and forbidden sequences

MVA **LLM for Code and Proof — TP1**, archived from Ying Jin's coursework.
The implementation performs autoregressive decoding through direct model forward
passes and logit masking, with beam search, rather than `model.generate()`.

## Two exercises

- **La Disparition:** construct a tokenizer-dependent mask for vocabulary entries
  containing `e`/`E`, then run a width-three beam search.
- **Toulouse Sequence:** enumerate spelling/whitespace and split-token variants,
  track matching sequence suffixes, and mask tokens that would complete a forbidden sequence.

`challenge.py` is the implemented local file originally named `challenge_v1 (1).py`.
`course_template/challenge.py` is the original unimplemented exercise template,
included separately to make the starting point visible. The empty peer-named file
was not included. Course exercise wording is retained; this is a student adaptation,
not a claim of authorship of the assignment or pretrained model.

## Run

Use Python 3.10+ with a suitable PyTorch environment:

```bash
python -m pip install -r requirements.txt
python challenge.py
```

The example downloads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` when not cached and uses
FP16 with automatic device placement. Choose a compatible device/dtype before a
CPU-only run. Dependencies above are packaging guidance, not a reconstructed
lockfile from the original coursework environment.

## What this snapshot demonstrates

Manual decoding loops, tokenizer-aware masking, cumulative log-probability beam
scoring, and suffix-based constraints. No benchmark score or universal constraint
guarantee is claimed. The publication check parsed the Python sources; it did not
download a model or rerun inference.

## Known limitations preserved from the coursework

- `LaDisparition` returns prompt plus completion, so an `e` already in the prompt
  remains in the returned text. Its EOS early-stop code is commented out.
- Matching raw vocabulary strings is not a tokenizer-independent proof about
  decoded text; byte/normalization/context effects need explicit testing.
- The Toulouse variant enumeration is heuristic, not an exhaustive automaton over
  every possible tokenization. Its `beam_width` expression takes `len()` of a
  `(prefix, target)` pair, so it evaluates to two when the list is nonempty.
- Empty forbidden lists, all-masked distributions, special tokens and scoring
  bias from different completion lengths need further handling.
- The loop recomputes prefixes and does not implement a KV cache.

The implementation is preserved instead of silently changing historical behavior.
See [provenance](docs/PROVENANCE.md) and [file hashes](docs/source_manifest.json).
