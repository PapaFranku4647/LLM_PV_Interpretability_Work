# Matched CodeBoost Pilot

Run completed on 2026-04-17 around 22:18 local time.

This is a one-trial pilot, not a paper-grade aggregate. It uses the same split
sizes as the core baseline matrix:

- train size: 256
- validation size: 256
- test size: 2000
- batch size: 256
- boost rounds: 1
- trials: 1 per dataset
- round retries: up to 8
- sampling: without replacement
- acceptance: `max_weak_error=0.3025`, with best-valid fallback up to 0.499
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/matched_codeboost_pilot_t1_b256_s1/`

## Results

| Function | Dataset | Representation | Test accuracy | Best baseline | Gap | API attempts | Estimated cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fn_o` | CDC diabetes | semantic | 0.7060 | 0.7225 | -0.0165 | 1 | $0.0645 |
| `fn_n` | mushroom | obfuscated | 0.5755 | 0.8528 | -0.2773 | 1 | $0.0970 |
| `fn_p` | HTRU2 | obfuscated | 0.8830 | 0.9340 | -0.0510 | 1 | $0.1935 |
| `fn_q` | chess | obfuscated | 0.5775 | 0.9615 | -0.3840 | 8 | $0.9341 |

Total estimated pilot cost: about $1.2891.

## Interpretation

CDC remains the strongest current story. The matched one-trial semantic run got
0.7060 test accuracy, close to the prior 30-seed semantic CDC mean of 0.7032 and
only 1.65 percentage points behind the best matched baseline, logistic
regression at 0.7225.

HTRU2 is the second-best signal. One generated learner reached 0.8830, which is
not competitive with the best baseline at 0.9340, but it is a meaningful
classifier from a single prompt over obfuscated numeric features.

Mushroom and chess are poor in the current obfuscated representation. Mushroom
reached only 0.5755, and chess reached 0.5775 after all 8 retries. These results
suggest that random categorical codes like `c0`, `c1`, and `x13` give the LLM
too little semantic structure, especially for high-cardinality categorical
tasks.

## Next Decision

Do not spend a full 5-trial LLM budget on mushroom or chess in the current
obfuscated format. The better next engineering step is to add semantic/named
feature representations for non-CDC tabular datasets, then rerun the pilot.

Reasonable immediate experiments:

1. Run the full 5-trial matched CDC semantic result.
2. Run a 5-trial HTRU2 current-format result if we want a numeric non-CDC
   comparator.
3. Add semantic representations for mushroom and chess before spending more LLM
   calls there.
4. After semantic non-CDC prompts exist, rerun this pilot and only then decide
   whether to run 5-trial aggregates.
