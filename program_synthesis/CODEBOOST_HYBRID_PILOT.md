# Hybrid CodeBoost Pilot

Run completed on 2026-04-18 local time.

This pilot tests `--tabular-representation hybrid` after the semantic
non-CDC pilot showed two different failure modes:

- HTRU2 semantic bins may discard useful numeric threshold detail.
- Mushroom semantic categories may hide missingness and original categorical
  codes that classical models can exploit.

Configuration:

- train size: 256
- validation size: 256
- test size: 2000
- batch size: 256
- boost rounds: 1
- trials: 1 per dataset
- round retries: up to 8
- sampling: without replacement
- representation: hybrid/named tabular rows
- acceptance: `max_weak_error=0.3025`, with best-valid fallback up to 0.499
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/hybrid_codeboost_pilot_t1_b256_s1/`

Aggregate CSV:

- `program_synthesis/codeboost_hybrid_pilot_t1_b256_s1.csv`

## Results

| Function | Dataset | Hybrid test acc | Semantic pilot | Prior obfuscated pilot | Best baseline | Gap to baseline | API attempts | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_n` | mushroom | 0.6115 | 0.6380 | 0.5755 | 0.8528 | -0.2413 | 2 | $0.2828 |
| `fn_p` | HTRU2 | 0.8770 | 0.8950 | 0.8830 | 0.9340 | -0.0570 | 1 | $0.1016 |

Total estimated pilot cost: about $0.3844.

Chess hybrid was not run. The previous semantic chess pilot needed all 8 retries
and still only reached 0.6310 against a 0.9615 baseline. The hybrid chess row
only adds UCI code tokens; it does not add the missing domain descriptions, so
it is unlikely to be a good use of budget.

## Interpretation

Hybrid is implemented and useful as an ablation, but this pilot did not improve
accuracy.

For HTRU2, adding z-scores did not help the one-shot learner. The accepted
program reached 0.8770 test accuracy, below both the semantic one-trial pilot
at 0.8950 and the obfuscated one-trial pilot at 0.8830. That suggests the issue
is not just lost numeric precision; the prompt may be too large or the generated
rule may overfit the 256-example batch.

For mushroom, hybrid preserved category codes and missingness but landed at
0.6115, below the semantic pilot at 0.6380 and still far from the best baseline
at 0.8528. The original code tokens did not compensate for the task complexity.

The main conclusion is that representation changes alone are not enough for
mushroom or HTRU2. The next serious accuracy lever should be sampler diversity:
select residual/mistake-focused but feature-diverse prompt batches so later
weak learners are less correlated. For chess, add real KRKPA7 feature
descriptions before spending more API budget.
