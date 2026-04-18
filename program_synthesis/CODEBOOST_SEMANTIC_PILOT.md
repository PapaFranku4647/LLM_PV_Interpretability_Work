# Semantic CodeBoost Pilot

Run completed on 2026-04-17 around 23:12 local time.

This pilot reruns the non-CDC one-trial matched setup after adding
`--tabular-representation semantic`.

Configuration:

- train size: 256
- validation size: 256
- test size: 2000
- batch size: 256
- boost rounds: 1
- trials: 1 per dataset
- round retries: up to 8
- sampling: without replacement
- representation: semantic/named tabular rows
- acceptance: `max_weak_error=0.3025`, with best-valid fallback up to 0.499
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/semantic_codeboost_pilot_t1_b256_s1/`

## Results

| Function | Dataset | Semantic test acc | Prior obfuscated pilot | Best baseline | Gap to baseline | API attempts | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_n` | mushroom | 0.6380 | 0.5755 | 0.8528 | -0.2148 | 2 | $0.1665 |
| `fn_p` | HTRU2 | 0.8950 | 0.8830 | 0.9340 | -0.0390 | 1 | $0.0522 |
| `fn_q` | chess | 0.6310 | 0.5775 | 0.9615 | -0.3305 | 8 | $0.7695 |

Total estimated pilot cost: about $0.9882.

## Interpretation

Semantic representation helped mushroom and chess compared with their obfuscated
one-trial pilots, but not enough to justify 5-trial spending yet. Mushroom moved
from 0.5755 to 0.6380; chess moved from 0.5775 to 0.6310 but still needed all 8
retries. The UCI chess feature abbreviations are probably not semantic enough for
the model to reason well.

HTRU2 semantic reached 0.8950. That is better than the one-trial obfuscated pilot
at 0.8830, but below the 5-trial obfuscated mean of 0.9032 and below the best
baseline at 0.9340. For HTRU2, binning may be discarding useful threshold detail.

The next best experiment is not a full mushroom/chess run. The better next step
is to improve the representation further:

- For HTRU2, try a hybrid semantic row with named features plus raw normalized
  numeric values instead of bins only.
- For mushroom, reduce unknowns by keeping missing-value indicators distinct
  from true category values and possibly include both raw category codes and
  readable labels.
- For chess, UCI abbreviations are still opaque; either add real feature
  explanations from Shapiro's KRKPA7 description or deprioritize chess.
