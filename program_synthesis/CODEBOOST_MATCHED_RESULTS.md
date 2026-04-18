# Matched CodeBoost Results

Run completed on 2026-04-17 around 22:36 local time.

This file records the full follow-up runs after the one-trial pilot. We only
spent the 5-trial budget on the datasets that looked worth continuing:

- CDC diabetes with semantic/named-feature prompts.
- HTRU2 with the current obfuscated numeric prompts.

Mushroom and chess were not expanded beyond the pilot because their current
obfuscated prompts performed poorly.

Configuration:

- train size: 256
- validation size: 256
- test size: 2000
- batch size: 256
- boost rounds: 1
- trials: 5
- round retries: up to 8
- sampling: without replacement
- acceptance: `max_weak_error=0.3025`, with best-valid fallback up to 0.499
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/matched_codeboost_t1_b256_s5/`

## Aggregate Results

| Function | Dataset | Representation | Mean test acc | Test std | Min | Max | Best baseline | Gap | API attempts | Total cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_o` | CDC diabetes | semantic | 0.7173 | 0.0102 | 0.7020 | 0.7300 | 0.7225 | -0.0052 | 6 | $0.4465 |
| `fn_p` | HTRU2 | obfuscated | 0.9032 | 0.0204 | 0.8795 | 0.9300 | 0.9340 | -0.0308 | 5 | $0.7939 |

## Per-Trial Results

| Function | Trial | Train acc | Val acc | Test acc | API attempts | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_o` | 1 | 0.7148 | 0.7148 | 0.7300 | 2 | $0.1690 |
| `fn_o` | 2 | 0.7305 | 0.7188 | 0.7020 | 1 | $0.0696 |
| `fn_o` | 3 | 0.7070 | 0.7109 | 0.7215 | 1 | $0.0643 |
| `fn_o` | 4 | 0.7227 | 0.6992 | 0.7165 | 1 | $0.0766 |
| `fn_o` | 5 | 0.7070 | 0.7031 | 0.7165 | 1 | $0.0671 |
| `fn_p` | 1 | 0.9375 | 0.9375 | 0.9170 | 1 | $0.1651 |
| `fn_p` | 2 | 0.9258 | 0.9141 | 0.9000 | 1 | $0.1211 |
| `fn_p` | 3 | 0.9297 | 0.9102 | 0.8895 | 1 | $0.1464 |
| `fn_p` | 4 | 0.9453 | 0.9609 | 0.9300 | 1 | $0.2113 |
| `fn_p` | 5 | 0.8906 | 0.9219 | 0.8795 | 1 | $0.1501 |

## Interpretation

CDC is now clearly viable as the flagship result. The 5-trial semantic CodeBoost
mean is 0.7173, only 0.52 percentage points below the best matched baseline
of 0.7225, and one trial reached 0.7300. Given the interpretability and
executable-code angle, this is strong enough to justify more CDC work.

HTRU2 is a useful secondary result, but not a win yet. The mean is 0.9032 against
the best baseline at 0.9340. One trial reached 0.9300, which is near the baseline,
but the variance is larger and the completions are expensive. It may improve
with a domain/named-feature prompt, but the current obfuscated numeric setup is
not enough for a headline claim.

The next technical step should not be blind multi-round boosting on the current
obfuscated datasets. The data says semantic representation matters. We should add
dataset context and named features for mushroom, HTRU2, and chess, then rerun the
one-trial pilot before spending 5-trial budgets.
