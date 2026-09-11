# Value architecture and fixed-horizon comparison

Only final-step models appear below. Train metrics use32 frames/episode; dense held-out metrics use128. Intermediate evaluations are in all_evaluations.csv.

| Architecture | Fixed steps | Train CE | Held-out CE | Train MAE | Held-out MAE | Held-out RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen encoders | 1500 | 3.0511 | 3.0768 | 0.033886 | 0.033571 | 0.044572 |
| Frozen language | 1500 | 1.9387 | 2.4662 | 0.006508 | 0.013921 | 0.021568 |
| Image + task + state | 1500 | 1.8594 | 2.5006 | 0.005986 | 0.013700 | 0.021107 |
| Image + task (no state) | 1500 | 1.8599 | 2.5113 | 0.006115 | 0.013674 | 0.020786 |
| Vision only | 1500 | 1.7786 | 2.4865 | 0.005973 | 0.013319 | 0.020267 |

## State reliance of the original step3000 model

| State intervention | Held-out CE | Held-out MAE |
| --- | ---: | ---: |
| none | 2.636288 | 0.01217783 |
| zero | 2.636206 | 0.01217860 |
| permute | 2.636034 | 0.01217796 |

Identical images and labels, altered state tokens only. Interventions can be out of distribution; this is reliance evidence, not proof that a retrained no-state model wins.

The10% held-out set is reused for selection (validation). Bootstrap intervals are conditional on this split and are not selection-adjusted. Samples within episodes are not independent. Time ordering/difference metrics use sparse frame gaps, not fixed50-frame Bellman residuals. A single-seed ranking is provisional; it cannot prove real robot success or a global optimum.
