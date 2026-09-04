# Official Evo-RL value and advantage contract

## Scope

The first run follows the Evo-RL official mainline at commit `6f2db449a21e1bac750b996f2e27cac6739aa63f`. It does **not** use the proposed shaped IQL reward. Official Pi*0.6 value training constructs a normalized episode-outcome/time-to-go target directly from `episode_success`, episode length and frame index.

## Value target

For task maximum episode length `T_max`, current frame `t`, episode length `T`, and `c_fail_coef = 1.0`:

```text
remaining_steps = T - t - 1
c_fail = T_max * c_fail_coef
g = -remaining_steps
if episode_success is false:
    g = g - c_fail
value_target = clip(g / (T_max + c_fail), -1, 0)
```

The value head is distributional over 201 bins in `[-1, 0]`. This target estimates progress/outcome under the behavior represented in the dataset. Because 66 episodes were completed after human intervention, it must not be described as a calibrated autonomous-success probability.

## ACP inference

- `n_step = 50`
- `positive_ratio = 0.30` (initial official default; not claimed optimal)
- `force_intervention_positive = true`
- value field: `complementary_info.value_evorl_official_v1`
- advantage field: `complementary_info.advantage_evorl_official_v1`
- indicator field: `complementary_info.acp_indicator_evorl_official_v1`

Intervention frames are positive ACP examples by official default. Inferred fields are written only to a derived dataset view inside this workspace, never to the frozen clean dataset.

## Known imbalance and gate

Only seven explicit failure episodes exist, and they terminate early. Their frames are about 3% of the dataset while the official trainer samples frames uniformly. The official-default smoke run is accepted only if held-out autonomous failures rank below autonomous successes and value outputs do not collapse to a narrow constant range. Any episode-balanced or failure-weighted correction is a separately named engineering variant, not the official-default result.

## Deferred IQL/AWR contract

The later extension may use sparse and shaped transition rewards, double Q, expectile V and continuous advantage weights on the original π0.5 flow-matching loss. Those rewards and critic masks are not mixed into the official Pi*0.6/ACP baseline.
