# PiperX saved-prediction diagnostics

Observed checkpoints: 250–5500; 56 holdout episodes, 32 fixed frames each.

This is post-hoc analysis, not a change to training selection. See JSON for limitations.

|Step|Train CE|Holdout CE|MAE|RMSE|MAE seconds|Spearman (time proxy)|Sampled delta MAE|
|---|---|---|---|---|---|---|---|
|1500|1.9896|2.4821|0.013357|0.020585|6.626|0.9801|0.007707|
|3000|1.6887|2.6365|0.012178|0.019837|6.041|0.9857|0.006403|
|4750|1.3536|3.0536|0.012097|0.019938|6.001|0.9865|0.005749|
|5500|1.2102|3.2633|0.012188|0.019628|6.046|0.9855|0.005757|

## Paired differences (newer minus older; negative is improvement)

- 1500 → 3000: MAE change -0.00117870, 95% CI [-0.002092415503430597, -0.00024815181631246245]; improved episodes 39/56.
- 1500 → 4750: MAE change -0.00126005, 95% CI [-0.0024053752959876614, -0.0001636725807478408]; improved episodes 37/56.
- 1500 → 5500: MAE change -0.00116944, 95% CI [-0.0022503417558726014, -0.00019180130395203158]; improved episodes 34/56.
- 3000 → 4750: MAE change -0.00008135, 95% CI [-0.0008192045147487598, 0.0006615994871838536]; improved episodes 25/56.
- 3000 → 5500: MAE change 0.00000926, 95% CI [-0.0007433493878839467, 0.000749417214808545]; improved episodes 24/56.
- 4750 → 5500: MAE change 0.00009061, 95% CI [-0.00023258028901896718, 0.0004183401261083678]; improved episodes 29/56.
