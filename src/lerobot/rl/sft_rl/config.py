from dataclasses import dataclass


@dataclass
class SFTReplayConfig:
    enable: bool = False
    contract_path: str | None = None
    selection_path: str | None = None
    selection_sha256: str | None = None
    export_validation_path: str | None = None
    high_fraction: float | None = None
    expected_global_batch: int | None = None
    expected_output_horizon: int | None = None
    expected_mixed_precision: str | None = None
    sft_checkpoint_sha256: str | None = None
    control_config_path: str | None = None
    control_config_sha256: str | None = None
    control_run_path: str | None = None
    control_run_sha256: str | None = None
    normalization_source: str | None = None
    normalization_stats_sha256: str | None = None

    def validate(self):
        if not self.enable:
            return
        for name in (
            "contract_path",
            "selection_path",
            "selection_sha256",
            "export_validation_path",
            "sft_checkpoint_sha256",
            "control_config_path",
            "control_config_sha256",
            "control_run_path",
            "control_run_sha256",
        ):
            if not getattr(self, name):
                raise ValueError(f"SFT+RL requires explicit {name}; do not guess the paired control")
        if self.high_fraction is None or not 0 < self.high_fraction < 1:
            raise ValueError("Specify the reviewed D_high sampling fraction, strictly between zero and one")
        if self.expected_global_batch is None or self.expected_global_batch < 1:
            raise ValueError("Specify the paired control global batch")
        if self.expected_output_horizon is None or self.expected_output_horizon < 1:
            raise ValueError("Specify policy output horizon independently of the 20-action retained window")
        if self.normalization_source not in {"checkpoint", "control_dataset"}:
            raise ValueError(
                "Explicitly choose the paired normalization source: checkpoint or control_dataset"
            )
        if self.expected_mixed_precision not in {"no", "bf16", "fp16"}:
            raise ValueError("Specify the paired Accelerate mixed precision setting")
        if self.normalization_source == "control_dataset" and not self.normalization_stats_sha256:
            raise ValueError("Pin the paired control normalization statistics")
