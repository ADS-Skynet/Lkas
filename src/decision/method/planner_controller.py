"""
Planner Decision Method

End-to-end neural network planner that replaces hand-written control logic.
Converts a BiSeNet segmentation mask into (steering, throttle) directly using
a trained PlannerModel.

The mask → build_lane_grid() → PlannerModel path mirrors planner_inference.py
but runs inside the LKAS decision server, consuming the mask already produced
by the detection server — no second BiSeNet pass required.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# ── Resolve planner-e2e directory ────────────────────────────────────────────
# planner_controller.py lives at  lkas/src/decision/method/
# planner-e2e lives at            ads-skynet/planner-e2e/
_PLANNER_E2E = Path(__file__).resolve().parents[4] / "planner-e2e"

if str(_PLANNER_E2E) not in sys.path:
    sys.path.insert(0, str(_PLANNER_E2E))

from planner_model import (  # noqa: E402
    PlannerModel,
    build_lane_grid,
    N_MAX_OBJECTS,
    OBJ_FEATURES,
    MAX_THROTTLE,
    SCENARIO_LANE_FOLLOW,
)

DEFAULT_MODEL_PATH: Path = _PLANNER_E2E / "planner_model.pth"


class PlannerDecisionMethod:
    """
    Drives the vehicle using a trained PlannerModel instead of a hand-tuned controller.

    Accepts the BiSeNet segmentation mask already produced by the detection server
    and converts it to (steering, throttle) via the spatial-grid → MLP pipeline.

    Object features are zero-padded because LKAS does not run YOLO.  The model
    handles all-zero object slots gracefully — it was trained with YOLO features
    but zero-padded slots are indistinguishable from "no objects in scene".
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        scenario: int = SCENARIO_LANE_FOLLOW,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path to planner_model.pth.
                        Defaults to ads-skynet/planner-e2e/planner_model.pth.
            scenario:   Scenario token fed to the embedding layer (0 = LANE_FOLLOW).
                        Fixed for the lifetime of this instance; can be updated via
                        set_scenario() at runtime.
            device:     Torch device string ("cpu" or "cuda:0").  Keep on CPU to
                        avoid contention with BiSeNet which runs on the GPU inside
                        the detection server.
        """
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"[PlannerDecisionMethod] Model not found: {model_path}\n"
                f"  Run train_planner.py inside planner-e2e/ first."
            )

        self.device = torch.device(device)
        self.scenario = scenario

        # ── Load model ────────────────────────────────────────────────────────
        self.model = PlannerModel().to(self.device)
        state = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[PlannerDecisionMethod] Loaded {model_path.name}  ({n_params:,} params)  device={self.device}")

        # ── Ego state ─────────────────────────────────────────────────────────
        # Warm-start: avoids the low-throttle feedback loop on the first frame
        # (same convention as planner_inference.py).
        self._prev_steering = 0.0
        self._prev_throttle = MAX_THROTTLE

        # Pre-built fixed tensors (recreated only when scenario changes)
        self._zero_objects = torch.zeros(
            1, N_MAX_OBJECTS * OBJ_FEATURES,
            dtype=torch.float32, device=self.device,
        )
        self._scenario_t = torch.tensor(
            [self.scenario], dtype=torch.long, device=self.device,
        )

    @torch.no_grad()
    def infer(
        self,
        mask: np.ndarray | None = None,
        lane_grid: list | None = None,
    ) -> tuple[float, float]:
        """
        Infer (steering, throttle) from lane features.

        Prefer the pre-computed lane_grid produced by the detection server.
        Falls back to computing the grid from the raw segmentation mask when
        lane_grid is not available (e.g. when planner-e2e is absent on detection side).

        Args:
            mask:      (H, W) uint8 segmentation mask (fallback if lane_grid is None)
            lane_grid: LANE_FEATURES floats pre-computed by DLLaneDetector

        Returns:
            (steering, throttle)
              steering  ∈ [-1.0,  1.0]
              throttle  ∈ [ 0.0,  MAX_THROTTLE]
        """
        if lane_grid is not None:
            lane_feats = lane_grid
        else:
            lane_feats = build_lane_grid(mask)
        ego_feats  = [self._prev_steering, self._prev_throttle / MAX_THROTTLE]

        lane_t = torch.tensor(lane_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        ego_t  = torch.tensor(ego_feats,  dtype=torch.float32, device=self.device).unsqueeze(0)

        out = self.model(self._zero_objects, lane_t, ego_t, self._scenario_t)  # (1, 2)

        steering = float(np.clip(out[0, 0].item(), -1.0, 1.0))
        throttle = float(np.clip(out[0, 1].item() * MAX_THROTTLE, 0.0, MAX_THROTTLE))

        self._prev_steering = steering
        self._prev_throttle = throttle

        return steering, throttle

    def set_scenario(self, scenario: int):
        """Update the scenario token at runtime."""
        self.scenario = scenario
        self._scenario_t = torch.tensor(
            [scenario], dtype=torch.long, device=self.device,
        )

    def reset_state(self):
        """Reset ego state to warm-start defaults."""
        self._prev_steering = 0.0
        self._prev_throttle = MAX_THROTTLE
