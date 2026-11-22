"""Headless Snake run for the rule-variant experiments (no visualiser windows).
Env: RULE_VARIANT (full|drift|fluct), SCORE_BETA, FREEZE_READOUT, MAX_EPISODES, LR,
BOARD_SIZE, VISIBLE_RANGE, SCALE, HID_E, HID_I, FAN_IN, CSV_PREFIX, REPETITION."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import snake
try:
    snake.train_snake_agent_with_ipc(episodes=int(os.environ.get("MAX_EPISODES", "20000")),
                                     metrics_q=None, best_run_q=None, random_run_q=None, sigma_q=None,
                                     compress_frames=True, compress_quality=80)
finally:
    snake.trace_logger.close()
