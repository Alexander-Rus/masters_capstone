from pathlib import Path

class Config:
    # === Paths ===
    base_dir = Path(__file__).parent.resolve()
    map_path = base_dir / "maps" / "map.npy"
    save_dir = base_dir / "save"

    # === Game Settings ===
    frame_width = 160
    frame_height = 120
    run_steps_per_action = 10
    max_rollout_time_ms = 2000

    # === Input Settings ===
    inputs = [
        {"accelerate": True, "brake": False, "left": False, "right": False},
        {"accelerate": True, "brake": False, "left": True, "right": False},
        {"accelerate": True, "brake": False, "left": False, "right": True},
        {"accelerate": False, "brake": True, "left": False, "right": False},
        {"accelerate": False, "brake": False, "left": True, "right": False},
        {"accelerate": False, "brake": False, "left": False, "right": True},
        {"accelerate": False, "brake": False, "left": False, "right": False},
        {"accelerate": True, "brake": True, "left": False, "right": False},
        {"accelerate": True, "brake": False, "left": True, "right": True},
    ]

    n_actions = len(inputs)
    n_quantiles = 16
    state_float_dim = 128
    hidden_dim = 256

    # === Training ===
    batch_size = 64
    iqn_embed_dim = 64
    learning_rate = 1e-4
    gamma = 0.99
    train_steps = 100000
    target_update_interval = 500
    save_interval = 10000
    updates_per_rollout = 1
    checkpoint_freq = 10000  # <-- Add this


    # === Buffer ===
    buffer_size = 100_000
    buffer_test_ratio = 0.1

    # === Others ===
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"


config = Config()
