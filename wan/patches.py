"""CLI args for Wan integration.

The Wan port follows the DiT subproject pattern: no Megatron core files are
modified. The training entry point registers these extra args and uses
Megatron's NullTokenizer.
"""


def wan_extra_args(parser):
    """Add Wan-specific CLI arguments."""
    group = parser.add_argument_group(title="wan-model")
    group.add_argument(
        "--wan-preset",
        type=str,
        default="t2v-1.3b",
        choices=["tiny", "t2v-1.3b", "t2v-14b", "ti2v-5b"],
        help="Wan architecture preset. tiny is for CPU/GPU smoke tests.",
    )
    group.add_argument("--wan-dim", type=int, default=0)
    group.add_argument("--wan-in-dim", type=int, default=0)
    group.add_argument("--wan-out-dim", type=int, default=0)
    group.add_argument("--wan-text-dim", type=int, default=0)
    group.add_argument("--wan-ffn-dim", type=int, default=0)
    group.add_argument("--wan-freq-dim", type=int, default=0)
    group.add_argument("--wan-num-heads", type=int, default=0)
    group.add_argument("--wan-num-layers", type=int, default=0)
    group.add_argument("--wan-patch-size", type=str, default="")
    group.add_argument("--wan-eps", type=float, default=1e-6)
    group.add_argument(
        "--wan-has-image-input",
        action="store_true",
        help="Enable Wan I2V image conditioning modules. T2V leaves this off.",
    )
    group.add_argument(
        "--wan-seperated-timestep",
        action="store_true",
        help="Enable DiffSynth Wan2.2 TI2V per-token timestep conditioning.",
    )
    group.add_argument(
        "--wan-fuse-vae-embedding-in-latents",
        action="store_true",
        help="Expect first-frame VAE latents to be fused into the denoising latents.",
    )
    group.add_argument(
        "--wan-load-official-ckpt",
        type=str,
        default=None,
        help="Path to official/DiffSynth Wan checkpoint file or directory.",
    )
    group.add_argument(
        "--wan-strict-load",
        action="store_true",
        help="Require exact checkpoint key match when loading official weights.",
    )
    group.add_argument(
        "--wan-attention-backend",
        type=str,
        default="te",
        choices=["te", "sdpa"],
        help=(
            "Attention implementation for Wan blocks. 'te' uses Megatron Core's "
            "TransformerEngine DotProductAttention with flash attention and CP p2p ring; "
            "'sdpa' is the local PyTorch SDPA reference path."
        ),
    )
    group.add_argument(
        "--wan-local-qkv",
        action="store_true",
        help=(
            "Keep Wan Q/K/V activations tensor-parallel local and use TP-aware "
            "full-hidden RMSNorm before TE attention. This removes QKV output "
            "all-gathers while preserving the official Q/K RMSNorm math."
        ),
    )

    group = parser.add_argument_group(title="wan-flow-match")
    group.add_argument("--wan-train-timesteps", type=int, default=1000)
    group.add_argument("--wan-sigma-shift", type=float, default=5.0)
    group.add_argument("--wan-noise-scale", type=float, default=1.0)
    group.add_argument("--wan-min-timestep-boundary", type=float, default=0.0)
    group.add_argument("--wan-max-timestep-boundary", type=float, default=1.0)
    group.add_argument(
        "--wan-disable-timestep-weight",
        action="store_true",
        help="Disable DiffSynth scheduler timestep weighting.",
    )
    group.add_argument(
        "--wan-gradient-checkpointing",
        action="store_true",
        help="Use torch activation checkpointing inside Wan DiT blocks.",
    )

    group = parser.add_argument_group(title="wan-data")
    group.add_argument(
        "--wan-sample-path",
        type=str,
        default=None,
        help="Single .pt sample for 1-sample overfit. Contains latents/context.",
    )
    group.add_argument(
        "--wan-data-path",
        type=str,
        default=None,
        help="JSONL manifest with latents/context paths for real training.",
    )
    group.add_argument(
        "--wan-shard-cache-size",
        type=int,
        default=2,
        help="Per-worker LRU cache size for packed Wan training shards.",
    )
    group.add_argument(
        "--wan-context-drop-prob",
        type=float,
        default=0.0,
        help="Classifier-free context dropout probability during training.",
    )
    return parser
