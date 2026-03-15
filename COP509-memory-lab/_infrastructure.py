"""
COP509 Memory-Efficient Model Loading — Infrastructure
========================================================
DO NOT MODIFY this file.  It provides the GPT-2 model architecture,
memory-tracking utilities, and visualisation helpers used by the
exercise notebook.

Based on Sebastian Raschka's "Build a Large Language Model From Scratch"
(Apache 2.0 License).
"""

import gc
import os
import time
import threading

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# 1.  GPT-2 Model Architecture
# ──────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ──────────────────────────────────────────────────────────────
# 2.  Model Configurations
# ──────────────────────────────────────────────────────────────

_BASE = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

GPT_CONFIGS = {
    "124M": {**_BASE, "emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {**_BASE, "emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {**_BASE, "emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {**_BASE, "emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

MODEL_SIZES_MB = {
    "124M": "~500 MB",
    "355M": "~1.4 GB",
    "774M": "~3.0 GB",
    "1558M": "~6.4 GB",
}


# ──────────────────────────────────────────────────────────────
# 3.  Device Detection
# ──────────────────────────────────────────────────────────────

def get_device():
    """Return 'cuda' if an NVIDIA GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {name} ({total_gb:.1f} GB)")
        return "cuda"
    else:
        print("No GPU detected — running on CPU. All techniques still work!")
        return "cpu"


def has_gpu():
    return torch.cuda.is_available()


# ──────────────────────────────────────────────────────────────
# 4.  Memory Tracking
# ──────────────────────────────────────────────────────────────

import psutil


def _get_cpu_mb():
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


class MemoryTracker:
    """Collects peak CPU/GPU memory for each loading technique."""

    def __init__(self):
        self.results = {}

    def record(self, name, peak_cpu_mb, peak_gpu_mb):
        self.results[name] = {
            "peak_cpu_mb": round(peak_cpu_mb, 1),
            "peak_gpu_mb": round(peak_gpu_mb, 1),
        }

    def __repr__(self):
        lines = [f"{'Technique':<25} {'Peak CPU (MB)':>14} {'Peak GPU (MB)':>14}"]
        lines.append("-" * 55)
        for name, r in self.results.items():
            gpu_str = f"{r['peak_gpu_mb']:>10.1f} MB" if r["peak_gpu_mb"] > 0 else "       N/A"
            lines.append(f"{name:<25} {r['peak_cpu_mb']:>10.1f} MB {gpu_str}")
        return "\n".join(lines)


class track_memory:
    """
    Context manager that measures peak CPU and GPU memory.

    Usage:
        tracker = MemoryTracker()
        with track_memory(tracker, "1. Standard"):
            model = GPTModel(cfg)
            ...

    On exit it prints the results and records them in the tracker.
    """

    def __init__(self, tracker: MemoryTracker, technique_name: str):
        self.tracker = tracker
        self.name = technique_name
        self._cpu_samples = []
        self._done = False

    def __enter__(self):
        # Force garbage collection before measuring
        gc.collect()
        if has_gpu():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self._baseline_cpu = _get_cpu_mb()

        # Start background CPU memory monitor
        self._done = False
        self._cpu_samples = [_get_cpu_mb()]
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def _monitor(self):
        while not self._done:
            self._cpu_samples.append(_get_cpu_mb())
            time.sleep(0.05)

    def __exit__(self, *exc):
        self._done = True
        self._thread.join(timeout=2)
        self._cpu_samples.append(_get_cpu_mb())

        peak_cpu = max(self._cpu_samples) - self._baseline_cpu
        peak_gpu = 0.0
        if has_gpu():
            peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)

        self.tracker.record(self.name, peak_cpu, peak_gpu)

        # Print results immediately
        print(f"\n{'='*50}")
        print(f"  {self.name}")
        print(f"{'='*50}")
        print(f"  Peak CPU memory used:  {peak_cpu:>8.1f} MB")
        if has_gpu():
            print(f"  Peak GPU memory used:  {peak_gpu:>8.1f} MB")
        print(f"{'='*50}")

        return False


def cleanup():
    """Free memory between techniques."""
    gc.collect()
    if has_gpu():
        torch.cuda.empty_cache()
        time.sleep(1)
        torch.cuda.reset_peak_memory_stats()


# ──────────────────────────────────────────────────────────────
# 4b. Time Tracking
# ──────────────────────────────────────────────────────────────


class TimeTracker:
    """Collects loading time for each technique."""

    def __init__(self):
        self.results = {}

    def record(self, name, elapsed_seconds):
        self.results[name] = round(elapsed_seconds, 3)

    def __repr__(self):
        lines = [f"{'Technique':<25} {'Time (s)':>10}"]
        lines.append("-" * 37)
        for name, t in self.results.items():
            lines.append(f"{name:<25} {t:>8.3f} s")
        return "\n".join(lines)


class track_time:
    """
    Context manager that measures loading time for a technique.

    Usage:
        timer = TimeTracker()
        with track_time(timer, "1. Standard"):
            model = GPTModel(cfg)
            ...
    """

    def __init__(self, timer: TimeTracker, technique_name: str):
        self.timer = timer
        self.name = technique_name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        elapsed = time.time() - self._start
        self.timer.record(self.name, elapsed)

        print(f"  {self.name}: {elapsed:.3f} s")
        return False


def plot_timing_comparison(timer: TimeTracker):
    """Display a bar chart comparing loading times across techniques."""
    import matplotlib.pyplot as plt

    if not timer.results:
        print("No results to plot yet. Complete the TODO cells first!")
        return

    names = list(timer.results.keys())
    times = [timer.results[n] for n in names]

    # COP509 colour theme
    bg_color = "#150b22"
    text_color = "#f0edf5"
    bar_color = "#d4006a"
    grid_color = "#3a2858"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bars = ax.bar(range(len(names)), times, color=bar_color, edgecolor="none", width=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.02,
            f"{height:.3f}s", ha="center", va="bottom",
            color=text_color, fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Time (seconds)", color=text_color, fontsize=12)
    ax.set_title("Loading Time by Technique", color=text_color, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, color=text_color, fontsize=10)
    ax.tick_params(colors=text_color)
    ax.yaxis.grid(True, color=grid_color, alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(grid_color)

    plt.tight_layout()
    plt.show()


def print_timing_summary(timer: TimeTracker):
    """Print a formatted timing summary table."""
    print(timer)


# ──────────────────────────────────────────────────────────────
# 5.  Model Saving Helper
# ──────────────────────────────────────────────────────────────

def create_and_save_model(size="124M", filename=None):
    """
    Create a GPT-2 model with random weights and save it.

    This simulates having a trained model checkpoint.
    In the real world you would download pretrained weights — here
    we use random weights because the focus is on *loading* techniques,
    not on the model's actual predictions.
    """
    if filename is None:
        filename = f"gpt2-{size}.pth"

    if os.path.exists(filename):
        file_mb = os.path.getsize(filename) / (1024 ** 2)
        print(f"'{filename}' already exists ({file_mb:.0f} MB) — skipping.")
        return filename

    cfg = GPT_CONFIGS[size]
    print(f"Creating GPT-2 {size} model...")
    model = GPTModel(cfg)

    # Verify it works
    test_input = torch.tensor([[1, 2, 3]])
    model.eval()
    with torch.no_grad():
        model(test_input)

    print(f"Saving to '{filename}'...")
    torch.save(model.state_dict(), filename)

    file_mb = os.path.getsize(filename) / (1024 ** 2)
    print(f"Saved! File size: {file_mb:.0f} MB")

    # Clean up
    del model, test_input
    gc.collect()

    return filename


# ──────────────────────────────────────────────────────────────
# 6.  Visualisation
# ──────────────────────────────────────────────────────────────

def plot_memory_comparison(tracker: MemoryTracker):
    """
    Display a grouped bar chart comparing all recorded techniques.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    if not tracker.results:
        print("No results to plot yet. Complete the TODO cells first!")
        return

    names = list(tracker.results.keys())
    cpu_vals = [tracker.results[n]["peak_cpu_mb"] for n in names]
    gpu_vals = [tracker.results[n]["peak_gpu_mb"] for n in names]
    show_gpu = any(v > 0 for v in gpu_vals)

    # COP509 colour theme
    bg_color = "#150b22"
    text_color = "#f0edf5"
    cpu_color = "#d4006a"
    gpu_color = "#7b2d8e"
    grid_color = "#3a2858"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    x = range(len(names))
    bar_width = 0.35 if show_gpu else 0.5

    if show_gpu:
        bars_cpu = ax.bar(
            [i - bar_width / 2 for i in x], cpu_vals,
            bar_width, label="Peak CPU (MB)", color=cpu_color, edgecolor="none"
        )
        bars_gpu = ax.bar(
            [i + bar_width / 2 for i in x], gpu_vals,
            bar_width, label="Peak GPU (MB)", color=gpu_color, edgecolor="none"
        )
        # Value labels on GPU bars
        for bar in bars_gpu:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 20,
                    f"{height:.0f}", ha="center", va="bottom",
                    color=text_color, fontsize=9, fontweight="bold"
                )
    else:
        bars_cpu = ax.bar(x, cpu_vals, bar_width, label="Peak CPU (MB)", color=cpu_color, edgecolor="none")

    # Value labels on CPU bars
    for bar in bars_cpu:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 20,
                f"{height:.0f}", ha="center", va="bottom",
                color=text_color, fontsize=9, fontweight="bold"
            )

    ax.set_ylabel("Peak Memory (MB)", color=text_color, fontsize=12)
    ax.set_title("Memory Usage by Loading Technique", color=text_color, fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, color=text_color, fontsize=10)
    ax.tick_params(colors=text_color)
    ax.yaxis.grid(True, color=grid_color, alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(grid_color)

    legend = ax.legend(facecolor=bg_color, edgecolor=grid_color, fontsize=10)
    for text in legend.get_texts():
        text.set_color(text_color)

    plt.tight_layout()
    plt.show()


def print_memory_summary(tracker: MemoryTracker):
    """Print a formatted summary table."""
    print(tracker)


