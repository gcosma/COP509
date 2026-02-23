"""
Benchmark & serving infrastructure — DO NOT MODIFY
====================================================
This module contains the plumbing that powers Extension C (benchmarking)
and serves the frontend.  Students never need to edit this file.

The only touch-point is in exercise.py where you call:
    register_routes(app, run_eval_fn=run_evaluation)
"""

import json
import os
import pathlib
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse

# ── Benchmark framework (optional dependency) ──
try:
    import inspect_evals  # noqa: F401 — registers benchmark tasks

    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False

BENCHMARK_TASKS = {
    "arc_easy": "inspect_evals/arc_easy",
    "arc_challenge": "inspect_evals/arc_challenge",
    "gsm8k": "inspect_evals/gsm8k",
    "hellaswag": "inspect_evals/hellaswag",
}

benchmark_status: dict = {
    "running": False,
    "result": None,
    "error": None,
    "phase": None,
    "total": 0,
    "completed": 0,
    "log_dir": None,
    "samples": None,
}

BENCHMARK_RUNS_DIR = pathlib.Path(__file__).parent / "benchmark_runs"


# ── Helpers ──────────────────────────────────────


def _read_completed_samples(log_dir_path):
    """Read completed samples from inspect_ai eval log (ZIP archive)."""
    from zipfile import ZipFile

    samples = []
    try:
        for f in pathlib.Path(log_dir_path).rglob("*.eval"):
            with ZipFile(f, "r") as zf:
                names = sorted(
                    set(
                        n
                        for n in zf.namelist()
                        if n.startswith("samples/") and n.endswith(".json")
                    )
                )
                for name in names:
                    raw = json.loads(zf.read(name))
                    scores = raw.get("scores", {})
                    first_scorer = next(iter(scores.values()), {})
                    samples.append(
                        {
                            "id": raw.get("id"),
                            "input": raw.get("input", ""),
                            "choices": raw.get("choices", []),
                            "target": raw.get("target", ""),
                            "model_answer": first_scorer.get("answer", ""),
                            "is_correct": first_scorer.get("value") == "C",
                        }
                    )
    except Exception:
        pass
    return samples


def _save_benchmark_run(bench_key, model, limit, accuracy, elapsed, samples):
    """Persist a completed benchmark run to disk."""
    os.makedirs(BENCHMARK_RUNS_DIR, exist_ok=True)
    run_id = str(uuid.uuid4())
    bench_label = bench_key.replace("_", " ").title()
    data = {
        "id": run_id,
        "benchmark": bench_key,
        "benchmark_label": bench_label,
        "model": model,
        "limit": limit,
        "accuracy": accuracy,
        "elapsed": elapsed,
        "samples": samples,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(BENCHMARK_RUNS_DIR / f"{run_id}.json", "w") as f:
        json.dump(data, f)
    return run_id


# ── Route registration ───────────────────────────


def register_routes(app, run_eval_fn=None):
    """
    Register benchmark endpoints and the frontend route onto *app*.

    Parameters
    ----------
    app : FastAPI
        The application instance from exercise.py.
    run_eval_fn : callable, optional
        ``run_eval_fn(task_name, model, limit, log_dir) -> (logs, accuracy)``
        Provided by the student's TODO 7 implementation.  When *None* the
        benchmark worker will raise an error explaining the TODO is incomplete.
    """

    # ── Benchmark list ──

    @app.get("/benchmarks")
    async def list_benchmarks():
        if not BENCHMARKS_AVAILABLE:
            return JSONResponse(
                {"error": "inspect_ai not installed"}, status_code=501
            )
        return {
            "benchmarks": {
                k: k.replace("_", " ").title() for k in BENCHMARK_TASKS
            },
            "limits": [5, 10, 25, 50, 100],
        }

    # ── Run a benchmark ──

    @app.post("/benchmarks/run")
    async def run_benchmark(request: Request):
        if not BENCHMARKS_AVAILABLE:
            return JSONResponse(
                {"error": "inspect_ai not installed"}, status_code=501
            )
        if benchmark_status["running"]:
            return JSONResponse(
                {"error": "A benchmark is already running"}, status_code=409
            )

        body = await request.json()
        bench_key = body.get("benchmark")
        model = body.get("model")
        limit = body.get("limit", 5)

        if bench_key not in BENCHMARK_TASKS:
            return JSONResponse(
                {"error": f"Unknown benchmark: {bench_key}"}, status_code=400
            )

        task_name = BENCHMARK_TASKS[bench_key]
        log_dir = tempfile.mkdtemp(prefix="inspect_bench_")
        benchmark_status.update(
            {
                "running": True,
                "result": None,
                "error": None,
                "phase": "setup",
                "total": limit,
                "completed": 0,
                "log_dir": log_dir,
                "samples": None,
            }
        )

        def worker():
            try:
                start = time.time()
                benchmark_status["phase"] = "evaluating"

                if run_eval_fn is None:
                    raise RuntimeError(
                        "TODO 7 is not implemented yet — "
                        "fill in run_evaluation() in exercise.py"
                    )

                logs, accuracy = run_eval_fn(task_name, model, limit, log_dir)

                elapsed = round(time.time() - start, 1)
                final_samples = _read_completed_samples(log_dir)
                run_id = _save_benchmark_run(
                    bench_key, model, limit, accuracy, elapsed, final_samples
                )
                benchmark_status.update(
                    {
                        "running": False,
                        "phase": None,
                        "log_dir": None,
                        "completed": limit,
                        "samples": final_samples,
                        "result": {
                            "accuracy": accuracy,
                            "benchmark": bench_key,
                            "model": model,
                            "limit": limit,
                            "elapsed": elapsed,
                            "run_id": run_id,
                        },
                    }
                )
            except Exception as e:
                benchmark_status.update(
                    {
                        "running": False,
                        "phase": None,
                        "log_dir": None,
                        "completed": 0,
                        "error": str(e),
                    }
                )

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return {"status": "started"}

    # ── Poll benchmark status ──

    @app.get("/benchmarks/status")
    async def benchmark_poll():
        status = dict(benchmark_status)
        if (
            status["running"]
            and status["phase"] == "evaluating"
            and status.get("log_dir")
        ):
            status["samples"] = _read_completed_samples(status["log_dir"])
            status["completed"] = len(status["samples"])
        status.pop("log_dir", None)
        return status

    # ── Past benchmark runs ──

    @app.get("/benchmark-runs")
    async def list_benchmark_runs():
        if not BENCHMARK_RUNS_DIR.exists():
            return []
        runs = []
        for filepath in BENCHMARK_RUNS_DIR.glob("*.json"):
            with open(filepath) as f:
                data = json.load(f)
            runs.append(
                {
                    "id": data["id"],
                    "benchmark": data.get("benchmark", ""),
                    "benchmark_label": data.get("benchmark_label", ""),
                    "model": data.get("model", ""),
                    "limit": data.get("limit", 0),
                    "accuracy": data.get("accuracy", 0),
                    "created_at": data.get("created_at", ""),
                }
            )
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs

    @app.get("/benchmark-runs/{run_id}")
    async def load_benchmark_run(run_id: str):
        filepath = BENCHMARK_RUNS_DIR / f"{run_id}.json"
        if not filepath.exists():
            return JSONResponse({"error": "Not found"}, status_code=404)
        with open(filepath) as f:
            return json.load(f)

    # ── Serve the frontend ──

    @app.get("/")
    async def serve_frontend():
        return FileResponse("index.html")
