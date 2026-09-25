"""Launcher and container-side runner for Modal training runs; no ``modal`` import.

``cloud/modal_app.py`` holds the Modal definitions and calls into this module on both
sides: locally to pack the code and merge the Hydra arguments, and in the container to
run ``train.py`` the way ``cloud/startup_gpu.sh`` does on a GCE VM. Keeping it free of
``modal`` makes it unit-testable. It also runs under Python 3.10 in the reference image,
so it must avoid newer syntax.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

RESULTS_VOLUME = "srnn-results"
DATA_VOLUME = "srnn-data"
RESULTS_MOUNT = "/vol/results"
DATA_MOUNT = "/vol/data"
CODE_DIR = "/root/train-srnn"
TELEMETRY = Path(__file__).resolve().with_name("run_telemetry.py")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# =============================================================================
# Shared
# =============================================================================

def result_rel_dir(run_name: str, model: str, task: str, seed: int) -> str:
    """Path of one run inside the results Volume; the same layout as the GCS bucket."""
    return f"results-pytorch/{run_name}/{model}/{task}/seed{seed}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# Launch side (laptop)
# =============================================================================

def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def validate_commit(sha: str) -> str:
    sha = sha.strip().lower()
    if not SHA_RE.match(sha):
        raise ValueError("--commit must be a full 40-character Git SHA")
    return sha


def dirty_paths(repo: Path) -> list[str]:
    """Modified, staged, or deleted tracked files; untracked files are ignored."""
    out = git(repo, "status", "--porcelain", "--untracked-files=no")
    return [line[3:] for line in out.splitlines() if line.strip()]


def untracked_python(repo: Path) -> list[str]:
    return git(repo, "ls-files", "--others", "--exclude-standard", "--", "*.py").split()


def tracked_files(repo: Path) -> list[str]:
    """Tracked paths that exist in the working tree (a deleted tracked file is skipped)."""
    names = git(repo, "ls-files", "-z", "--cached").split("\0")
    return sorted(n for n in names if n and (repo / n).is_file())


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
    return info


def worktree_tarball(repo: Path) -> bytes:
    """gzip tar of the tracked files as they are on disk; untracked files never ship.

    Timestamps and ownership are zeroed so identical content gives an identical sha256.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in tracked_files(repo):
                tar.add(str(repo / name), arcname=name, recursive=False, filter=_normalized)
    return buf.getvalue()


def commit_tarball(repo: Path, sha: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz", sha],
                          check=True, capture_output=True).stdout


def experiment_env_text(repo: Path, task: str, commit: str | None) -> str | None:
    """``cloud/experiments/<task>.env`` from the code being shipped, or None if absent."""
    rel = f"cloud/experiments/{task}.env"
    if commit is None:
        path = repo / rel
        return path.read_text() if path.is_file() else None
    result = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                            capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def source_args(env_text: str | None) -> str:
    """``$ARGS`` after bash sources the file, exactly as ``submit.sh`` does."""
    if env_text is None:
        return ""
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write(env_text)
    try:
        return subprocess.run(["bash", "-c", 'source "$1"; printf %s "${ARGS:-}"', "_", f.name],
                              check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(f.name)


def supports_resume(repo: Path, commit: str | None) -> bool:
    """Whether the shipped code has ``resume=true`` (train_srnn/config.py ``resume: bool``)."""
    rel = "train_srnn/config.py"
    if commit is None:
        text = (repo / rel).read_text()
    else:
        text = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                              check=True, capture_output=True, text=True).stdout
    return re.search(r"^\s+resume: bool", text, re.MULTILINE) is not None


def merge_args(experiment_args: str, user_args: str) -> list[str]:
    """Experiment ARGS then the caller's args, split like ``startup_gpu.sh``.

    ``submit.sh`` strips single quotes from the caller's args and prepends ARGS;
    ``startup_gpu.sh`` expands them unquoted under ``set -f``, i.e. plain whitespace
    splitting with no quote removal and no globbing.
    """
    return f"{experiment_args} {user_args.replace(chr(39), '')}".split()


def build_code(repo: Path, commit: str | None = None, allow_dirty: bool = False) -> dict:
    """The code tarball plus the provenance that describes it.

    Worktree mode refuses uncommitted changes to tracked files unless ``allow_dirty``;
    commit mode ships ``git archive`` of that exact commit.
    """
    if commit:
        sha = validate_commit(commit)
        git(repo, "cat-file", "-e", f"{sha}^{{commit}}")
        tgz = commit_tarball(repo, sha)
        info = {"code_source": "commit", "commit": sha, "dirty": False, "dirty_paths": []}
    else:
        dirty = dirty_paths(repo)
        if dirty and not allow_dirty:
            raise RuntimeError("uncommitted changes to tracked files (commit them or pass "
                               "--allow-dirty):\n  " + "\n  ".join(dirty))
        tgz = worktree_tarball(repo)
        info = {"code_source": "worktree", "commit": git(repo, "rev-parse", "HEAD").strip(),
                "dirty": bool(dirty), "dirty_paths": dirty}
    info["code_sha256"] = sha256_bytes(tgz)
    info["code_bytes"] = len(tgz)
    return {"tgz": tgz, **info}


# =============================================================================
# Container side
# =============================================================================

def expand_placeholders(args: list[str], results_root: str = RESULTS_MOUNT,
                        data_root: str = DATA_MOUNT) -> list[str]:
    """``{results}`` and ``{data}`` in script arguments become the Volume mount points."""
    return [a.replace("{results}", results_root).replace("{data}", data_root) for a in args]


def extract_code(code_tgz: bytes, code_dir: str) -> None:
    shutil.rmtree(code_dir, ignore_errors=True)
    Path(code_dir).mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(code_tgz), mode="r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(code_dir, filter="data")
        else:
            tar.extractall(code_dir)


def run_script(spec: dict, code_tgz: bytes, *, commit_volume, results_root: str = RESULTS_MOUNT,
               data_root: str = DATA_MOUNT, code_dir: str = CODE_DIR) -> dict:
    """Run one repository script (analysis, evaluation) next to the Volumes.

    ``spec["script"]`` is a path inside the repository and ``spec["args"]`` its
    arguments, with ``{results}``/``{data}`` placeholders. Output goes to the log file
    ``spec["log"]`` (relative to the results Volume) and to stdout.
    """
    extract_code(code_tgz, code_dir)
    log_path = Path(results_root) / spec["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, spec["script"], *expand_placeholders(spec["args"], results_root, data_root)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=code_dir, SRNN_DATA_DIR=data_root,
               SRNN_HOME=os.environ.get("SRNN_HOME", "/tmp/srnn"))
    start = time.time()
    try:
        with log_path.open("a", buffering=1) as log:
            log.write(f"+ {' '.join(cmd)}\n# code {spec.get('code_source')} {spec.get('commit')}"
                      f" sha256 {spec.get('code_sha256')}\n")
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=code_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        _stream(proc, log_path)
        exit_code = proc.wait()
    finally:
        commit_volume()
    return {"exit_code": exit_code, "duration_seconds": round(time.time() - start), "log": spec["log"]}


class Terminated(Exception):
    """SIGTERM reached the runner (e.g. ``modal app stop``)."""


def train_command(python: str, *, model: str, task: str, seed: int, train_args: list[str],
                  run_name: str, output_dir: str) -> list[str]:
    return [python, "train.py", f"model={model}", f"task={task}", f"seed={seed}",
            *train_args, f"run_name={run_name}_seed{seed}", f"output_dir={output_dir}"]


def redelivery_guard(out: Path, overwrite: bool, resumable: bool = False) -> Path | None:
    """Decide what a populated output directory means for this attempt.

    Modal re-runs an input after a worker preemption (even with ``retries=0``) or a
    container crash. Code with ``resume=true`` simply continues the run, so the
    directory is kept (return None). For older code, record the attempt beside the
    files and return that note's path; the caller must then leave the directory alone,
    since a rerun would append to its histories. ``overwrite`` clears the directory
    (only on the first attempt; see ``run_training``).
    """
    if not out.is_dir() or not any(out.iterdir()):
        return None
    if overwrite:
        shutil.rmtree(out)
        return None
    if resumable:
        return None
    note = out / f"redelivery_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    note.write_text(json.dumps({"reason": "output directory already populated; not rerun",
                                "time": time.time()}, indent=2) + "\n")
    return note


ATTEMPTS = "attempts.jsonl"


def previous_attempts(out: Path) -> int:
    path = out / ATTEMPTS
    return len(path.read_text().splitlines()) if path.is_file() else 0


def record_attempt(out: Path, call_id: str) -> int:
    """Append this container start to ``attempts.jsonl``; returns the attempt number."""
    number = previous_attempts(out) + 1
    with (out / ATTEMPTS).open("a") as f:
        f.write(json.dumps({"attempt": number, "time": time.time(), "function_call_id": call_id,
                            "container_id": os.environ.get("MODAL_TASK_ID", "")}) + "\n")
    return number


def installed_packages() -> dict[str, str]:
    from importlib import metadata
    return {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}


def _stream(proc: subprocess.Popen, log_path: Path) -> None:
    with log_path.open("a", buffering=1) as log:
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)


def run_training(spec: dict, code_tgz: bytes, *, commit_volume, call_id: str,
                 results_root: str = RESULTS_MOUNT, data_root: str = DATA_MOUNT,
                 code_dir: str = CODE_DIR) -> dict:
    """Run one training job inside a Modal container and return ``run_metadata``.

    Mirrors ``startup_gpu.sh``: provenance, one-second GPU memory samples, ``train.py``
    with its output teed to ``training_log.txt``, then (always) a GPU summary,
    ``run_metadata.json`` and a Volume commit.
    """
    import run_telemetry

    start, start_iso = time.time(), run_telemetry.utc_now()
    out = Path(results_root) / result_rel_dir(spec["run_name"], spec["model"], spec["task"],
                                              spec["seed"])
    resumable = bool(spec.get("supports_resume", False))
    attempt = previous_attempts(out) + 1
    # --overwrite clears the directory once; a retried attempt must keep what it wrote.
    overwrite = spec.get("overwrite", False) and attempt == 1
    note = redelivery_guard(out, overwrite, resumable)
    if note is not None:
        commit_volume()
        raise RuntimeError(f"{out} already holds a run; recorded {note.name} and stopped")
    out.mkdir(parents=True, exist_ok=True)
    attempt = record_attempt(out, call_id)

    exit_code, sampler, proc, error = 1, None, None, None
    previous = None

    def on_term(signum, _frame):
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)
        raise Terminated(f"signal {signum}")

    try:
        previous = signal.signal(signal.SIGTERM, on_term)
    except ValueError:  # not the main thread; rely on the finally block alone
        previous = None

    try:
        (out / "code.tar.gz").write_bytes(code_tgz)
        (out / "code.sha256").write_text(f"{sha256_bytes(code_tgz)}  code.tar.gz\n")
        extract_code(code_tgz, code_dir)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the container")
        data_dir = Path(data_root) / spec.get("dataset", spec["task"])
        hash_dir = data_dir if data_dir.is_dir() else Path(tempfile.mkdtemp())
        extra = {key: spec[key] for key in ("image", "gpu", "code_source", "dirty",
                                            "dirty_paths", "code_sha256")}
        train_args = list(spec["train_args"]) + (["resume=true"] if resumable else [])
        cmd = train_command(sys.executable, model=spec["model"], task=spec["task"],
                            seed=spec["seed"], train_args=train_args,
                            run_name=spec["run_name"], output_dir=str(out))
        extra.update(launcher="modal", function_call_id=call_id,
                     container_id=os.environ.get("MODAL_TASK_ID", ""),
                     image_id=os.environ.get("MODAL_IMAGE_ID", ""),
                     train_argv=cmd[1:], packages=installed_packages())
        extra["attempt"] = attempt
        name = ("runtime_provenance.json" if attempt == 1
                else f"runtime_provenance_attempt{attempt}.json")
        run_telemetry.provenance(out, hash_dir, spec["commit"], extra, name=name)

        sampler = subprocess.Popen([sys.executable, str(TELEMETRY), "sample", "--output", str(out)])
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=code_dir,
                   SRNN_DATA_DIR=data_root, SRNN_HOME=os.environ.get("SRNN_HOME", "/tmp/srnn"))
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=code_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        _stream(proc, out / "training_log.txt")
        exit_code = proc.wait()
    except BaseException as exc:  # record every failure, including SIGTERM
        error = f"{type(exc).__name__}: {exc}"
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
            try:
                run_telemetry.summarize(out)
            except (OSError, KeyError, ValueError) as exc:
                print(f"GPU summary failed: {exc}", flush=True)
        extra = {"launcher": "modal", "image": spec["image"], "code_source": spec["code_source"],
                 "dirty": spec["dirty"], "code_sha256": spec["code_sha256"],
                 "attempts": previous_attempts(out)}
        if error:
            extra["error_message"] = error
        meta = run_telemetry.write_run_metadata(
            out, run_name=spec["run_name"], experiment=spec["task"], model=spec["model"],
            seed=spec["seed"], exit_code=exit_code if error is None else 1, vm_name=call_id,
            hardware=f"modal-{spec['gpu']}", start_time=start_iso,
            duration_seconds=time.time() - start, commit=spec["commit"],
            train_args=" ".join(spec["train_args"]), extra=extra)
        commit_volume()
    return meta
