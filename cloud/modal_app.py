"""Train on a Modal GPU: the Modal counterpart of ``cloud/submit.sh`` + ``startup_gpu.sh``.

    uv run modal run --detach cloud/modal_app.py --run-name R --task cheetah100 \\
        [--model srnn] [--seed 1] [--args "epochs=3 device=cuda"] [--gpu L4] \\
        [--image default|reference] [--commit <40-hex sha>] [--allow-dirty]
        [--overwrite | --resume] [--wait]

Always pass --detach: the launcher spawns the training call and returns (or, with --wait,
waits for it), and only a detached app keeps that call alive after the launcher exits.

The code is shipped as a tarball argument: the tracked files of the working tree (which
must be clean unless --allow-dirty), or ``git archive`` of --commit. Results go straight to
the ``srnn-results`` Volume under ``results-pytorch/<run>/<model>/<task>/seed<seed>/``;
datasets are read from ``srnn-data:/<task>/``. See docs/modal.md.
"""
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # modal_run and run_telemetry sit next to this file on both sides

import modal_run  # noqa: E402

REPO = HERE.parent
DEFAULT_GPU = "L4"
TIMEOUT_SECONDS = 24 * 3600  # Modal's maximum; only execution time is billed


def _with_launcher(image: modal.Image) -> modal.Image:
    return (image
            .env({"SRNN_HOME": "/tmp/srnn", "PYTHONUNBUFFERED": "1"})
            .add_local_file(HERE / "modal_run.py", "/root/modal_run.py")
            .add_local_file(HERE / "run_telemetry.py", "/root/run_telemetry.py"))


# The environment pinned by uv.lock.
default_image = _with_launcher(
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(REPO), frozen=True, extra_options="--no-dev")
)

# The GCE Deep Learning VM environment of regression-baseline-20ep-20260915
# (python_packages.txt): Python 3.10, torch 2.9.1+cu129. torch pins its nvidia-*-cu12
# wheels exactly; the provenance records every installed version for comparison.
reference_image = _with_launcher(
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install("torch==2.9.1", "triton==3.5.1",
                    index_url="https://download.pytorch.org/whl/cu129")
    .uv_pip_install("numpy==2.2.6", "scipy==1.15.3", "pandas==2.3.3", "hydra-core==1.3.5",
                    "omegaconf==2.3.1", "tqdm==4.70.0")
)

app = modal.App("train-srnn")
results_volume = modal.Volume.from_name(modal_run.RESULTS_VOLUME, create_if_missing=True)
data_volume = modal.Volume.from_name(modal_run.DATA_VOLUME, create_if_missing=True)
# Retries rerun an input after a container crash; with resume=true (see modal_run) the
# rerun continues from the newest checkpoint. Preempted inputs are rerun regardless.
RETRIES = modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0)
FUNCTION_OPTIONS = dict(
    gpu=DEFAULT_GPU, timeout=TIMEOUT_SECONDS, retries=RETRIES,
    volumes={modal_run.RESULTS_MOUNT: results_volume,
             modal_run.DATA_MOUNT: data_volume.read_only()},
)


def _run(spec: dict, code_tgz: bytes) -> dict:
    return modal_run.run_training(spec, code_tgz, commit_volume=results_volume.commit,
                                  call_id=modal.current_function_call_id() or "")


@app.function(image=default_image, **FUNCTION_OPTIONS)
def train(spec: dict, code_tgz: bytes) -> dict:
    return _run(spec, code_tgz)


@app.function(image=reference_image, **FUNCTION_OPTIONS)
def train_reference(spec: dict, code_tgz: bytes) -> dict:
    return _run(spec, code_tgz)


IMAGES = {"default": train, "reference": train_reference}



def _exists(volume: modal.Volume, path: str) -> bool:
    try:
        return bool(volume.listdir(path))
    except (modal.exception.NotFoundError, FileNotFoundError):
        return False


@app.local_entrypoint()
def main(run_name: str, task: str, model: str = "srnn", seed: int = 1, args: str = "",
         gpu: str = DEFAULT_GPU, image: str = "default", commit: str = "",
         allow_dirty: bool = False, overwrite: bool = False, resume: bool = False,
         wait: bool = False, dataset: str = ""):
    if image not in IMAGES:
        sys.exit(f"--image must be one of {sorted(IMAGES)}, got {image!r}")
    if resume and overwrite:
        sys.exit("--resume and --overwrite are mutually exclusive")
    rel = modal_run.result_rel_dir(run_name, model, task, seed)
    exists = _exists(results_volume, rel)
    if exists and not (overwrite or resume):
        sys.exit(f"{modal_run.RESULTS_VOLUME}:/{rel} already exists; choose a new run name, "
                 "pass --resume to continue it, or --overwrite to replace it")
    if resume and not exists:
        print(f"NOTE: nothing at {modal_run.RESULTS_VOLUME}:/{rel} yet; --resume starts fresh")
    if dataset and not _exists(data_volume, dataset):
        sys.exit(f"no dataset at {modal_run.DATA_VOLUME}:/{dataset}; upload it with "
                 f"uv run modal volume put {modal_run.DATA_VOLUME} <dir> /{dataset}")
    if not dataset and not _exists(data_volume, task):
        print(f"WARNING: no dataset at {modal_run.DATA_VOLUME}:/{task} "
              "(fine for data-free tasks such as synthetic)")

    try:
        code = modal_run.build_code(REPO, commit or None, allow_dirty)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    for name in modal_run.untracked_python(REPO):
        print(f"WARNING: untracked {name} is not shipped")
    resumable = modal_run.supports_resume(REPO, commit or None)
    if resume and not resumable:
        sys.exit("--resume needs code with resume=true (train_srnn/config.py); this commit predates it")
    env_text = modal_run.experiment_env_text(REPO, task, commit or None)
    train_args = modal_run.merge_args(modal_run.source_args(env_text), args)
    if dataset:
        train_args.append(f"task.dataset={dataset}")
    spec = {"run_name": run_name, "task": task, "model": model, "seed": seed,
            "dataset": dataset or task,
            "train_args": train_args, "gpu": gpu, "image": image, "overwrite": overwrite,
            "supports_resume": resumable,
            **{k: v for k, v in code.items() if k != "tgz"}}

    print(f"Code:     {code['code_source']} {code['commit']}"
          f"{' (dirty)' if code['dirty'] else ''}, {code['code_bytes'] / 1e6:.1f} MB")
    print(f"Args:     {' '.join(train_args)}{' resume=true' if resumable else ''}")
    print(f"Results:  {modal_run.RESULTS_VOLUME}:/{rel}/")
    print(f"Logs:     uv run modal app logs {app.app_id}")
    print(f"Download: uv run modal volume get {modal_run.RESULTS_VOLUME} {rel} <local dir>")

    fn = IMAGES[image]
    if gpu != DEFAULT_GPU:
        fn = fn.with_options(gpu=gpu)
    # spawn, not remote: an awaited .remote() input is cancelled when this local process is
    # killed, even under --detach. A spawned call on a detached app runs to completion.
    call = fn.spawn(spec, code["tgz"])
    print(f"Call:     {call.object_id} (run_metadata.json appears when it ends)")
    if not wait:
        return
    meta = call.get()
    print(f"Finished: exit_code={meta['exit_code']} duration={meta['duration_seconds']} s")
    if meta["exit_code"] != 0:
        sys.exit(1)
