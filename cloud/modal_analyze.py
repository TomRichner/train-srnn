"""Run a repository script on a Modal GPU next to the Volumes (evaluation, analysis).

    uv run modal run --detach cloud/modal_analyze.py --script scripts/eval_speed.py \\
        --args "{results}/results-pytorch/<run>/srnn/cheetah100/seed1 --data-root {data} ..."

``{results}`` and ``{data}`` stand for the ``srnn-results`` and ``srnn-data`` mount points.
The code ships as in ``modal_app.py`` (tracked files of a clean tree, or --commit), the
image is the ``uv.lock`` one, and the log goes to
``srnn-results:/analysis_logs/<time>_<script>.log``. By default the launcher waits for the
result; the spawned call keeps running if the launcher is killed (pass --detach).
"""
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import modal_app as base  # noqa: E402  (images and Volumes)
import modal_run  # noqa: E402

app = modal.App("train-srnn-analyze")
# The container re-imports this file, which imports modal_app for the shared definitions.
image = base.default_image.add_local_file(HERE / "modal_app.py", "/root/modal_app.py")


@app.function(image=image, gpu=base.DEFAULT_GPU, timeout=6 * 3600, retries=0,
              volumes=base.FUNCTION_OPTIONS["volumes"])
def analyze(spec: dict, code_tgz: bytes) -> dict:
    return modal_run.run_script(spec, code_tgz, commit_volume=base.results_volume.commit)


@app.local_entrypoint()
def main(script: str, args: str = "", commit: str = "", allow_dirty: bool = False,
         wait: bool = True):
    try:
        code = modal_run.build_code(base.REPO, commit or None, allow_dirty)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    spec = {"script": script, "args": args.split(),
            "log": f"analysis_logs/{stamp}_{Path(script).stem}.log",
            **{k: v for k, v in code.items() if k != "tgz"}}
    call = analyze.spawn(spec, code["tgz"])
    print(f"Code:     {code['code_source']} {code['commit']}{' (dirty)' if code['dirty'] else ''}")
    print(f"Call:     {call.object_id}; log {modal_run.RESULTS_VOLUME}:/{spec['log']}")
    if wait:
        result = call.get()
        print(f"Finished: exit_code={result['exit_code']} duration={result['duration_seconds']} s")
        if result["exit_code"] != 0:
            sys.exit(1)
