"""Modal launcher helpers: code packing, argument merging, guards and metadata."""
import ast
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
import modal_run  # noqa: E402


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    (repo / "gone.py").write_text("G = 1\n")
    (repo / "cloud" / "experiments").mkdir(parents=True)
    (repo / "cloud" / "experiments" / "demo.env").write_text('# comment\nARGS="epochs=60"\n')
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "init")
    return repo


def _names(tgz):
    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tar:
        return sorted(m.name for m in tar.getmembers() if m.isfile())


def test_worktree_ships_tracked_files_only(repo):
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "scratch.py").write_text("x = 1\n")
    code = modal_run.build_code(repo)
    assert _names(code["tgz"]) == ["cloud/experiments/demo.env", "gone.py", "pkg/a.py"]
    assert code["code_source"] == "worktree" and code["dirty"] is False
    assert code["commit"] == subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    assert code["code_sha256"] == modal_run.sha256_bytes(code["tgz"])
    assert modal_run.untracked_python(repo) == ["scratch.py"]
    os.utime(repo / "pkg" / "a.py", (1, 1))
    assert modal_run.build_code(repo)["code_sha256"] == code["code_sha256"]


def test_dirty_tree_is_refused_unless_allowed(repo):
    (repo / "pkg" / "a.py").write_text("A = 2\n")
    (repo / "gone.py").unlink()
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        modal_run.build_code(repo)
    code = modal_run.build_code(repo, allow_dirty=True)
    assert code["dirty"] is True and sorted(code["dirty_paths"]) == ["gone.py", "pkg/a.py"]
    assert _names(code["tgz"]) == ["cloud/experiments/demo.env", "pkg/a.py"]
    with tarfile.open(fileobj=io.BytesIO(code["tgz"]), mode="r:gz") as tar:
        assert tar.extractfile("pkg/a.py").read() == b"A = 2\n"


def test_commit_mode_ships_the_commit_not_the_worktree(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (repo / "pkg" / "a.py").write_text("A = 2\n")
    code = modal_run.build_code(repo, commit=sha.upper())
    assert code["code_source"] == "commit" and code["commit"] == sha and not code["dirty"]
    with tarfile.open(fileobj=io.BytesIO(code["tgz"]), mode="r:gz") as tar:
        assert tar.extractfile("pkg/a.py").read() == b"A = 1\n"
    for bad in ("abc", "g" * 40, sha[:39]):
        with pytest.raises(ValueError, match="40-character"):
            modal_run.build_code(repo, commit=bad)
    with pytest.raises(subprocess.CalledProcessError):
        modal_run.build_code(repo, commit="0" * 40)


def test_experiment_args_come_from_the_shipped_code(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (repo / "cloud" / "experiments" / "demo.env").write_text('ARGS="epochs=5"\n')
    assert modal_run.source_args(modal_run.experiment_env_text(repo, "demo", None)) == "epochs=5"
    assert modal_run.source_args(modal_run.experiment_env_text(repo, "demo", sha)) == "epochs=60"
    assert modal_run.experiment_env_text(repo, "missing", None) is None
    assert modal_run.source_args(None) == ""


def _bash_split(experiment_args, user_args):
    """What submit.sh + startup_gpu.sh do: strip ', prepend ARGS, expand under set -f."""
    script = ('EXTRA="${2//\\\'/}"; ALL="$1 $EXTRA"; set -f; '
              'for w in $ALL; do printf "%s\\n" "$w"; done')
    out = subprocess.run(["bash", "-c", script, "_", experiment_args, user_args],
                         check=True, capture_output=True, text=True).stdout
    return out.splitlines()


@pytest.mark.parametrize("user_args", [
    "epochs=3 device=cuda",
    "'model.variants=[srnn-no-adapt,srnn-sfa1-std1]' model.variant_seeds=[1,2,3]",
    '  lr=0.001\tgrad_clip=1  task.name="x y" ',
    "",
])
def test_merge_args_matches_bash_word_splitting(user_args):
    experiment = modal_run.source_args((ROOT / "cloud/experiments/cheetah100.env").read_text())
    assert modal_run.merge_args(experiment, user_args) == _bash_split(experiment, user_args)


def test_bracket_lists_pass_through_intact():
    args = modal_run.merge_args("epochs=60", "model.variants=[a,b] 'x=[1,2]'")
    assert args == ["epochs=60", "model.variants=[a,b]", "x=[1,2]"]


def test_train_command_and_result_layout():
    cmd = modal_run.train_command("py", model="srnn", task="cheetah100", seed=2,
                                  train_args=["epochs=3"], run_name="r", output_dir="/o")
    assert cmd == ["py", "train.py", "model=srnn", "task=cheetah100", "seed=2", "epochs=3",
                   "run_name=r_seed2", "output_dir=/o"]
    assert modal_run.result_rel_dir("r", "srnn", "cheetah100", 2) == \
        "results-pytorch/r/srnn/cheetah100/seed2"


def test_redelivery_guard_preserves_a_populated_run(tmp_path):
    out = tmp_path / "seed1"
    assert modal_run.redelivery_guard(out, overwrite=False) is None
    out.mkdir()
    assert modal_run.redelivery_guard(out, overwrite=False) is None
    (out / "training_history.csv").write_text("epoch\n0\n")
    note = modal_run.redelivery_guard(out, overwrite=False)
    assert note.parent == out and note.name.startswith("redelivery_")
    assert (out / "training_history.csv").read_text() == "epoch\n0\n"
    assert modal_run.redelivery_guard(out, overwrite=True) is None and not out.exists()


def test_run_training_redelivery_raises_without_touching_the_run(tmp_path):
    out = tmp_path / modal_run.result_rel_dir("r", "srnn", "synthetic", 1)
    out.mkdir(parents=True)
    (out / "last.pt").write_bytes(b"x")
    commits = []
    spec = {"run_name": "r", "model": "srnn", "task": "synthetic", "seed": 1}
    with pytest.raises(RuntimeError, match="already holds a run"):
        modal_run.run_training(spec, b"", commit_volume=lambda: commits.append(1),
                               call_id="fc", results_root=str(tmp_path))
    assert commits == [1] and (out / "last.pt").read_bytes() == b"x"
    assert sorted(p.name for p in out.iterdir())[0] == "last.pt"


@pytest.mark.parametrize("name", ["modal_app.py", "modal_run.py", "run_telemetry.py"])
def test_container_side_files_parse_as_python_310(name):
    ast.parse((ROOT / "cloud" / name).read_text(), feature_version=(3, 10))


def test_modal_app_function_settings():
    pytest.importorskip("modal")
    import modal_app

    assert modal_app.FUNCTION_OPTIONS["timeout"] == 24 * 3600
    assert modal_app.FUNCTION_OPTIONS["retries"].max_retries == 2
    assert modal_app.FUNCTION_OPTIONS["gpu"] == "L4"
    assert set(modal_app.FUNCTION_OPTIONS["volumes"]) == {modal_run.RESULTS_MOUNT,
                                                           modal_run.DATA_MOUNT}
    assert set(modal_app.IMAGES) == {"default", "reference"}



class _FakeVolume:
    """listdir/read_file stand-in for modal.Volume."""

    def __init__(self, files):
        self.files = files

    def listdir(self, prefix, recursive=False):
        from types import SimpleNamespace as NS
        entries = [NS(path=p, type=NS(name="FILE")) for p in self.files if p.startswith(prefix + "/")]
        if not entries:
            raise RuntimeError("NotFoundError")
        return [NS(path=prefix + "/.hydra", type=NS(name="DIRECTORY")), *entries]

    def read_file(self, path):
        data = self.files[path]
        yield data[:2]
        yield data[2:]


def test_download_from_volume_keeps_subdirs_and_skips_cached(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from _runs import download_from_volume

    prefix = "results-pytorch/r/srnn/t/seed1"
    vol = _FakeVolume({f"{prefix}/last.pt": b"weights", f"{prefix}/.hydra/config.yaml": b"a: 1",
                       "results-pytorch/r/srnn/t/seed10/last.pt": b"other seed"})
    assert download_from_volume(vol, prefix, tmp_path) == (2, 0)
    assert (tmp_path / "last.pt").read_bytes() == b"weights"
    assert (tmp_path / ".hydra" / "config.yaml").read_bytes() == b"a: 1"
    assert not list(tmp_path.rglob("*.part"))
    assert download_from_volume(vol, prefix, tmp_path) == (0, 2)
    with pytest.raises(FileNotFoundError):
        download_from_volume(vol, "results-pytorch/missing", tmp_path)


def test_supports_resume_reads_the_shipped_config(repo):
    config = repo / "train_srnn" / "config.py"
    config.parent.mkdir()
    config.write_text("class TrainConfig:\n    init_ckpt: str = None\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "old")
    old = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    config.write_text("class TrainConfig:\n    resume: bool = False\n")
    assert modal_run.supports_resume(repo, None)
    assert not modal_run.supports_resume(repo, old)


def test_resumable_redelivery_keeps_the_run_and_counts_attempts(tmp_path):
    out = tmp_path / "seed1"
    out.mkdir()
    (out / "epoch_004.pt").write_bytes(b"x")
    assert modal_run.redelivery_guard(out, overwrite=False, resumable=True) is None
    assert (out / "epoch_004.pt").exists()
    assert modal_run.previous_attempts(out) == 0
    assert modal_run.record_attempt(out, "fc-1") == 1
    assert modal_run.record_attempt(out, "fc-1") == 2
    assert [json.loads(l)["attempt"] for l in (out / "attempts.jsonl").read_text().splitlines()] == [1, 2]
