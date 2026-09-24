"""Cloud telemetry summary uses the maximum observed whole-device memory."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_telemetry", Path(__file__).parents[1] / "cloud" / "run_telemetry.py"
)
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


def test_gpu_summary_keeps_devices_separate(tmp_path):
    (tmp_path / "gpu_memory_samples.csv").write_text(
        "time_unix,index,memory_used_mib,memory_total_mib,utilization_pct\n"
        "1,0,100,1000,50\n1,1,500,2000,75\n2,0,850,1000,90\n"
    )
    report = telemetry.summarize(tmp_path)
    assert report["devices"]["0"] == {
        "peak_used_mib": 850, "total_mib": 1000, "samples": 2, "peak_fraction": 0.85,
    }
    assert report["devices"]["1"]["peak_fraction"] == 0.25
    assert json.loads((tmp_path / "gpu_memory_summary.json").read_text()) == report


def test_no_samples_is_not_a_zero_memory_success(tmp_path):
    (tmp_path / "gpu_memory_samples.csv").write_text(
        "time_unix,index,memory_used_mib,memory_total_mib,utilization_pct\n"
    )
    assert telemetry.summarize(tmp_path)["devices"] == {}


def test_dispatch_commit_metadata_without_cloud_calls(tmp_path, monkeypatch):
    import os
    import subprocess

    root = Path(__file__).parents[1]
    fake = tmp_path / "gcloud"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['MOCK_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if 'list' in sys.argv and os.environ['MOCK_SUBMIT']=='1': print('zone-a')\n"
        "if 'describe' in sys.argv:\n"
        " if os.environ['MOCK_SUBMIT']=='1': print('TERMINATED')\n"
        " else: sys.exit(1)\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MOCK_LOG", str(log))
    sha = "abcd" * 10
    settings = (
        "GCP_PROJECT=project GCP_ZONE=zone GCP_BUCKET=gs://bucket "
        "MAX_CONCURRENT_VMS=1 DEFAULT_MACHINE_TYPE_FAMILY=g2 "
        "DEFAULT_MACHINE_TIER=standard-8 GCP_USE_SPOT=false GPU_COUNT=1 "
        "GPU_TYPE=nvidia-l4 GCP_IMAGE_FAMILY=family GCP_IMAGE_PROJECT=image-project "
        "BOOT_DISK_SIZE=100GB BOOT_DISK_TYPE=pd-balanced REPO_URL=https://example.invalid/repo"
    )
    for name, args, submit in [
        ("launch_run_gpu.sh", [f"--commit={sha}", "run", "synthetic", "srnn", "1"], "0"),
        ("submit.sh", ["vm", "run", "synthetic", "srnn", "1", f"--commit={sha}"], "1"),
    ]:
        script = tmp_path / name
        # Replace the source directive; never read the actual environment file.
        script.write_text((root / "cloud" / name).read_text().replace(
            'source "$SCRIPT_DIR/config.gpu.env"', settings
        ))
        monkeypatch.setenv("MOCK_SUBMIT", submit)
        subprocess.run(["bash", str(script), *args], check=True, capture_output=True, text=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(f"expected-commit={sha}" in arg for call in calls for arg in call)
        log.unlink()
        invalid = [arg.replace(sha, "not-a-sha") for arg in args]
        result = subprocess.run(["bash", str(script), *invalid], capture_output=True, text=True)
        assert result.returncode != 0
        assert "full 40-character Git SHA" in result.stderr
        assert not log.exists()


def test_dataset_hash_streams_larger_than_one_chunk(tmp_path):
    import hashlib

    payload = b"dataset\x00" * 200_000
    path = tmp_path / "train.npz"
    path.write_bytes(payload)
    assert telemetry.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_run_metadata_matches_gce_fields_and_marks_failure(tmp_path):
    common = dict(run_name="r", experiment="cheetah100", model="srnn", seed="2",
                  vm_name="fc-1", hardware="modal-L4", start_time="2026-01-01T00:00:00Z",
                  duration_seconds=12.4, commit="a" * 40, train_args="epochs=3")
    ok = telemetry.write_run_metadata(tmp_path, exit_code=0, extra={"platform": "modal"}, **common)
    assert json.loads((tmp_path / "run_metadata.json").read_text()) == ok
    assert {"run_name", "experiment", "model", "seed", "exit_code", "vm_name", "hardware",
            "start_time", "completed", "duration_seconds", "commit", "train_args"} <= set(ok)
    assert ok["seed"] == 2 and ok["duration_seconds"] == 12 and ok["platform"] == "modal"
    assert "error" not in ok and "failed_at" not in ok
    failed = telemetry.write_run_metadata(tmp_path, exit_code=3, **common)
    assert failed["error"] is True and failed["failed_at"] == failed["completed"]
