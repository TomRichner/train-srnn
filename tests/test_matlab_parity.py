"""Optional live-reference tests: generate fixtures with the MATLAB MCP exporter."""
import os
from pathlib import Path

import pytest
import torch

from scripts.verify_matlab_parity import CONDITIONS, verify_fixture


@pytest.mark.parametrize("condition", CONDITIONS)
def test_matlab_equations_and_sra1(condition):
    directory=os.environ.get("SRNN_MATLAB_FIXTURES")
    if not directory:
        pytest.skip("Set SRNN_MATLAB_FIXTURES to MATLAB exporter output")
    fixture=Path(directory)/f"{condition}.mat"
    assert fixture.is_file(), f"Missing requested MATLAB fixture: {fixture}"
    verify_fixture(fixture,trajectories=False)


@pytest.mark.slow
def test_matlab_long_trajectories():
    directory=os.environ.get("SRNN_MATLAB_FIXTURES")
    if not directory:
        pytest.skip("Set SRNN_MATLAB_FIXTURES to MATLAB exporter output")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        verify_fixture(Path(directory)/"sfa3_std2.mat", trajectories=True)
    finally:
        torch.set_num_threads(previous)
