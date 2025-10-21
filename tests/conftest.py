import os
import json
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "params.json")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

@pytest.fixture(scope="session")
def cfg():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def cfg_A(cfg):
    return cfg["settings"]["A"]

@pytest.fixture(scope="session")
def cfg_B(cfg):
    return cfg["settings"]["B"]

@pytest.fixture(scope="session")
def seeds():
    return dict(A=1, B=10001)
