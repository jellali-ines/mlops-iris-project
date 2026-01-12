"""
Basic tests for CI/CD pipeline
"""
import pytest
from pathlib import Path

def test_models_exist():
    """Test that model files exist"""
    assert Path("models/model_v1.pkl").exists()
    assert Path("models/model_v2.pkl").exists()

def test_models_loadable():
    """Test that models can be loaded"""
    import pickle
    
    with open("models/model_v1.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None

def test_dockerfile_exists():
    """Test that Dockerfile exists"""
    assert Path("Dockerfile").exists()

def test_docker_compose_exists():
    """Test that docker-compose.yml exists"""
    assert Path("docker-compose.yml").exists()