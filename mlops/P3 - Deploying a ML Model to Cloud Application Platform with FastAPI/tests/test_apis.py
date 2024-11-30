import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from main import app
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    client = TestClient(app)
    return client


def test_get(client):
    """Test get the root of the website"""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {
        "message": "Welcome to the Project 3 of the Udacity MLOps course!"}


def test_post_below(client):
    """Test for salary below 50K"""

    res = client.post("/predict", json={
        "age": 37,
        "workclass": "State-gov",
        "fnlgt": 482927,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 65,
        "native-country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'Result': '<=50K'}


def test_post_above(client):
    """Test for salary above 50K"""
    res = client.post("/predict", json={
        "age": 62,
        "workclass": "State-gov",
        "fnlgt": 202056,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Divorced",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'Result': '>50K'}


def test_get_invalid_url(client):
    """Test invalid url"""
    res = client.get("/invalid_url")
    assert res.status_code == 404
    assert res.json() == {'detail': 'Not Found'}
