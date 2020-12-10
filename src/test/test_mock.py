import pytest
import random
from itertools import product
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK , HTTP_422_UNPROCESSABLE_ENTITY
from src import main
from src.main import get_model, n_features
from src.model.mock_model import MockModel

client = TestClient(main.app)

def get_model_override():
    clf = MockModel()
    return clf

main.app.dependency_overrides[get_model] = get_model_override


@pytest.fixture()
def test_client():
    return TestClient(main.app)

@pytest.mark.parametrize("n_instances", range(1,10))
def test_predict(n_instances: int, test_client:TestClient):
    fake_data = [[random.random() for _ in range(n_features)] for _ in range(n_instances)]
    response = test_client.post("/predict", json={"data": fake_data})
    assert response.status_code == HTTP_200_OK
    assert len(response.json()["data"]) == n_instances

@pytest.mark.parametrize("n_instances, test_data_n_features", product(range(1,10), [n for n in range(1,20) if n != n_features]),)
def test_predict_with_wrong_input(n_instances:int, test_data_n_features:int, test_client: TestClient):
    fake_data = [[random.random() for _ in range(test_data_n_features)] for _ in range(n_instances)]
    response = test_client.post("/predict", json={"data": fake_data})
    assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY






