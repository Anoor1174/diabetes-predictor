def test_fairness_metrics(client):
    response = client.get("/api/fairness_metrics")
    assert response.status_code == 200

    data = response.get_json()
    assert "overall_recall" in data
    assert "overall_accuracy" in data
    assert "groups" in data



def test_performance_fairness_comparison(client):
    response = client.get("/api/performance_fairness_comparison")
    assert response.status_code == 200

    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
