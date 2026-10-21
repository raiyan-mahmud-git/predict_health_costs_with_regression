from health_costs.training import evaluate_classical_cv


def test_reference_models_run_grouped_cross_validation(sample_data):
    results = evaluate_classical_cv(sample_data, n_splits=5)
    assert set(results["model"]) == {
        "mean_baseline",
        "linear_with_interaction",
        "gradient_boosting_with_interaction",
        "gradient_boosting_log_target",
        "gradient_boosting_without_interaction",
    }
    assert (results["folds"] == 5).all()
    assert results["mae_mean"].notna().all()
    assert (results["mae_std"] >= 0).all()
