def implementation_status():
    return [
        {"method": name, "implementation": detail, "status": "implemented"}
        for name, detail in [
            (
                "random_partial",
                "fingerprint nearest-neighbor acquired-label imputation",
            ),
            ("softimpute", "iterative singular-value shrinkage"),
            ("imc", "fixed-feature linear low-rank factors"),
            ("independent_lgbm", "one LightGBM loss regressor per expert"),
            (
                "fformpp_b",
                "author movingknots.fformpp package, budgeted complete-row labels; install requirements.txt",
            ),
            (
                "pmf_a",
                "GP over learned expert embeddings plus expected-improvement acquisition",
            ),
            (
                "scm_passive",
                "five-member nonlinear Huber/ranking recovery plus random acquisition",
            ),
            (
                "fame_scm_active",
                "five-member recovery, four active batches, masked router and calibration",
            ),
            ("full_observation", "all measured competence plus same trained router"),
            ("lightgbm", "single expert forecast reference"),
            (
                "nn_ei_c",
                "matched five-member Huber recovery without ranking, cost-aware Gaussian expected improvement",
            ),
            (
                "independent_neural",
                "independent nonlinear acquired-label loss regressors",
            ),
        ]
    ]
