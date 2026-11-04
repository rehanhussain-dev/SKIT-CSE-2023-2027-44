import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

RNG_SEED = 42
N_SAMPLES = 500

# Which input features belong to which stress group
GROUPS = {
    "heat_stress": ["lst_celsius", "tmax_feb_mar"],
    "salinity_stress": ["salinity_index"],
    "other": ["ndvi"],
}


def make_mock_table(n=N_SAMPLES, seed=RNG_SEED):
    """Fake stress table where yield follows a formula we know."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        {
            "ndvi": rng.uniform(0.2, 0.9, n),
            "salinity_index": rng.uniform(0.0, 1.0, n),
            "lst_celsius": rng.uniform(25, 45, n),
            "tmax_feb_mar": rng.uniform(22, 40, n),
        }
    )
    noise = rng.normal(0, 0.1, n)
    table["yield_t_ha"] = (
        1.0
        + 3.5 * table["ndvi"]
        - 1.2 * table["salinity_index"]
        - 0.06 * (table["lst_celsius"] - 25)
        - 0.05 * (table["tmax_feb_mar"] - 22)
        + noise
    )
    return table


def stress_shares(model, features):
    """Mean absolute SHAP per feature, grouped and turned into percentages."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0), index=features.columns
    )
    group_totals = {
        name: float(mean_abs[cols].sum()) for name, cols in GROUPS.items()
    }
    total = sum(group_totals.values())
    return {name: round(100 * v / total, 1) for name, v in group_totals.items()}


def stress_split(shares):
    """Heat vs salinity only, rescaled so the two add up to 100."""
    heat = shares["heat_stress"]
    salinity = shares["salinity_stress"]
    total = heat + salinity
    return {
        "heat_percent": round(100 * heat / total, 1),
        "salinity_percent": round(100 * salinity / total, 1),
    }


if __name__ == "__main__":
    table = make_mock_table()
    features = table.drop(columns="yield_t_ha")

    model = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=RNG_SEED
    )
    model.fit(features, table["yield_t_ha"])
    print("Model R2 on training data:", round(model.score(features, table["yield_t_ha"]), 3))

    shares = stress_shares(model, features)
    print("Stress shares (%):", shares)
    print("Sum:", round(sum(shares.values()), 1))
    split = stress_split(shares)
    print("Heat vs salinity (%):", split)