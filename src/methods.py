import numpy as np
import pandas as pd

def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(X.shape[0]), X]

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    # small ridge to improve stability
    ridge = 1e-8 * np.eye(XtX.shape[0])
    beta = np.linalg.solve(XtX + ridge, X.T @ y)
    return beta

def _logit_fit_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = _sigmoid(eta)
        w = p * (1.0 - p)
        # guard against zeros
        w = np.clip(w, 1e-8, None)
        z = eta + (y - p) / w
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWz = X.T @ W @ z
        # ridge for stability
        ridge = 1e-6 * np.eye(X.shape[1])
        beta_new = np.linalg.solve(XtWX + ridge, XtWz)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta

# Generic, setting-agnostic estimators and wrapper

def _build_propensity_design(data: pd.DataFrame, setting: str, spec: str = "true"):
    # Indicator features Ij = 1(Vj > 0)
    V1 = data["V1"].to_numpy()
    V2 = data["V2"].to_numpy()
    V3 = data["V3"].to_numpy()
    V4 = data["V4"].to_numpy()
    I1 = (V1 > 0).astype(float)
    I2 = (V2 > 0).astype(float)
    I3 = (V3 > 0).astype(float)
    I12 = I1 * I2

    if setting == "A":
        # True: [1, I1, I2, I3, I1*I2]
        # False: [I1, I3] (no intercept)
        if spec == "true":
            X = np.c_[np.ones(len(V1)), I1, I2, I3, I12]
        else:
            X = np.c_[I1, I3]
        y = data["R"].to_numpy()
        return X, y
    elif setting == "B":
        # True: same as A true
        # False: [I3, V4] (no intercept)
        if spec == "true":
            X = np.c_[np.ones(len(V1)), I1, I2, I3, I12]
        else:
            X = np.c_[I3, V4]
        y = data["A"].to_numpy()
        return X, y
    else:
        raise ValueError("setting must be 'A' or 'B'")


def _build_outcome_design(data: pd.DataFrame, setting: str, spec: str = "true"):
    V1 = data["V1"].to_numpy()
    V2 = data["V2"].to_numpy()
    V3 = data["V3"].to_numpy()
    if setting == "A":
        # True: [1, V1^2, V2, V2*V3]
        # False: [1, V1, V2^2]
        df_obs = data.loc[data["R"] == 1]
        V1o = df_obs["V1"].to_numpy()
        V2o = df_obs["V2"].to_numpy()
        V3o = df_obs["V3"].to_numpy()
        Yo = df_obs["Y"].to_numpy()
        if spec == "true":
            X_obs = np.c_[np.ones(len(V1o)), V1o**2, V2o, V2o * V3o]
            X_all = np.c_[np.ones(len(V1)), V1**2, V2, V2 * V3]
        else:
            X_obs = np.c_[np.ones(len(V1o)), V1o, V2o**2]
            X_all = np.c_[np.ones(len(V1)), V1, V2**2]
        return X_obs, Yo, X_all
    elif setting == "B":
        # True: [1, V1^2*Δ, V2*Δ, V2*V3*(1-Δ), V3*(1-Δ)]
        # False: [1, V1*Δ, V2^2*(1-Δ)]
        A = data["A"].to_numpy().reshape(-1, 1)
        one_minus_A = 1.0 - A
        Y = data["Y"].to_numpy()
        if spec == "true":
            X = np.c_[np.ones(len(V1)), (V1**2).reshape(-1,1)*A, V2.reshape(-1,1)*A, (V2*V3).reshape(-1,1)*one_minus_A, V3.reshape(-1,1)*one_minus_A]
            X1 = np.c_[np.ones(len(V1)), (V1**2).reshape(-1,1), V2.reshape(-1,1), np.zeros((len(V1),1)), np.zeros((len(V1),1))]
            X0 = np.c_[np.ones(len(V1)), np.zeros((len(V1),1)), np.zeros((len(V1),1)), (V2*V3).reshape(-1,1), V3.reshape(-1,1)]
        else:
            X = np.c_[np.ones(len(V1)), V1.reshape(-1,1)*A, (V2**2).reshape(-1,1)*one_minus_A]
            X1 = np.c_[np.ones(len(V1)), V1.reshape(-1,1), np.zeros((len(V1),1))]
            X0 = np.c_[np.ones(len(V1)), np.zeros((len(V1),1)), (V2**2).reshape(-1,1)]
        return X, Y, X1, X0
    else:
        raise ValueError("setting must be 'A' or 'B'")


def ipw_estimate(data: pd.DataFrame, setting: str, propensity_spec: str = "true") -> float:
    eps = 1e-6
    Xp, y = _build_propensity_design(data, setting, spec=propensity_spec)
    alpha_hat = _logit_fit_irls(Xp, y)
    pi_hat = _sigmoid(Xp @ alpha_hat)
    pi_hat = np.clip(pi_hat, eps, 1 - eps)
    if setting == "A":
        R = data["R"].to_numpy()
        Y = data["Y"].to_numpy()
        return float(np.mean(R * Y / pi_hat))
    else:
        A = data["A"].to_numpy()
        Y = data["Y"].to_numpy()
        return float(np.mean(A * Y / pi_hat - (1 - A) * Y / (1 - pi_hat)))


def or_estimate(data: pd.DataFrame, setting: str, outcome_spec: str = "true") -> float:
    if setting == "A":
        Xo, Yo, Xall = _build_outcome_design(data, setting, spec=outcome_spec)
        beta_hat = _ols_fit(Xo, Yo)
        s_hat = Xall @ beta_hat
        return float(np.mean(s_hat))
    else:
        X, y, X1, X0 = _build_outcome_design(data, setting, spec=outcome_spec)
        beta_hat = _ols_fit(X, y)
        s1 = X1 @ beta_hat
        s0 = X0 @ beta_hat
        return float(np.mean(s1 - s0))


def dr_estimate(data: pd.DataFrame, setting: str, outcome_spec: str = "true", propensity_spec: str = "true") -> float:
    eps = 1e-6
    Xp, yp = _build_propensity_design(data, setting, spec=propensity_spec)
    alpha_hat = _logit_fit_irls(Xp, yp)
    pi_hat = _sigmoid(Xp @ alpha_hat)
    pi_hat = np.clip(pi_hat, eps, 1 - eps)

    if setting == "A":
        Xo, Yo, Xall = _build_outcome_design(data, setting, spec=outcome_spec)
        beta_hat = _ols_fit(Xo, Yo)
        s_hat = Xall @ beta_hat
        Y = data["Y"].to_numpy()
        R = data["R"].to_numpy()
        return float(np.mean(s_hat + R * (Y - s_hat) / pi_hat))
    else:
        X, y, X1, X0 = _build_outcome_design(data, setting, spec=outcome_spec)
        beta_hat = _ols_fit(X, y)
        s1 = X1 @ beta_hat
        s0 = X0 @ beta_hat
        A = data["A"].to_numpy()
        Y = data["Y"].to_numpy()
        term1 = A * (Y - s1) / pi_hat
        term0 = (1 - A) * (Y - s0) / (1 - pi_hat)
        return float(np.mean(term1 - term0 + s1 - s0))


def run_all_methods(
    data: pd.DataFrame,
    setting: str,
    beta_true: np.ndarray,
    alpha_true: np.ndarray,
    beta_false: np.ndarray,
    alpha_false: np.ndarray,
) -> dict:
    results: dict[str, dict] = {}

    # IPW true/false (alpha only)
    for spec, name, alpha_used in [("true", "IPW", alpha_true), ("false", "IPW.fal", alpha_false)]:
        est = ipw_estimate(data, setting, propensity_spec=spec)
        results[name] = {
            "estimate": float(est),
            "alpha": alpha_used.tolist(),
            "beta": None,
            "propensity_spec": spec,
            "outcome_spec": None,
        }

    # OR true/false (beta only)
    for spec, name, beta_used in [("true", "OR", beta_true), ("false", "OR.fal", beta_false)]:
        est = or_estimate(data, setting, outcome_spec=spec)
        results[name] = {
            "estimate": float(est),
            "alpha": None,
            "beta": beta_used.tolist(),
            "propensity_spec": None,
            "outcome_spec": spec,
        }

    # DR combinations (both beta and alpha)
    combos = [
        ("true", "true", "DR", beta_true, alpha_true),
        ("false", "true", "DR.ofal", beta_false, alpha_true),
        ("true", "false", "DR.pfal", beta_true, alpha_false),
        ("false", "false", "DR.o&pfal", beta_false, alpha_false),
    ]
    for o_spec, p_spec, name, beta_used, alpha_used in combos:
        est = dr_estimate(data, setting, outcome_spec=o_spec, propensity_spec=p_spec)
        results[name] = {
            "estimate": float(est),
            "alpha": alpha_used.tolist(),
            "beta": beta_used.tolist(),
            "propensity_spec": p_spec,
            "outcome_spec": o_spec,
        }

    return results
 