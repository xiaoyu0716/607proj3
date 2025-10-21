import numpy as np
import pandas as pd

def evaluate(data, setting="A"):
    """
    calculate the evaluation metrics
    parameters:
        data: pd.DataFrame
        
    metrics:
        Bias:
        Variance
        IQR
    returns:
        metrics: dict
    """
    # Accept data as: sequence of estimates, or a DataFrame with column 'estimate'
    if isinstance(data, pd.DataFrame):
        if "estimate" not in data.columns:
            raise ValueError("When passing a DataFrame, it must contain an 'estimate' column.")
        estimates = data["estimate"].to_numpy()
    elif isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        estimates = np.asarray(data, dtype=float)
    else:
        raise ValueError("'data' must be a sequence of estimates or a DataFrame with 'estimate' column.")

    # Determine truth: allow passing a scalar truth or the truth dict from generate_data
    truth = None
    if isinstance(setting, (int, float, np.floating)):
        truth = float(setting)
    elif isinstance(setting, dict):
        if "mu_true" in setting:
            truth = float(setting["mu_true"])
        elif "ATE_true" in setting:
            truth = float(setting["ATE_true"])
        else:
            raise ValueError("Truth dict must contain 'mu_true' or 'ATE_true'.")
    elif isinstance(setting, str):
        # If a plain setting string was provided without truth, we cannot compute bias.
        raise ValueError("Please provide the true value (float) or the truth dict as the second argument.")
    else:
        raise ValueError("Unsupported type for 'setting'. Provide a float truth or a truth dict.")

    estimates = np.asarray(estimates, dtype=float)
    if estimates.ndim != 1:
        raise ValueError("'data' (estimates) must be one-dimensional.")
    if estimates.size == 0:
        raise ValueError("No estimates provided.")

    bias = float(np.mean(estimates - truth))
    variance = float(np.var(estimates, ddof=1)) if estimates.size > 1 else 0.0
    q75, q25 = np.percentile(estimates, [75, 25])
    iqr = float(q75 - q25)

    return {"bias": bias, "variance": variance, "IQR": iqr}