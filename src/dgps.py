import numpy as np
import pandas as pd
from scipy.special import expit  # logistic / inverse-logit

def generate_data(n=1000, setting="A", seed=None, beta=None, alpha=None):
    """
    Generate simulated data for Bang & Robins (2005):
      A = cross-sectional missing-data model  (E[Y])
      B = treatment-effect model (ATE)

    Parameters
    ----------
    n : int
        sample size
    setting : str
        'A' -> Nonlongitudinal (missing data)
        'B' -> Treatment-effect
    seed : int or None
        random seed
    beta : array-like
        outcome model parameters (must match setting spec)
    alpha : array-like
        propensity/missingness model parameters (must match setting spec)

    Returns
    -------
    pd.DataFrame
        simulated dataset
    dict
        dictionary of true parameter values (e.g., true mean or true ATE)
    np.ndarray
        beta used
    np.ndarray
        alpha used
    """
    if seed is not None:
        np.random.seed(seed)

    # four covariates V1–V4 ~ N(0,1)
    V = np.random.normal(size=(n, 4))
    V1, V2, V3, V4 = V.T
    I1 = (V1 > 0).astype(float)
    I2 = (V2 > 0).astype(float)
    I3 = (V3 > 0).astype(float)
    I12 = I1 * I2

    if setting == "A":
        # True spec: s(V;β) = β·[1, V1^2, V2, V2*V3]', logit{π(V;α)} = α·[1, I1, I2, I3, I1*I2]'
        if beta is None:
            beta = np.array([0.0, 1.0, 2.5, 3.0])
        if alpha is None:
            alpha = np.array([-1.0, 1.0, 0.0, 0.0, -1.0])
        beta = np.asarray(beta, dtype=float)
        alpha = np.asarray(alpha, dtype=float)

        # outcome model: Y = β0 + β1*V1^2 + β2*V2 + β3*V2*V3 + ε
        Y = beta[0] + beta[1]*(V1**2) + beta[2]*V2 + beta[3]*(V2*V3) + np.random.normal(size=n)
        # E[Y] = β0 + β1*E[V1^2] + β2*E[V2] + β3*E[V2*V3]
        #      = β0 + β1*1 + β2*0 + β3*0  (since V~N(0,1), E[V^2]=1, E[V2*V3]=E[V2]*E[V3]=0)
        mu_true = beta[0] + beta[1]

        # missingness model: logit P(R=1|V) = α0 + α1*I1 + α2*I2 + α3*I3 + α4*I1*I2
        logit_pi = alpha[0] + alpha[1]*I1 + alpha[2]*I2 + alpha[3]*I3 + alpha[4]*I12
        pi = expit(logit_pi)
        R = np.random.binomial(1, pi, size=n)

        data = pd.DataFrame(dict(V1=V1, V2=V2, V3=V3, V4=V4, Y=Y, R=R))
        truth = {"mu_true": mu_true}
        return data, truth, beta, alpha

    elif setting == "B":
        # True spec: s(Δ,V;β) = β·[1, V1^2*Δ, V2*Δ, V2*V3*(1-Δ), V3*(1-Δ)]'
        #            logit{π(V;α)} = α·[1, I1, I2, I3, I1*I2]'
        if beta is None:
            beta = np.array([0.0, 2.0, 3.0, 2.0, -4.0])
        if alpha is None:
            alpha = np.array([-3.0, 2.5, 3.0, 1.0, -3.0])
        beta = np.asarray(beta, dtype=float)
        alpha = np.asarray(alpha, dtype=float)

        # treatment assignment: logit P(A=1|V) = α0 + α1*I1 + α2*I2 + α3*I3 + α4*I1*I2
        logit_pi = alpha[0] + alpha[1]*I1 + alpha[2]*I2 + alpha[3]*I3 + alpha[4]*I12
        pi = expit(logit_pi)
        A = np.random.binomial(1, pi, size=n)

        # outcome model: Y = β0 + β1*V1^2*A + β2*V2*A + β3*V2*V3*(1-A) + β4*V3*(1-A) + ε
        Y = (beta[0] + beta[1]*(V1**2)*A + beta[2]*V2*A 
             + beta[3]*(V2*V3)*(1-A) + beta[4]*V3*(1-A) + np.random.normal(size=n))

        # potential outcomes for truth
        # E[Y(1)] = β0 + β1*E[V1^2] + β2*E[V2] = β0 + β1*1 + β2*0 = β0 + β1
        # E[Y(0)] = β0 + β3*E[V2*V3] + β4*E[V3] = β0 + β3*0 + β4*0 = β0
        # ATE = E[Y(1) - Y(0)] = β1
        ATE_true = beta[1]

        data = pd.DataFrame(dict(A=A, V1=V1, V2=V2, V3=V3, V4=V4, Y=Y))
        truth = {"ATE_true": ATE_true}
        return data, truth, beta, alpha

    else:
        raise ValueError("setting must be 'A' or 'B'")