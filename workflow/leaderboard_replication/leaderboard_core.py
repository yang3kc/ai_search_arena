"""
Core leaderboard functions extracted from the shared notebook.
These functions implement Bradley-Terry rating system with style control.
"""

import math
import numpy as np
import pandas as pd
from functools import partial
from scipy.special import expit
from scipy.optimize import minimize


def get_matchups_models(df):
    """Extract model matchups and model list from battle data."""
    n_rows = len(df)
    model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_bt(df):
    """Preprocess data for Bradley-Terry model."""
    n_rows = len(df)
    # the 3 columns of schedule represent: model_a id, model_b id, outcome_id
    schedule = np.full((n_rows, 3), fill_value=1, dtype=np.int32)
    # set the two model cols by mapping the model names to their int ids
    schedule[:, [0, 1]], models = get_matchups_models(df)
    # map outcomes to integers (must be same dtype as model ids so it can be in the same array)
    # model_a win -> 2, tie -> 1 (prefilled by default), model_b win -> 0
    schedule[df["winner"] == "model_a", 2] = 2
    schedule[df["winner"] == "model_b", 2] = 0
    # count the number of occurances of each observed result
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:, [0, 1]]
    # map 2 -> 1.0, 1 -> 0.5, 0 -> 0.0 which will be used as labels during optimization
    outcomes = matchups_outcomes[:, 2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    # each possible result is weighted according to number of times it occured in the dataset
    return matchups, outcomes, models, weights


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    """Bradley-Terry loss function and gradient."""
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)
    # this form naturally counts a draw as half a win and half a loss
    loss = -(
        (np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights
    ).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    # aggregate gradients at the model level using the indices in matchups
    np.add.at(
        model_grad,
        matchups[:, [0, 1]],
        matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64),
    )
    return loss, model_grad


def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
    """Fit Bradley-Terry model."""
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def scale_and_offset(
    ratings,
    models,
    scale,
    init_rating,
    anchor_model_and_rating=None,
):
    """Convert ratings from the natural scale to the Elo rating scale with an anchored baseline."""
    scaled_ratings = (ratings * scale) + init_rating
    if anchor_model_and_rating is not None:
        anchor_model, anchor_rating = anchor_model_and_rating
        baseline_idx = models.index(anchor_model)
        scaled_ratings += anchor_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings


def compute_bt(
    df,
    base=10.0,
    scale=400.0,
    init_rating=1000,
    tol=1e-6,
    anchor_model_and_rating=None,
):
    """Compute Bradley-Terry ratings."""
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(base), tol)
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating, anchor_model_and_rating
    )
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def compute_bootstrap_bt(
    battles,
    num_round,
    base=10.0,
    scale=400.0,
    init_rating=1000.0,
    tol=1e-6,
    num_cpu=None,
    anchor_model_and_rating=None,
    offset=0.0,
):
    """Compute bootstrap Bradley-Terry ratings for confidence intervals."""
    matchups, outcomes, models, weights = preprocess_for_bt(battles)
    # bootstrap sample the unique outcomes and their counts directly using the multinomial distribution
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(battles), pvals=weights / weights.sum(), size=(num_round)
    )
    # only the distribution over their occurance counts changes between samples (and it can be 0)
    boot_weights = idxs.astype(np.float64) / len(battles)

    # the only thing different across samples is the distribution of weights
    bt_fn = partial(
        fit_bt, matchups, outcomes, n_models=len(models), alpha=np.log(base), tol=tol
    )
    results = []
    for weights in boot_weights:
        results.append(bt_fn(weights))

    ratings = np.array(results)
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating + offset, anchor_model_and_rating
    )
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


def preprocess_for_elo(df):
    """Preprocess data for Elo rating (used by style control)."""
    matchups, models = get_matchups_models(df)
    outcomes = np.full(len(df), 0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return matchups, outcomes, models


def preprocess_for_style(df, style_elements, add_one=True):
    """Preprocess data for style control."""
    apply_ratio = list(np.ones(len(style_elements)//2))
    matchups, outcomes, models = preprocess_for_elo(df)

    n = matchups.shape[0]
    k = int(len(style_elements) / 2)

    def extract_style_feature(x, feature):
        val = x[feature]
        if isinstance(val, int) or isinstance(val, float):
            return val
        else:
            return sum(val.values())

    style_vector = np.zeros(shape=(2 * k, n), dtype=np.int32)
    for idx, element in enumerate(style_elements):
        style_vector[idx, :] = df.conv_metadata.map(
            partial(extract_style_feature, feature=element)
        ).values
    style_vector = np.ascontiguousarray(style_vector)

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio = np.flatnonzero(apply_ratio)

    # Apply ratio where necessary (length, etc)
    style_diff[apply_ratio] /= style_sum[apply_ratio]

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    return matchups, features, outcomes, models


DIFF_MASK = np.array([1.0, -1.0], dtype=np.float64)


def contextual_bt_loss_and_grad(
    params,
    n_competitors,
    matchups,
    features,
    outcomes,
    alpha=1.0,
    reg=1.0,
    half_reg=0.5,
):
    """Contextual Bradley-Terry loss function and gradient."""
    reg_loss = half_reg * np.inner(params, params)

    # Split params into ratings and feature parameters
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )

    error = outcomes - probs
    grad = reg * params  # initialize the grad as the regularization grad
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad[n_competitors:] -= np.dot(features.T, error)
    return loss, grad


def fit_contextual_bt(
    matchups,
    features,
    outcomes,
    models,
    idxs=None,
    alpha=math.log(10.0),
    reg=0.5,
    tol=1e-6,
):
    """Fit contextual Bradley-Terry model."""
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    # sample idxs optionally allow for fitting on a bootstrap sample of the dataset
    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]

    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control(
    df,
    style_elements,
    alpha=math.log(10.0),
    reg=0.5,
    init_rating=1000.0,
    scale=400.0,
    tol=1e-6,
    anchor_model_and_rating=None,
):
    """Compute style-controlled Bradley-Terry ratings."""
    matchups, features, outcomes, models = preprocess_for_style(df, style_elements=style_elements)
    ratings_params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = ratings_params[: len(models)]
    params = ratings_params[len(models) :]
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating, anchor_model_and_rating
    )
    scaled_ratings = pd.Series(scaled_ratings, index=models).sort_values(
        ascending=False
    )
    return scaled_ratings, params


def compute_bootstrap_style_control(
    df,
    style_elements,
    num_round,
    alpha=math.log(10.0),
    reg=0.5,
    init_rating=1000.0,
    scale=400.0,
    tol=1e-6,
    num_cpu=None,
    offset=0.0,
    anchor_model_and_rating=None,
):
    """Compute bootstrap style-controlled Bradley-Terry ratings."""
    matchups, features, outcomes, models = preprocess_for_style(df, style_elements=style_elements)

    contextual_bt_fn = partial(
        fit_contextual_bt,
        matchups,
        features,
        outcomes,
        models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )

    np.random.seed(0)
    boot_idxs = np.random.randint(
        low=0, high=matchups.shape[0], size=(num_round, matchups.shape[0])
    )

    results = []
    for idx in boot_idxs:
        results.append(contextual_bt_fn(idx))

    ratings_params = np.array(results)
    ratings = ratings_params[:, : len(models)]
    params = ratings_params[:, len(models) :]
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating + offset, anchor_model_and_rating
    )
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index], params


def run_leaderboard(
    battle_data,
    anchor_model,
    anchor_rating,
    style_elements=None,
    num_bootstrap_samples=100,
):
    """Run leaderboard computation."""
    if style_elements is None:
        bt_ratings = compute_bt(battle_data)
        offset_score = (anchor_rating - bt_ratings[anchor_model])
        bt_ratings += offset_score
        bt_ratings_bootstrap = compute_bootstrap_bt(battle_data, num_round=num_bootstrap_samples, offset=offset_score)
        style_coef_bootstrap = None
    else:
        bt_ratings, _ = compute_style_control(battle_data, style_elements=style_elements)
        offset_score = (anchor_rating - bt_ratings[anchor_model])
        bt_ratings += offset_score
        bt_ratings_bootstrap, style_coef_bootstrap = compute_bootstrap_style_control(
            battle_data, 
            style_elements=style_elements, 
            num_round=num_bootstrap_samples, 
            offset=offset_score
        )

    model_order = list(bt_ratings.keys())
    model_rating_q025 = bt_ratings_bootstrap.quantile(0.025)
    model_rating_q975 = bt_ratings_bootstrap.quantile(0.975)

    # Compute rankings
    ranking = {}
    for i, model_a in enumerate(model_order):
        ranking[model_a] = 1
        for j, model_b in enumerate(model_order):
            if i == j:
                continue
            if model_rating_q025[model_b] > model_rating_q975[model_a]:
                ranking[model_a] += 1

    leaderboard_table = pd.DataFrame(
        {
            "rating": bt_ratings,
            "variance": bt_ratings_bootstrap.var(),
            "rating_q975": bt_ratings_bootstrap.quantile(0.975),
            "rating_q025": bt_ratings_bootstrap.quantile(0.025),
            "num_battles": battle_data["model_a"].value_counts().add(battle_data["model_b"].value_counts(), fill_value=0),
            "final_ranking": pd.Series(ranking),
        }
    )
    leaderboard_table = leaderboard_table.sort_values(by='rating', ascending=False)
    
    return leaderboard_table, bt_ratings_bootstrap, style_coef_bootstrap