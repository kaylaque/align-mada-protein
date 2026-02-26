import numpy as np

def ndcg_at_10(predicted_values: list, experimental_values: list, top_fraction: float = 0.1) -> float:
    """
    Compute NDCG@10% given predicted and experimental value lists.
    
    Args:
        predicted_values: List of predicted scores/values (one per variant, in original order)
        experimental_values: List of ground truth experimental values (same order)
        top_fraction: Fraction of top variants to consider (default 0.1 = top 10%)
    
    Returns:
        NDCG score between 0.0 and 1.0
    """
    predicted_values = np.array(predicted_values, dtype=float)
    experimental_values = np.array(experimental_values, dtype=float)
    
    assert len(predicted_values) == len(experimental_values), "Lists must be the same length"
    
    n_total = len(experimental_values)
    k = max(1, int(n_total * top_fraction))  # top 10% cutoff

    # Min-max normalize experimental values to get gains
    v_min, v_max = experimental_values.min(), experimental_values.max()
    if v_max == v_min:
        return 0.0
    gains = (experimental_values - v_min) / (v_max - v_min)

    # DCG: rank variants by predicted values (descending), sum gains of top-k
    predicted_order = np.argsort(predicted_values)[::-1]
    dcg = sum(
        gains[idx] / np.log2(i + 2)  # i+2 because i is 0-indexed (log2(i+1+1))
        for i, idx in enumerate(predicted_order[:k])
    )

    # IDCG: rank variants by experimental values (descending) — perfect oracle
    ideal_order = np.argsort(experimental_values)[::-1]
    idcg = sum(
        gains[idx] / np.log2(i + 2)
        for i, idx in enumerate(ideal_order[:k])
    )

    return 0.0 if idcg == 0 else dcg / idcg