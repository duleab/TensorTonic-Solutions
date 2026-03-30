def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    """
    return [
        (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * global_mean
        for r, v in items
    ]