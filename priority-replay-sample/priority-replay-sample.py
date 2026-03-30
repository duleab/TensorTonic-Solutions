def priority_replay_sample(priorities, alpha, beta):
    N = len(priorities)
    # Compute powered priorities
    powered_priorities = [p ** alpha for p in priorities]
    total_priority = sum(powered_priorities)
    
    # Compute sampling probabilities
    probs = [p / total_priority for p in powered_priorities]
    
    # Compute importance sampling weights: w_i = (N * P(i))^-beta
    weights = [(N * p) ** (-beta) for p in probs]
    
    # Normalize weights by the maximum weight
    max_w = max(weights)
    norm_weights = [w / max_w for w in weights]
    
    return [probs, norm_weights]