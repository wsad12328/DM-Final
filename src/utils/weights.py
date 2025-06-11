from collections import Counter

def compute_per_instance_weights(Y):
    """Compute sample weights for each instance to balance classes in the fold."""
    class_counter = Counter(Y)
    max_class_count = max(class_counter.values())
    return Y.map(lambda class_label: max_class_count / class_counter[class_label])