import pandas as pd


def parse_owners(value):
    try:
        low, high = value.replace(",", "").split("-")
        return (int(low) + int(high)) / 2
    except Exception:
        return None


def parse_multilabel_series(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(lambda items: [item.strip() for item in items if item.strip() != ""])
    )


def get_frequent_labels(series, min_frequency=100):
    parsed = parse_multilabel_series(series)

    counts = {}

    for labels in parsed:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    return sorted([
        label for label, count in counts.items()
        if count >= min_frequency
    ])


def encode_with_fixed_labels(series, labels, prefix):
    parsed = parse_multilabel_series(series).apply(set)

    encoded = pd.DataFrame(
        {
            f"{prefix}_{label}": parsed.apply(lambda x: int(label in x))
            for label in labels
        },
        index=series.index
    )

    return encoded
