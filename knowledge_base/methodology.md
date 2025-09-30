# Methodology Guide

**Calibrated Probability**: This is not just a raw model score. The model output is adjusted to ensure that when it predicts 70%, it's correct about 70% of the time. This makes the probability trustworthy.

**Conformal Prediction Set**: Instead of just one prediction, this gives a *set* of possible outcomes (e.g., {"Home Win"}, or {"Home Win", "Away Win"}). It is generated with a 90% confidence guarantee. If the set contains both outcomes, it means the model is not confident enough to exclude either possibility. If it contains only one outcome, the model is highly confident.

**Data Source**: All game data is sourced from the NHL API (https://statsapi.web.nhl.com), which provides comprehensive team statistics and game results.