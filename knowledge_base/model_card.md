# NHL Win Prediction Model Card

**Model Purpose**: To provide a calibrated win probability and an uncertainty-aware prediction set for NHL matchups.

**Model Type**: XGBoost Classifier wrapped with a MAPIE Conformal Prediction layer.

**Key Features Used**:
- `epa_diff`: The difference in the 3-game rolling average of Offensive EPA per play between the home and away team.
- `success_diff`: The difference in the 3-game rolling average of offensive success rate between the home and away team.

**Evaluation Metric**: Brier Score. This measures the accuracy of probabilistic predictions. Lower is better.

**Uncertainty Method**: Conformal Prediction (`alpha=0.1`). This generates a "prediction set" of plausible outcomes with 90% statistical validity, meaning the true outcome should fall within the set 90% of the time over the long run.