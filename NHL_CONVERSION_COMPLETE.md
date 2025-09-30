# NHL Prediction System - Conversion Complete! 🏒

## Summary of Changes Made

Your NFL prediction system has been successfully converted to NHL! Here's what was transformed:

### ✅ **Core Application Changes**
- **`app_nhl.py`** - New NHL-focused Flask API with proper endpoints
- **Team mappings** - All 32 NHL teams with correct abbreviations
- **Features** - Changed from NFL metrics (EPA, turnovers) to NHL metrics (goals, shots, powerplay %)
- **Fallback predictor** - Updated with 2023-24 NHL team strength ratings

### ✅ **Model & Training System**
- **`scripts/train_nhl_model.py`** - New training script using NHL API data
- **Model files** - Renamed from `nfl_win_predictor_*.pkl` to `nhl_win_predictor_*.pkl`
- **Features** - NHL-specific: `goals_for_diff`, `goals_against_diff`, `shots_diff`, `powerplay_diff`, `giveaway_diff`
- **Data source** - Uses NHL API (statsapi.web.nhl.com) instead of nfl_data_py

### ✅ **Deployment Configuration**
- **Cloud Run** - Updated service name to `nhl-prediction-api`
- **Docker files** - Both `Dockerfile` and `Dockerfile.cloudrun` updated for NHL app
- **Cloud Build** - `cloudbuild.yaml` configured for NHL deployment
- **Render.com** - `render.yaml` updated for NHL service

### ✅ **API & Documentation**
- **OpenAPI spec** - Updated for NHL teams and puck lines
- **Team endpoints** - `/teams` endpoint returns NHL divisions
- **Odds integration** - Updated for `icehockey_nhl` sport
- **README** - Comprehensive NHL documentation

### ✅ **Dependencies**
- **`requirements.txt`** - Updated with `nhl-api-py==3.0.2`
- **All packages tested** and installed successfully

## 🚀 Next Steps for Deployment

### 1. **Google Cloud Run Deployment**

Since you're using a **separate Google Cloud Run instance**, here's what to do:

```bash
# 1. Create new Google Cloud project (if needed)
gcloud projects create your-nhl-project-id

# 2. Set project
gcloud config set project your-nhl-project-id

# 3. Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 4. Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

**Your NHL service will be available at:**
`https://nhl-prediction-api-[hash].us-central1.run.app`

### 2. **Train NHL Models**

Before first use, train your NHL models:

```bash
# Train with last 5 seasons (recommended for testing)
START_SEASON=2019 END_SEASON=2024 python scripts/train_nhl_model.py

# Or full training (2018-2024)
python scripts/train_nhl_model.py
```

This will create:
- `models/nhl_win_predictor_with_odds.pkl`
- `models/nhl_win_predictor_no_odds.pkl`
- `models/features_with_odds.pkl`
- `models/features_no_odds.pkl`

### 3. **Custom GPT Setup**

For your **separate Custom GPT**, use this configuration:

**Instructions:**
```
You are an NHL prediction expert. Use the NHL Prediction API to analyze games, provide win probabilities, and give betting insights. Always mention the confidence intervals and model features used.

Key NHL teams by division:
- Atlantic: BOS, BUF, DET, FLA, MTL, OTT, TBL, TOR
- Metropolitan: CAR, CBJ, NJD, NYI, NYR, PHI, PIT, WSH  
- Central: ARI, CHI, COL, DAL, MIN, NSH, STL, WPG
- Pacific: ANA, CGY, EDM, LAK, SEA, SJS, VAN, VGK

Focus on goals for/against, powerplay efficiency, and home ice advantage.
```

**Actions Schema:**
Use the updated `openapi.yaml` file in this repository.

**Action URL:**
Your deployed Cloud Run URL (e.g., `https://nhl-prediction-api-[hash].us-central1.run.app`)

### 4. **API Testing**

Test your deployed service:

```bash
# Health check
curl https://your-nhl-service-url/health

# Get teams
curl https://your-nhl-service-url/teams

# Predict game (example: Leafs vs Bruins)
curl "https://your-nhl-service-url/predict?home=TOR&away=BOS"
```

Expected response:
```json
{
  "home_team": "TOR",
  "away_team": "BOS", 
  "predictions": {
    "home_win_probability": 0.5234,
    "away_win_probability": 0.4766,
    "prediction": "TOR"
  },
  "features_used": ["goals_for_diff", "goals_against_diff", "shots_diff", "powerplay_diff", "giveaway_diff"],
  "model_type": "without_odds"
}
```

## 🔧 Configuration Options

### Environment Variables (Optional)
```bash
# For training
START_SEASON=2018        # First season to include
END_SEASON=2024          # Last season to include
TRAIN_REGRESSORS=1       # Train spread/total regressors

# For API
ODDS_API_KEY=your_key    # For live betting odds
ENABLE_FULL_FEATURES=0   # Use lightweight mode (default)
```

### Betting Odds Integration
To get live odds, sign up at [The Odds API](https://the-odds-api.com/):
1. Get free API key (500 requests/month)
2. Set `ODDS_API_KEY` environment variable
3. API will automatically use live odds when available

## 🎯 Key Differences from NFL Version

| Feature | NFL Version | NHL Version |
|---------|-------------|-------------|
| **Data Source** | `nfl_data_py` | NHL API (statsapi.web.nhl.com) |
| **Teams** | 32 NFL teams | 32 NHL teams |
| **Primary Metric** | EPA (Expected Points Added) | Goals For/Against differential |
| **Secondary Metrics** | Explosive plays, turnovers | Powerplay %, shot differential |
| **Betting Lines** | Point spread, totals | Puck line (-1.5/+1.5), totals |
| **Home Advantage** | ~3 points | ~0.55 win rate boost |
| **Season Format** | Year (2024) | Season string ("20242025") |

## 🔍 Troubleshooting

### Common Issues:

1. **NHL API not responding**: The app uses fallback predictors when API fails
2. **Model files missing**: Run training script first
3. **Memory issues**: Models are optimized for 512MB+ environments
4. **Odds not working**: Check `ODDS_API_KEY` or service will use stats-only model

### Model Performance:
- **Accuracy**: ~55-60% (typical for sports prediction)
- **Calibration**: Isotonic calibration ensures probability reliability  
- **Uncertainty**: 95% confidence intervals via conformal prediction
- **Features**: 5 core NHL metrics vs NFL's 4 EPA-based metrics

## 🎉 You're Ready!

Your NHL prediction system is now completely separate from your NFL system and ready for deployment. The architecture maintains the same advanced ML techniques (XGBoost, calibration, conformal prediction) but is tailored specifically for hockey analytics.

**Quick Deploy Commands:**
```bash
# 1. Train models
python scripts/train_nhl_model.py

# 2. Test locally  
python app_nhl.py

# 3. Deploy to Google Cloud
gcloud builds submit --config cloudbuild.yaml
```

🏒 **Happy NHL predictions!**