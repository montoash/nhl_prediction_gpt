#!/bin/bash
# Manual deployment script to ensure NHL API is deployed to Cloud Run

echo "🏒 Deploying NHL Prediction API to Cloud Run..."

# Set your Google Cloud project ID
PROJECT_ID="nhl-prediction-gpt-455856529947"
REGION="us-central1"
# Align with the live service name used in the URL: https://nhl-prediction-gpt-455856529947.us-central1.run.app
SERVICE_NAME="nhl-prediction-gpt"

echo "📦 Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME -f Dockerfile.cloudrun .

echo "🚀 Pushing to Container Registry..."
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME

echo "☁️  Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars GIT_SHA=$(git rev-parse --short HEAD),APP_VARIANT=NHL \
  --set-env-vars ODDS_SPORT_KEYS="icehockey_nhl,icehockey_nhl_preseason" \
  ${ODDS_API_KEY:+--set-env-vars ODDS_API_KEY=$ODDS_API_KEY}

echo "✅ Deployment complete!"
echo "🔗 Your NHL API is available at:"
echo "   https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app"

echo "🧪 Testing endpoints..."
echo "Health: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/health"
echo "Teams: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/teams" 
echo "Predict: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/predict?home=TOR&away=BOS"
echo "OpenAPI: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/openapi.json"