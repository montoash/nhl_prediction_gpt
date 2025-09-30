#!/bin/bash
# Manual deployment script to ensure NHL API is deployed to Cloud Run

echo "🏒 Deploying NHL Prediction API to Cloud Run..."

# Set your Google Cloud project ID
PROJECT_ID="nhl-prediction-gpt-455856529947"
REGION="us-central1"
SERVICE_NAME="nhl-prediction-api"

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
  --max-instances 10

echo "✅ Deployment complete!"
echo "🔗 Your NHL API is available at:"
echo "   https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app"

echo "🧪 Testing endpoints..."
echo "Health: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/health"
echo "Teams: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/teams" 
echo "Predict: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/predict?home=TOR&away=BOS"
echo "OpenAPI: https://$SERVICE_NAME-$PROJECT_ID.$REGION.run.app/openapi.json"